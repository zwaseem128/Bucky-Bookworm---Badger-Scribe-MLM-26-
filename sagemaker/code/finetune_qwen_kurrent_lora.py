"""LoRA fine-tune of the Qwen3-VL-4B Kurrent adapter on Kade letter lines.
Runs as a SageMaker training job, launched from notebooks/sagemaker_kade_lora.ipynb.

What it does, in order:
  1. Load Qwen3-VL-4B and the Kurrent adapter (dh-unibe/qwen3vl-german-xix-v2),
     and merge the adapter into the model, so its Kurrent reading is built in.
  2. Add a new, small LoRA adapter on top. Only that one is trained.
  3. Score the held-out documents before training (= the Kurrent adapter as it is).
  4. Train on the Kade line crops, scoring the held-out documents after every
     epoch and keeping the adapter from the best epoch (or the last one, --keep last).
  5. Optionally (--predict_test 1) read the Kaggle test pages with the kept
     adapter and write a predictions file for them.

SageMaker passes hyperparameters as command-line flags and mounts the `train`
channel at $SM_CHANNEL_TRAIN. prepare_kade_lines.py builds that folder:

    lines.zip         kade_lines/<page_id>/000.png, 001.png, ...  (kraken crops)
    train_lines.csv   page_id, doc_id, crop, text   one row per training line
    dev_pages.csv     page_id, doc_id, text         pages that can be scored, with full labels
    test_pages.csv    page_id                       Kaggle test pages that have crops

A document is either trained on or scored, never both: the lines of the
--holdout_docs documents are dropped from training, and every document in
dev_pages.csv that isn't trained on is scored.

What gets written to $SM_MODEL_DIR (SageMaker tars it into model.tar.gz):

    adapter_config.json, adapter_model.safetensors  the kept adapter (best epoch, or last with --keep last)
    final_adapter/                                  the last epoch's adapter
    eval_epoch<N>.csv                               per-page CER (epoch 0 = before training)
    train_log.json, train_summary.json              loss and CER over time
    kade_test_predictions.csv                       page_id, text   (only with --predict_test 1)

To use the result you need all three layers: Qwen3-VL-4B, the Kurrent adapter
merged into it, then this adapter on top (see load_for_inference below).
"""

import argparse
import json
import os
import random
import shutil
import time
import zipfile

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from torch.utils.data import Dataset

from kurrent_common import (
    BASE_MODEL,
    MAX_PIXELS,
    MIN_PIXELS,
    START_ADAPTER,
    full_messages,
    load_line_image,
    page_cer,
    strip_postmark,
    user_messages,
)


def parse_args():
    """Every knob the notebook can set arrives here as a command-line flag."""
    p = argparse.ArgumentParser()
    # --- what to start from ---
    # A local folder works as well as a Hugging Face name. If the notebook gives
    # the job "base" or "adapter" channels (copies staged in S3), they win.
    p.add_argument("--base_model", type=str, default=os.environ.get("SM_CHANNEL_BASE", BASE_MODEL))
    p.add_argument("--start_adapter", type=str, default=os.environ.get("SM_CHANNEL_ADAPTER", START_ADAPTER),
                   help="adapter merged in before training; 'none' to start from plain Qwen3-VL")
    # --- the new adapter ---
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    # --- training ---
    p.add_argument("--epochs", type=float, default=3.0, help="0 = only score the dev pages (a baseline run)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4, help="lines per step")
    p.add_argument("--grad_accum", type=int, default=4, help="effective batch = batch_size * grad_accum")
    p.add_argument("--min_pixels", type=int, default=MIN_PIXELS)
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--max_train_lines", type=int, default=0, help="0 = all; small (e.g. 64) for a smoke test")
    p.add_argument("--holdout_docs", type=str, default="kade_007",
                   help="documents kept out of training and scored instead (comma-separated); "
                        "'none' trains on them, for the final model")
    p.add_argument("--keep", type=str, default="best", choices=["best", "last"],
                   help="which epoch's adapter to keep; 'last' for a final run with nothing held out to pick by")
    # --- scoring ---
    p.add_argument("--dev_pages", type=int, default=0, help="pages scored each epoch; 0 = all")
    p.add_argument("--eval_batch", type=int, default=16, help="lines transcribed at once while scoring")
    p.add_argument("--max_new_tokens", type=int, default=128, help="a line is ~40 tokens; this stops loops")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--predict_test", type=int, default=0, help="1 = also read the Kaggle test pages at the end")
    # --- SageMaker paths ---
    p.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    # SageMaker syncs this folder to S3 while the job runs (when the estimator
    # sets checkpoint_s3_uri), so a stopped job can pick up where it was.
    p.add_argument("--checkpoint_dir", type=str, default="/opt/ml/checkpoints")
    return p.parse_args()


def pick_dtype():
    """bf16 on GPUs that have it (A10G on g5, L4 on g6). The T4 on g4dn doesn't,
    so it trains in fp16. CPU (only for testing the script) uses fp32."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class LineDataset(Dataset):
    """One training example per line: the crop, and its line of the label."""

    def __init__(self, df, root):
        self.rows = df.to_dict("records")
        self.root = root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"image": load_line_image(os.path.join(self.root, r["crop"])), "answer": r["text"]}


class Collator:
    """Turn a batch of lines into tensors, and hide everything except the
    answer from the loss: the model is graded only on what it writes."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        proc = self.processor
        proc.tokenizer.padding_side = "right"
        images = [b["image"] for b in batch]
        full = [proc.apply_chat_template(full_messages(b["answer"]), tokenize=False) for b in batch]
        prompt = proc.apply_chat_template(user_messages(), tokenize=False, add_generation_prompt=True)
        enc = proc(text=full, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, img in enumerate(images):
            # How many tokens the prompt takes once this image's tokens are in.
            n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        return enc


# ---------------------------------------------------------------------------
# Scoring: transcribe every line of a dev page, join them, compare with the label
# ---------------------------------------------------------------------------
@torch.no_grad()
def transcribe_lines(model, processor, images, max_new_tokens, batch):
    processor.tokenizer.padding_side = "left"
    prompt = processor.apply_chat_template(user_messages(), tokenize=False, add_generation_prompt=True)
    out = []
    for s in range(0, len(images), batch):
        chunk = images[s:s + batch]
        enc = processor(text=[prompt] * len(chunk), images=chunk, padding=True, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        out += [t.strip() for t in processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    return out


def read_page(model, processor, root, page_id, args):
    """Read every line crop of a page, top to bottom, one line per row."""
    folder = os.path.join(root, "kade_lines", page_id)
    crops = sorted(f for f in os.listdir(folder) if f.endswith(".png")) if os.path.isdir(folder) else []
    lines = transcribe_lines(model, processor, [load_line_image(os.path.join(folder, c)) for c in crops],
                             args.max_new_tokens, args.eval_batch)
    return "\n".join(lines)


def evaluate(model, processor, dev_df, root, args, label):
    """Score the held-out pages two ways:

    cer          against the full label, Postmark line included, exactly as
                 Kaggle scores it, so it is comparable with metric.py.
    cer_reading  against the label without its Postmark line, which isn't in
                 the image. This is how well the model reads what it can see.

    Returns both as means over all scored pages, plus `cer` per document."""
    was_training = model.training
    model.eval()
    rows, t0 = [], time.time()
    for r in dev_df.to_dict("records"):
        pred = read_page(model, processor, root, r["page_id"], args)
        rows.append({"page_id": r["page_id"], "doc_id": r["doc_id"],
                     "cer": page_cer(pred, r["text"]), "cer_reading": page_cer(pred, strip_postmark(r["text"])),
                     "text_ref": r["text"], "text_pred": pred})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.eval_dir, f"eval_{label}.csv"), index=False)
    result = {"dev_cer": round(float(df["cer"].mean()), 4),
              "reading_cer": round(float(df["cer_reading"].mean()), 4),
              "by_doc": {d: round(float(c), 4) for d, c in df.groupby("doc_id")["cer"].mean().items()}}
    print(f"[dev {label}] {len(df)} pages in {time.time() - t0:.0f}s -> mean page CER {result['dev_cer']:.4f} "
          f"(reading {result['reading_cer']:.4f}; by document {result['by_doc']})", flush=True)
    if was_training:
        model.train()
    return result


def predict_test(model, processor, root, args):
    """Read every Kaggle test page that has crops; one row per page."""
    path = os.path.join(args.train_dir, "test_pages.csv")
    if not os.path.exists(path):
        print("no test_pages.csv in the input folder; skipping the test predictions", flush=True)
        return
    pages = pd.read_csv(path)["page_id"].tolist()
    model.eval()
    t0 = time.time()
    rows = [{"page_id": p, "text": read_page(model, processor, root, p, args)} for p in pages]
    out = os.path.join(args.model_dir, "kade_test_predictions.csv")
    pd.DataFrame(rows, columns=["page_id", "text"]).to_csv(out, index=False)
    print(f"[test] {len(rows)} pages read in {time.time() - t0:.0f}s -> {out}", flush=True)


# ---------------------------------------------------------------------------
def load_model(args, dtype, device):
    """Qwen3-VL-4B with the Kurrent adapter merged in. This is the model the
    new adapter learns on top of, and the model inference needs too."""
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"loading {args.base_model} in {dtype}", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base_model, dtype=dtype, attn_implementation="sdpa").to(device)
    if args.start_adapter.lower() != "none":
        print(f"merging {args.start_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, args.start_adapter).merge_and_unload()
    # The processor (image resizing + tokenizer) from the adapter repo when
    # there is one, so images are cut into tokens the way it was trained.
    proc_src = args.start_adapter if args.start_adapter.lower() != "none" else args.base_model
    processor = AutoProcessor.from_pretrained(proc_src, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    return model, processor


def load_for_inference(adapter_dir, base_model=BASE_MODEL, start_adapter=START_ADAPTER):
    """For later: rebuild the fine-tuned model from model.tar.gz's contents."""
    from peft import PeftModel
    args = argparse.Namespace(base_model=base_model, start_adapter=start_adapter,
                              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    model, processor = load_model(args, pick_dtype(), "cuda" if torch.cuda.is_available() else "cpu")
    return PeftModel.from_pretrained(model, adapter_dir).eval(), processor


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    # --- data ---
    root = args.train_dir
    if os.path.exists(os.path.join(args.train_dir, "lines.zip")):
        root = "/tmp/kade_data"   # the channel can be read-only, so unpack elsewhere
        with zipfile.ZipFile(os.path.join(args.train_dir, "lines.zip")) as z:
            z.extractall(root)
    train_df = pd.read_csv(os.path.join(args.train_dir, "train_lines.csv"), keep_default_na=False)
    dev_df = pd.read_csv(os.path.join(args.train_dir, "dev_pages.csv"), keep_default_na=False)
    holdout = [] if args.holdout_docs.lower() == "none" else [d.strip() for d in args.holdout_docs.split(",")]
    train_df = train_df[~train_df.doc_id.isin(holdout)]
    # Score every document we don't train on. With --holdout_docs none that
    # leaves only the team's dev document.
    dev_df = dev_df[~dev_df.doc_id.isin(train_df.doc_id.unique())]
    if args.max_train_lines > 0:
        train_df = train_df.sample(n=min(args.max_train_lines, len(train_df)), random_state=args.seed)
    if args.dev_pages > 0:
        dev_df = dev_df.sort_values("page_id").head(args.dev_pages)
    print(f"{len(train_df)} training lines from {train_df.page_id.nunique()} pages "
          f"({', '.join(sorted(train_df.doc_id.unique()))}); {len(dev_df)} pages scored "
          f"({', '.join(sorted(dev_df.doc_id.unique()))})", flush=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # --- model ---
    dtype = pick_dtype()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args, dtype, device)

    # The new adapter: on the language model's attention and MLP layers, like
    # the Kurrent adapter. The vision encoder (names containing "visual") stays
    # frozen. LoRA starts at zero, so before training this is exactly the
    # merged Kurrent model.
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)   # the trainable LoRA weights are kept in fp32
    model.enable_input_require_grads()    # needed for gradient checkpointing with a frozen model
    model.print_trainable_parameters()

    # --- checkpoints, and whether this is a restarted job ---
    ckpt_dir = args.checkpoint_dir if os.path.isdir(args.checkpoint_dir) \
        else os.path.join(args.output_data_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = get_last_checkpoint(ckpt_dir)
    state_path = os.path.join(ckpt_dir, "dev_curve.json")
    curve = json.load(open(state_path)) if (last_ckpt and os.path.exists(state_path)) else []
    print(f"checkpoints -> {ckpt_dir}; resuming from {last_ckpt or 'nothing (fresh run)'}", flush=True)
    # Per-page scores also live in the checkpoint folder, so a restarted job
    # still has the ones from before the restart.
    args.eval_dir = os.path.join(ckpt_dir, "evals")
    os.makedirs(args.eval_dir, exist_ok=True)

    best_dir = os.path.join(ckpt_dir, "best_adapter")   # in the checkpoint folder, so a restart keeps it

    def record(epoch, result):
        # Keep the adapter from the best epoch (epoch 0 = no change yet), judged
        # by the official CER over all scored pages. On a tie the earlier epoch
        # stays, the same one the summary reports.
        if not curve or result["dev_cer"] < min(c["dev_cer"] for c in curve):
            model.save_pretrained(best_dir)
            print(f"  best so far: epoch {epoch}, adapter saved", flush=True)
        curve.append({"epoch": epoch, **result})
        json.dump(curve, open(state_path, "w"))

    if not curve:
        record(0, evaluate(model, processor, dev_df, root, args, "epoch0"))

    class DevCallback(TrainerCallback):
        def on_epoch_end(self, targs_, state, control, **kw):
            ep = int(round(state.epoch))
            record(ep, evaluate(model, processor, dev_df, root, args, f"epoch{ep}"))

    train_secs, final_loss = 0.0, None
    if args.epochs > 0:
        targs = TrainingArguments(
            output_dir=ckpt_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=args.warmup_ratio,
            weight_decay=0.0,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=1,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=2,
            seed=args.seed,
            use_cpu=device == "cpu",
        )
        trainer = Trainer(model=model, args=targs, train_dataset=LineDataset(train_df, root),
                          data_collator=Collator(processor), callbacks=[DevCallback()])
        t0 = time.time()
        result = trainer.train(resume_from_checkpoint=last_ckpt)
        train_secs, final_loss = time.time() - t0, round(float(result.training_loss), 4)
        print(f"training finished in {train_secs / 60:.1f} min; final loss {final_loss}", flush=True)
        model.save_pretrained(os.path.join(args.model_dir, "final_adapter"))
        loss = [{"step": h["step"], "epoch": round(h["epoch"], 3), "loss": h["loss"]}
                for h in trainer.state.log_history if "loss" in h]
    else:
        loss = []

    best = min(curve, key=lambda c: c["dev_cer"])
    if args.keep == "best" or args.epochs == 0:
        kept = best
        shutil.copytree(best_dir, args.model_dir, dirs_exist_ok=True)
        if args.predict_test and args.epochs > 0:
            # The model in memory holds the last epoch; switch it to the kept one.
            from peft import load_peft_weights, set_peft_model_state_dict
            set_peft_model_state_dict(model, load_peft_weights(best_dir))
    else:
        kept = curve[-1]
        model.save_pretrained(args.model_dir)
    print(f"kept the adapter from epoch {kept['epoch']} (--keep {args.keep})", flush=True)
    shutil.copytree(args.eval_dir, args.model_dir, dirs_exist_ok=True)
    processor.save_pretrained(args.model_dir)
    if args.predict_test:
        predict_test(model, processor, root, args)
    with open(os.path.join(args.model_dir, "train_log.json"), "w") as f:
        json.dump({"loss": loss, "dev_cer_by_epoch": curve}, f, indent=2)
    summary = {
        "base_model": args.base_model,
        "start_adapter": args.start_adapter,
        "train_docs": sorted(train_df.doc_id.unique().tolist()),
        "train_lines": int(len(train_df)),
        "train_pages": int(train_df.page_id.nunique()),
        "scored_docs": sorted(dev_df.doc_id.unique().tolist()),
        "dev_pages": int(len(dev_df)),
        "epochs": args.epochs,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "effective_batch": args.batch_size * args.grad_accum,
        "min_pixels": args.min_pixels,
        "seed": args.seed,
        "dtype": str(dtype).replace("torch.", ""),
        "train_minutes": round(train_secs / 60, 1),
        "final_loss": final_loss,
        "dev_cer_before": curve[0]["dev_cer"],
        "dev_cer_best": best["dev_cer"],
        "best_epoch": best["epoch"],
        "reading_cer_before": curve[0]["reading_cer"],
        "reading_cer_best": best["reading_cer"],
        "by_doc_before": curve[0]["by_doc"],
        "by_doc_best": best["by_doc"],
        "kept_epoch": kept["epoch"],
        "test_predictions": bool(args.predict_test),
    }
    with open(os.path.join(args.model_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
