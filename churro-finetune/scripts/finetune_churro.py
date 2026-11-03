"""LoRA fine-tune of CHURRO-3B on whole Badger Scribe pages, all three collections.

Run by 2_finetune.ipynb on the notebook instance's GPU. Adapted from
finetune_churro_lora.py in the "ML with AWS SageMaker" lesson
(carpentries-incubator/ML_with_AWS_SageMaker, MIT licence). Changes:

- held-out pages are the team's dev set (whole documents), not a random 15% of
  pages, so scores are honest about documents the model hasn't seen;
- survey spreads are trained and read as two half-pages (see common.py);
- dev pages are scored with the competition's own metric.py, per collection;
- several pages are read at once, `#` is never a training target, and
  training images can be lightly varied (--augment).

--epochs 0 only scores the dev pages with the model as it is (a baseline).

Everything it writes goes to --out_dir:

    adapter_config.json, adapter_model.safetensors   the LoRA adapter (~30M numbers, ~120 MB)
    dev_scores.csv        per dev page: CER, the model's text and the label
    summary.json          settings, timings, dev CER per collection and overall
    curve.json            dev CER on a few pages after every epoch, and the training loss
"""

import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_ID,
    augment,
    build_prompt,
    clean_output,
    full_messages,
    load_image,
    load_metadata,
    user_messages,
)


def page_cer_function():
    """The competition's metric.py (top of the repository), so dev scores match Kaggle's."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    import metric
    return metric.page_cer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="folder with train_items.csv, dev_pages.csv, parts.csv, images/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model_id", default=MODEL_ID)
    # --- training ---
    p.add_argument("--epochs", type=float, default=2.0, help="0 = only score the dev pages (a baseline)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--train_vision", type=int, default=0, help="1 = also train the image side of the model")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--augment", default="", help="collections whose training images are varied, e.g. dominy_accounts, or all")
    p.add_argument("--max_train_items", type=int, default=0, help="0 = all; small for a smoke test")
    p.add_argument("--use_metadata", type=int, default=1)
    # --- scoring ---
    p.add_argument("--dev_pages", type=int, default=0, help="dev pages scored at the end; 0 = all")
    p.add_argument("--curve_pages", type=int, default=3, help="dev pages per collection scored after every epoch (0 = none)")
    p.add_argument("--eval_batch", type=int, default=4, help="images read at once")
    p.add_argument("--max_new_tokens", type=int, default=1536)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pick_dtype():
    """bf16 on GPUs that have it (A10G on g5, L4 on g6). The T4 on g4dn doesn't, so fp16;
    CPU (only for testing the script) uses fp32."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def check_cudnn(dtype):
    """Turn cuDNN off if it can't load in this kernel (seen on conda_python3 with
    pip-installed PyTorch). The model uses it only for the patch-cutting layer,
    which PyTorch can run without it, a little slower."""
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        layer = torch.nn.Conv3d(3, 8, kernel_size=(2, 14, 14), stride=(2, 14, 14)).to("cuda", dtype)
        layer(torch.randn(4, 3, 2, 14, 14, device="cuda", dtype=dtype))
        torch.cuda.synchronize()
    except RuntimeError as e:
        torch.backends.cudnn.enabled = False
        print(f"cuDNN doesn't work in this kernel ({str(e).splitlines()[0][:90]}). Carrying on without it, "
              "a little slower. The conda_pytorch_p310 kernel avoids this.", flush=True)


def load_model(model_id, max_pixels, adapter_dir=None):
    """CHURRO-3B (and an adapter on top, if given) on the GPU, with its processor."""
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = pick_dtype()
    check_cudnn(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_id} in {dtype}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa").to(device)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=MIN_PIXELS, max_pixels=max_pixels)
    if adapter_dir:
        print(f"adding the adapter in {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), processor, dtype


# ---------------------------------------------------------------------------
# Reading pages
# ---------------------------------------------------------------------------
def parse_box(box):
    return tuple(int(v) for v in box.split(",")) if isinstance(box, str) and box else None


@torch.no_grad()
def read_pages(model, processor, page_ids, parts, data_dir, doc_ctx, categories, max_pixels,
               max_new_tokens, batch, log_every=1):
    """{page_id: text}. Each page is read whole or, for a survey spread, as two halves
    (left, then right) joined with a newline; the text is then cleaned up."""
    jobs = []   # (page_id, order, image path, box, prompt)
    for pid in page_ids:
        rows = parts[parts.page_id == pid]
        prompt = build_prompt(categories[pid], doc_ctx.get(pid.rsplit("_", 1)[0], ""))
        for order, r in enumerate(rows.to_dict("records")):
            jobs.append((pid, order, os.path.join(data_dir, r["image"]), parse_box(r["box"]), prompt))
    # Similar sizes together, so a batch doesn't pad small images up to a big one.
    jobs.sort(key=lambda j: (j[3] is not None, j[4]))
    pieces = {pid: {} for pid in page_ids}
    processor.tokenizer.padding_side = "left"
    start, done = time.time(), 0
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        texts = [processor.apply_chat_template(user_messages(j[4]), tokenize=False, add_generation_prompt=True)
                 for j in chunk]
        images = [load_image(j[2], j[3], max_pixels) for j in chunk]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.05,
                             use_cache=True)
        for j, text in zip(chunk, processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            pieces[j[0]][j[1]] = text.strip()
        done += len(chunk)
        if log_every and (s // batch) % log_every == 0:
            print(f"  read {done}/{len(jobs)} images, {(time.time() - start) / done:.1f}s each", flush=True)
    processor.tokenizer.padding_side = "right"
    return {pid: clean_output("\n".join(p[k] for k in sorted(p)), categories[pid]) for pid, p in pieces.items()}


def score_dev(model, processor, dev, parts, args, doc_ctx, page_cer, label):
    """Read the dev pages, score each with metric.py, and report per collection."""
    was_training = model.training
    model.eval()
    start = time.time()
    preds = read_pages(model, processor, list(dev.page_id), parts, args.data_dir, doc_ctx,
                       dict(zip(dev.page_id, dev.category)), args.max_pixels, args.max_new_tokens, args.eval_batch,
                       log_every=0)
    rows = [{"page_id": r["page_id"], "category": r["category"], "cer": page_cer(preds[r["page_id"]], r["text"]),
             "prediction": preds[r["page_id"]], "reference": r["text"]} for r in dev.to_dict("records")]
    df = pd.DataFrame(rows)
    by_cat = df.groupby("category")["cer"].mean()
    print(f"[dev {label}] {len(df)} pages in {(time.time() - start) / 60:.1f} min: "
          + ", ".join(f"{c} {v:.3f}" for c, v in by_cat.items()) + f"; mean of collections {by_cat.mean():.3f}",
          flush=True)
    if was_training:
        model.train()
    return df, by_cat


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
class PageDataset(Dataset):
    def __init__(self, items, data_dir, doc_ctx, max_pixels, augment_categories, seed):
        self.rows = items.to_dict("records")
        self.data_dir, self.doc_ctx, self.max_pixels = data_dir, doc_ctx, max_pixels
        self.augment_categories = augment_categories
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(os.path.join(self.data_dir, r["image"]), parse_box(r["box"]), self.max_pixels)
        if "all" in self.augment_categories or r["category"] in self.augment_categories:
            img = augment(img, self.rng)
        prompt = build_prompt(r["category"], self.doc_ctx.get(r["doc_id"], ""))
        return {"image": img, "prompt": prompt, "answer": r["text"]}


class Collator:
    """Image + prompt + answer as tensors. Only the answer is graded; and within it,
    tokens made only of '#' are not: '#' marks what the human transcriber couldn't
    read, and the model shouldn't learn to write it."""

    def __init__(self, processor):
        self.processor = processor
        vocab = processor.tokenizer.get_vocab()
        self.hash_ids = torch.tensor(sorted(
            i for tok, i in vocab.items()
            if (s := processor.tokenizer.convert_tokens_to_string([tok]).strip()) and set(s) == {"#"}))

    def __call__(self, batch):
        proc = self.processor
        proc.tokenizer.padding_side = "right"
        images = [b["image"] for b in batch]
        full = [proc.apply_chat_template(full_messages(b["prompt"], b["answer"]), tokenize=False) for b in batch]
        enc = proc(text=full, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, b in enumerate(batch):
            prompt = proc.apply_chat_template(user_messages(b["prompt"]), tokenize=False, add_generation_prompt=True)
            n_prompt = proc(text=[prompt], images=[images[i]], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        if len(self.hash_ids):
            labels[torch.isin(enc["input_ids"], self.hash_ids)] = -100
        enc["labels"] = labels
        return enc


def per_category(df, pick):
    """Apply `pick` to each collection's rows and put them back together."""
    return pd.concat([pick(df[df.category == c]) for c in sorted(df.category.unique())]) if len(df) else df


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    page_cer = page_cer_function()

    items = pd.read_csv(os.path.join(args.data_dir, "train_items.csv"), keep_default_na=False)
    dev = pd.read_csv(os.path.join(args.data_dir, "dev_pages.csv"), keep_default_na=False)
    parts = pd.read_csv(os.path.join(args.data_dir, "parts.csv"), keep_default_na=False)
    doc_ctx = load_metadata(os.path.join(args.data_dir, "metadata.csv")) if args.use_metadata else {}
    if args.max_train_items:
        items = per_category(items, lambda g: g.sample(min(len(g), max(1, args.max_train_items // 3)),
                                                       random_state=args.seed))
    if args.dev_pages:
        dev = per_category(dev, lambda g: g.sort_values("page_id").head(max(1, args.dev_pages // 3)))
    curve_dev = per_category(dev, lambda g: g.sort_values("page_id").head(args.curve_pages)) \
        if args.curve_pages else dev.iloc[0:0]
    print(f"training images: {len(items)} ({', '.join(f'{c} {n}' for c, n in items.category.value_counts().items())}); "
          f"dev pages: {len(dev)}", flush=True)

    model, processor, dtype = load_model(args.model_id, args.max_pixels)
    summary = {"model_id": args.model_id, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "train_vision": args.train_vision, "augment": args.augment,
               "max_pixels": args.max_pixels, "effective_batch": args.batch_size * args.grad_accum,
               "train_images": len(items), "dev_pages": len(dev), "dtype": str(dtype).replace("torch.", "")}
    curve, loss_log = [], []

    if args.epochs > 0:
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, TrainerCallback, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint

        # LoRA on the language model's attention and MLP layers; with --train_vision 1,
        # also on the image encoder's ("visual") blocks.
        layers = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        target = (rf".*\.(?:{layers[1:-1]}|qkv|attn\.proj)$" if args.train_vision
                  else rf"^(?!.*visual).*\.{layers}$")
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=args.lora_dropout, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=target))
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        if len(curve_dev):
            _, by_cat = score_dev(model, processor, curve_dev, parts, args, doc_ctx, page_cer, "before training")
            curve.append({"epoch": 0, **by_cat.round(4).to_dict(), "mean": round(by_cat.mean(), 4)})

        class Watch(TrainerCallback):
            def on_log(self, targs, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    loss_log.append({"step": state.global_step, "loss": logs["loss"]})
                    if not math.isfinite(logs["loss"]):
                        raise SystemExit("The training loss became NaN. This happens with fp16 on a T4 GPU "
                                         "(ml.g4dn). Use an ml.g5.xlarge or ml.g6.xlarge instance, which trains in bf16.")

            def on_epoch_end(self, targs, state, control, **kw):
                if len(curve_dev):
                    _, by_cat = score_dev(model, processor, curve_dev, parts, args, doc_ctx, page_cer,
                                          f"after epoch {round(state.epoch)}")
                    curve.append({"epoch": round(state.epoch), **by_cat.round(4).to_dict(),
                                  "mean": round(by_cat.mean(), 4)})

        ckpt_dir = os.path.join(args.out_dir, "checkpoints")
        last = get_last_checkpoint(ckpt_dir) if os.path.isdir(ckpt_dir) else None
        print(f"checkpoints -> {ckpt_dir}; resuming from {last or 'nothing (fresh run)'}", flush=True)
        targs = TrainingArguments(
            output_dir=ckpt_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
            warmup_ratio=0.03, weight_decay=0.0, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=5, save_strategy="epoch", save_total_limit=1, report_to="none",
            remove_unused_columns=False, dataloader_num_workers=2, seed=args.seed,
            use_cpu=not torch.cuda.is_available())
        augment_categories = [c.strip() for c in args.augment.split(",") if c.strip()]
        trainer = Trainer(model=model, args=targs,
                          train_dataset=PageDataset(items, args.data_dir, doc_ctx, args.max_pixels,
                                                    augment_categories, args.seed),
                          data_collator=Collator(processor), callbacks=[Watch()])
        start = time.time()
        result = trainer.train(resume_from_checkpoint=last)
        summary["train_minutes"] = round((time.time() - start) / 60, 1)
        summary["final_loss"] = round(float(result.training_loss), 4)
        print(f"training finished in {summary['train_minutes']} min; final loss {summary['final_loss']}", flush=True)
        model.save_pretrained(args.out_dir)

    processor.save_pretrained(args.out_dir)
    scores, by_cat = score_dev(model, processor, dev, parts, args, doc_ctx, page_cer,
                               "after training" if args.epochs > 0 else "baseline (no training)")
    scores.to_csv(os.path.join(args.out_dir, "dev_scores.csv"), index=False)
    summary["dev_cer"] = {**by_cat.round(4).to_dict(), "mean of collections": round(by_cat.mean(), 4)}
    with open(os.path.join(args.out_dir, "curve.json"), "w") as f:
        json.dump({"dev_cer_by_epoch": curve, "loss": loss_log}, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
