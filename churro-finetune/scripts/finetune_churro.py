"""LoRA fine-tune of CHURRO-3B on whole Badger Scribe pages, all three collections,
tailored to the problems in Lokesh's dev-set audit.

Run by 2_finetune.ipynb on the notebook instance's GPU. The training loop is
adapted from finetune_churro_lora.py in the "ML with AWS SageMaker" lesson
(carpentries-incubator/ML_with_AWS_SageMaker, MIT licence). What it adds, by
audit problem (docs/dev-set-error-audit.md on lokesh/data-dev-set):

 1. Survey spreads read in the wrong order: spreads are trained and read as two
    half-pages (prepare_data.py); a reading that stopped early or got stuck is
    redone in two parts (--retry).
 2. Kade dev postcards are small, faint pencil: small images are enlarged to
    the full pixel budget (--upscale), and training images are sometimes
    shrunk and faded so the model sees such pages (--augment).
 3. Kade postmark line not in the image: never trained on; on small cards
    the part every postmark line shares is added (--postmark_guess).
 4. Dialect spelling: no spelling correction anywhere, by design.
 5. Damaged Dominy pages: `#` is never a training target; training images
    get faint mirrored bleed-through (--augment); contrast stretch (--enhance).
 6. Ledger symbols: check marks and × are removed from the output (common.py).
 7. Few or short pages count as much: Dominy's 27 training pages are repeated
    each epoch (--oversample), each time with different random changes.
 8. Survey vocabulary: learned from the 365 survey training images.
 9. Machine-drafted labels: scores on expert-typed pages are reported apart.

--epochs 0 only scores the dev pages with the model as it is (a baseline).

Everything it writes goes to --out_dir:

    adapter_config.json, adapter_model.safetensors   the LoRA adapter (~30M numbers, ~120 MB)
    dev_scores.csv        per dev page: CER, length ratio, problem, the model's text and the label
    summary.json          settings (training and reading), timings, dev CER per collection
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
    add_postmark_guess,
    augment,
    build_prompt,
    clean_output,
    cut_at_repeat,
    full_messages,
    load_image,
    load_metadata,
    looks_wrong,
    split_survey_label,
    top_bottom,
    training_text,
    user_messages,
)

# How pages are read. Saved in summary.json, so reading the test pages later
# (transcribe.py) works exactly the way the dev pages were scored.
READING_SETTINGS = ("max_pixels", "max_new_tokens", "upscale", "enhance", "retry", "postmark_guess",
                    "repetition_penalty", "retry_pick", "cut_repeats", "repeat_chars")


def without_postmark(text):
    """The text without its 'Postmark ...' lines. The postmark is on the back of a Kade card,
    so no model can read it; scoring without it shows how well the front was read."""
    return "\n".join(l for l in str(text).split("\n") if not l.strip().startswith("Postmark"))


def page_cer_function():
    """The competition's metric.py (top of the repository), so dev scores match Kaggle's."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, repo)
    import metric
    return metric.page_cer, metric.normalize_text


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
    p.add_argument("--augment", default="all", help="collections whose training images get random changes: "
                                                     "all, none, or a comma-separated list")
    p.add_argument("--oversample", default="dominy_accounts:4",
                   help="collections whose training images are used several times per epoch, e.g. dominy_accounts:4")
    p.add_argument("--max_train_items", type=int, default=0, help="0 = all; small for a smoke test")
    p.add_argument("--use_metadata", type=int, default=1)
    # --- reading (training and scoring alike) ---
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--upscale", type=int, default=1, help="1 = enlarge small images to the full pixel budget")
    p.add_argument("--enhance", default="", help="collections whose images get a contrast stretch, e.g. dominy_accounts")
    p.add_argument("--retry", type=int, default=1, help="1 = redo a reading that stopped early or got stuck, in two parts")
    p.add_argument("--postmark_guess", type=int, default=1, help="1 = add 'Postmark , 19' on small Kade cards")
    p.add_argument("--max_new_tokens", type=int, default=1536)
    p.add_argument("--repetition_penalty", type=float, default=1.05,
                   help="above 1.0 discourages repeating text; 1.0 = off")
    p.add_argument("--retry_pick", default="longer", choices=("longer", "closer"),
                   help="when both readings of a part look wrong, keep the longer one (as before), or the one "
                        "whose length is closer to the usual length")
    p.add_argument("--cut_repeats", default="",
                   help="collections whose readings are cut where they start repeating themselves, e.g. kade_letters")
    p.add_argument("--repeat_chars", type=int, default=22, help="how long a repeated stretch must be to cut there")
    p.add_argument("--adapter_dir", default="",
                   help="with --epochs 0: score this saved adapter (e.g. a run's epoch3 folder) instead of plain CHURRO")
    p.add_argument("--train_on_dev", type=int, default=0,
                   help="1 = also train on the dev pages (for the final model; its dev score then means nothing)")
    # --- which pages ---
    p.add_argument("--categories", default="all",
                   help="collections to train and score on: all, or a comma-separated list, e.g. kade_letters")
    p.add_argument("--holdout_docs", default="",
                   help="documents (doc_id) left out of training and scored instead of the usual dev pages of "
                        "their collection, e.g. kade_007,kade_112")
    p.add_argument("--train_score_pages", type=int, default=0,
                   help="training pages per collection scored at the end, to see the overfitting gap (0 = none)")
    p.add_argument("--save_every_epoch", type=int, default=0,
                   help="1 = also keep the adapter of every epoch, in epoch1/, epoch2/, ...")
    # --- scoring ---
    p.add_argument("--dev_pages", type=int, default=0, help="dev pages scored at the end; 0 = all")
    p.add_argument("--curve_pages", type=int, default=3, help="dev pages per collection scored after every epoch (0 = none)")
    p.add_argument("--eval_batch", type=int, default=4, help="images read at once")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def listed(setting, category):
    """Is this collection named in a setting like 'all', 'none' or 'dominy_accounts,kade_letters'?"""
    names = {s.strip() for s in str(setting).split(",") if s.strip()}
    return "all" in names or category in names


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
def generate(model, processor, jobs, opts, batch):
    """Read a list of images: jobs are (key, image path, box, prompt, category). Returns {key: text}."""
    out_texts = {}
    jobs = sorted(jobs, key=lambda j: (j[2] is not None, j[3]))   # similar images together: less padding
    processor.tokenizer.padding_side = "left"
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        texts = [processor.apply_chat_template(user_messages(j[3]), tokenize=False, add_generation_prompt=True)
                 for j in chunk]
        images = [load_image(j[1], j[2], opts["max_pixels"], bool(opts["upscale"]), listed(opts["enhance"], j[4]))
                  for j in chunk]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**enc, max_new_tokens=opts["max_new_tokens"], do_sample=False,
                             repetition_penalty=opts.get("repetition_penalty", 1.05), use_cache=True)
        for j, text in zip(chunk, processor.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out_texts[j[0]] = text.strip()
    processor.tokenizer.padding_side = "right"
    return out_texts


def read_pages(model, processor, page_ids, parts, data_dir, doc_ctx, categories, opts, expected, batch):
    """{page_id: text} and the set of pages that needed a second reading.

    Each page is read whole or, for a survey spread, as two halves (left, then
    right). A part whose reading looks wrong (nothing, far too short, or stuck
    repeating) is read again as its top and bottom halves; the second reading
    is kept if it looks right. Then the text is cleaned up, and small Kade cards
    get the postmark guess."""
    jobs, info = [], {}
    for pid in page_ids:
        cat = categories[pid]
        prompt = build_prompt(cat, doc_ctx.get(pid.rsplit("_", 1)[0], ""))
        for order, r in enumerate(parts[parts.page_id == pid].to_dict("records")):
            box = parse_box(r["box"])
            key = (pid, order)
            jobs.append((key, os.path.join(data_dir, r["image"]), box, prompt, cat))
            info[key] = {"expected": expected.get(f"{cat}/{'half' if box else 'page'}", 0),
                         "size": (int(r.get("width") or 0), int(r.get("height") or 0))}
    texts = generate(model, processor, jobs, opts, batch)

    retried = set()
    if opts["retry"]:
        again = []
        for key, path, box, prompt, cat in jobs:
            if looks_wrong(texts[key], info[key]["expected"]):
                w, h = info[key]["size"]
                for half, half_box in zip(("top", "bottom"), top_bottom(box, w, h)):
                    again.append(((key, half), path, half_box, prompt, cat))
        if again:
            second = generate(model, processor, again, opts, batch)
            for key in {k for (k, _), *_ in again}:
                joined = (second[(key, "top")] + "\n" + second[(key, "bottom")]).strip()
                exp = info[key]["expected"]
                if opts.get("retry_pick", "longer") == "closer" and exp > 0:
                    # A reading stuck repeating is long, so "longer" would keep the worse one.
                    off = lambda t: abs(math.log((len(" ".join(t.split())) + 1) / (exp + 1)))
                    better = off(joined) < off(texts[key])
                else:
                    better = len(joined) > len(texts[key])
                if not looks_wrong(joined, exp) or better:
                    texts[key] = joined
                    retried.add(key[0])

    result = {}
    for pid in page_ids:
        keys = sorted(k for k in texts if k[0] == pid)
        text = clean_output("\n".join(texts[k] for k in keys), categories[pid])
        if listed(opts.get("cut_repeats", ""), categories[pid]):
            text = cut_at_repeat(text, int(opts.get("repeat_chars", 22)))
        if opts["postmark_guess"] and keys:
            w, h = info[keys[0]]["size"]
            text = add_postmark_guess(text, categories[pid], w, h)
        result[pid] = text
    return result, retried


def score_dev(model, processor, dev, parts, args, opts, doc_ctx, expected, metric_fns, label):
    """Read the dev pages, score each with metric.py, and report per collection,
    along with the audit's length check: a reading under 0.7 of the label's length
    probably stopped early or skipped part of the page; over 1.1, it probably
    repeated something or read a part twice."""
    page_cer, normalize = metric_fns
    was_training = model.training
    model.eval()
    start = time.time()
    preds, retried = read_pages(model, processor, list(dev.page_id), parts, args.data_dir, doc_ctx,
                                dict(zip(dev.page_id, dev.category)), opts, expected, args.eval_batch)
    rows = []
    for r in dev.to_dict("records"):
        pred, ref = preds[r["page_id"]], r["text"]
        ratio = len(normalize(pred)) / max(1, len(normalize(ref)))
        rows.append({"page_id": r["page_id"], "doc_id": r.get("doc_id", ""), "category": r["category"],
                     "label_source": r.get("label_source", ""),
                     "cer": page_cer(pred, ref), "cer_no_postmark": page_cer(without_postmark(pred), without_postmark(ref)),
                     "length_ratio": round(ratio, 2),
                     "problem": "too short" if ratio < 0.7 else "too long" if ratio > 1.1 else "",
                     "read_twice": r["page_id"] in retried, "prediction": pred, "reference": ref})
    df = pd.DataFrame(rows)
    by_cat = df.groupby("category")["cer"].mean()
    print(f"[dev {label}] {len(df)} pages in {(time.time() - start) / 60:.1f} min: "
          + ", ".join(f"{c} {v:.3f}" for c, v in by_cat.items()) + f"; mean of collections {by_cat.mean():.3f}"
          + (f"; {len(retried)} read twice" if retried else ""), flush=True)
    if (df.category == "kade_letters").any():
        kade = df[df.category == "kade_letters"]
        print(f"    kade_letters without postmarks {kade.cer_no_postmark.mean():.3f}; by document: "
              + ", ".join(f"{d} {v:.3f}" for d, v in kade.groupby("doc_id")["cer"].mean().items()), flush=True)
    if was_training:
        model.train()
    return df, by_cat


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
def oversampled(items, setting):
    """Repeat the training images of collections like 'dominy_accounts:4', so a
    collection with few pages still counts in training (the score is the mean
    of the three collections, so its 36 test pages weigh as much as Kade's 141)."""
    extra = []
    for part in [s for s in str(setting).split(",") if ":" in s]:
        cat, times = part.split(":")
        extra += [items[items.category == cat.strip()]] * (int(times) - 1)
    return pd.concat([items, *extra]) if extra else items


class PageDataset(Dataset):
    def __init__(self, items, data_dir, doc_ctx, opts, augment_setting, seed):
        self.rows = items.to_dict("records")
        self.data_dir, self.doc_ctx, self.opts = data_dir, doc_ctx, opts
        self.augment_setting = augment_setting
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        cat = r["category"]
        img = load_image(os.path.join(self.data_dir, r["image"]), parse_box(r["box"]), self.opts["max_pixels"],
                         bool(self.opts["upscale"]), listed(self.opts["enhance"], cat))
        if listed(self.augment_setting, cat):
            img = augment(img, self.rng, cat)
        prompt = build_prompt(cat, self.doc_ctx.get(r["doc_id"], ""))
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
    metric_fns = page_cer_function()
    opts = {k: getattr(args, k) for k in READING_SETTINGS}

    items = pd.read_csv(os.path.join(args.data_dir, "train_items.csv"), keep_default_na=False)
    dev = pd.read_csv(os.path.join(args.data_dir, "dev_pages.csv"), keep_default_na=False)
    parts = pd.read_csv(os.path.join(args.data_dir, "parts.csv"), keep_default_na=False)
    expected_path = os.path.join(args.data_dir, "expected_chars.json")
    expected = json.load(open(expected_path)) if os.path.exists(expected_path) else {}
    doc_ctx = load_metadata(os.path.join(args.data_dir, "metadata.csv")) if args.use_metadata else {}
    train_pages = None
    if args.holdout_docs or args.train_score_pages:
        train_pages = pd.read_csv(os.path.join(args.data_dir, "train.csv"), keep_default_na=False, encoding="utf-8-sig")
        if "category" not in train_pages.columns:
            from common import category_of
            train_pages["category"] = train_pages["page_id"].map(category_of)
        if "label_source" not in train_pages.columns:
            train_pages["label_source"] = ""
        train_pages = train_pages[train_pages.page_id.isin(set(parts.page_id))]   # pages with an image
    if args.holdout_docs:
        # Score whole documents the model never trained on. Pages within a document share pen,
        # paper and scan, so held-out pages of a trained document would flatter the score.
        held = {d.strip() for d in args.holdout_docs.split(",") if d.strip()}
        unknown = held - set(train_pages.doc_id)
        if unknown:
            raise SystemExit(f"No such document in train.csv: {', '.join(sorted(unknown))}")
        items = items[~items.doc_id.isin(held)]
        held_pages = train_pages[train_pages.doc_id.isin(held)]
        dev = pd.concat([dev[~dev.category.isin(set(held_pages.category))],
                         held_pages[["page_id", "doc_id", "category", "label_source", "text"]]], ignore_index=True)
        print(f"held out: {', '.join(f'{d} ({n} pages)' for d, n in held_pages.doc_id.value_counts().sort_index().items())}",
              flush=True)
    if args.categories != "all":
        items = items[items.category.map(lambda c: listed(args.categories, c))]
        dev = dev[dev.category.map(lambda c: listed(args.categories, c))]
        if not len(items) and args.epochs > 0:
            raise SystemExit(f"No training images in collections '{args.categories}'.")
    if args.train_on_dev:
        if args.holdout_docs:
            raise SystemExit("--train_on_dev and --holdout_docs don't go together: the held-out pages are the dev pages.")
        extra = []
        for r in dev.to_dict("records"):
            prs = parts[parts.page_id == r["page_id"]].to_dict("records")
            if not prs:
                continue
            text = training_text(r["text"], r["category"])
            halves = split_survey_label(text) if len(prs) == 2 else None   # a survey spread, as in prepare_data.py
            base = {"page_id": r["page_id"], "doc_id": r["doc_id"], "category": r["category"]}
            if halves:
                extra += [{**base, "image": p["image"], "box": p["box"], "text": h} for p, h in zip(prs, halves)]
            else:
                extra.append({**base, "image": prs[0]["image"], "box": "", "text": text})
        items = pd.concat([items, pd.DataFrame(extra)], ignore_index=True)
        print(f"also training on the {len(dev)} dev pages ({len(extra)} images)", flush=True)
    trained_on = sorted(set(items.page_id))
    if args.max_train_items:
        items = per_category(items, lambda g: g.sample(min(len(g), max(1, args.max_train_items // 3)),
                                                       random_state=args.seed))
    items = oversampled(items, args.oversample)
    if args.dev_pages:
        dev = per_category(dev, lambda g: g.sort_values("page_id").head(max(1, args.dev_pages // 3)))
    curve_dev = per_category(dev, lambda g: g.sort_values("page_id").head(args.curve_pages)) \
        if args.curve_pages else dev.iloc[0:0]
    print(f"training images per epoch: {len(items)} ({', '.join(f'{c} {n}' for c, n in items.category.value_counts().items())}); "
          f"dev pages: {len(dev)}", flush=True)

    if args.adapter_dir and args.epochs > 0:
        raise SystemExit("--adapter_dir only scores a saved adapter; use it with --epochs 0.")
    model, processor, dtype = load_model(args.model_id, args.max_pixels, args.adapter_dir or None)
    summary = {"model_id": args.model_id, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "train_vision": args.train_vision, "augment": args.augment,
               "oversample": args.oversample, "effective_batch": args.batch_size * args.grad_accum,
               "train_images_per_epoch": len(items), "dev_pages": len(dev),
               "categories": args.categories, "holdout_docs": args.holdout_docs,
               "trained_on_docs": sorted(set(items.doc_id)), "adapter_dir": args.adapter_dir,
               "train_on_dev": args.train_on_dev,
               "dtype": str(dtype).replace("torch.", ""), "reading": opts}
    curve, loss_log = [], []

    if args.epochs > 0:
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer, TrainerCallback, TrainingArguments
        from transformers.trainer_utils import get_last_checkpoint

        # LoRA on the language model's attention and MLP layers; with --train_vision 1,
        # also on the image encoder's ("visual") blocks.
        layers = "q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj"
        target = rf".*\.(?:{layers}|qkv|attn\.proj)$" if args.train_vision else rf"^(?!.*visual).*\.(?:{layers})$"
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                                 lora_dropout=args.lora_dropout, bias="none",
                                                 task_type="CAUSAL_LM", target_modules=target))
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        def curve_point(epoch, label):
            df, by_cat = score_dev(model, processor, curve_dev, parts, args, opts, doc_ctx, expected, metric_fns, label)
            point = {"epoch": epoch, **by_cat.round(4).to_dict(), "mean": round(by_cat.mean(), 4)}
            if (df.category == "kade_letters").any():
                point["kade_letters without postmarks"] = round(df[df.category == "kade_letters"].cer_no_postmark.mean(), 4)
            df.to_csv(os.path.join(args.out_dir, f"curve_scores_epoch{epoch}.csv"), index=False)
            curve.append(point)

        if len(curve_dev):
            curve_point(0, "before training")

        class Watch(TrainerCallback):
            def on_log(self, targs, state, control, logs=None, **kw):
                if logs and "loss" in logs:
                    loss_log.append({"step": state.global_step, "loss": logs["loss"]})
                    if not math.isfinite(logs["loss"]):
                        raise SystemExit("The training loss became NaN. This happens with fp16 on a T4 GPU "
                                         "(ml.g4dn). Use an ml.g5.xlarge or ml.g6.xlarge instance, which trains in bf16.")

            def on_epoch_end(self, targs, state, control, **kw):
                if args.save_every_epoch:
                    model.save_pretrained(os.path.join(args.out_dir, f"epoch{round(state.epoch)}"))
                if len(curve_dev):
                    curve_point(round(state.epoch), f"after epoch {round(state.epoch)}")

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
        trainer = Trainer(model=model, args=targs,
                          train_dataset=PageDataset(items, args.data_dir, doc_ctx, opts, args.augment, args.seed),
                          data_collator=Collator(processor), callbacks=[Watch()])
        start = time.time()
        result = trainer.train(resume_from_checkpoint=last)
        summary["train_minutes"] = round((time.time() - start) / 60, 1)
        summary["final_loss"] = round(float(result.training_loss), 4)
        print(f"training finished in {summary['train_minutes']} min; final loss {summary['final_loss']}", flush=True)
        model.save_pretrained(args.out_dir)

    processor.save_pretrained(args.out_dir)
    scores, by_cat = score_dev(model, processor, dev, parts, args, opts, doc_ctx, expected, metric_fns,
                               "after training" if args.epochs > 0 else "baseline (no training)")
    scores.to_csv(os.path.join(args.out_dir, "dev_scores.csv"), index=False)
    summary["dev_cer"] = {**by_cat.round(4).to_dict(), "mean of collections": round(by_cat.mean(), 4)}
    summary["dev_cer_no_postmark"] = scores.groupby("category")["cer_no_postmark"].mean().round(4).to_dict()
    summary["dev_cer_by_doc"] = scores.groupby("doc_id")["cer"].mean().round(4).to_dict()
    if args.train_score_pages:
        # The same reading on pages the model trained on. A big gap between these and the
        # held-out pages means it learned those pages rather than how to read.
        seen = train_pages[train_pages.page_id.isin(trained_on)]
        seen = per_category(seen, lambda g: g.sort_values("page_id").head(args.train_score_pages))
        train_scores, train_by_cat = score_dev(model, processor, seen, parts, args, opts, doc_ctx, expected,
                                               metric_fns, "training pages (overfitting check)")
        train_scores.to_csv(os.path.join(args.out_dir, "train_page_scores.csv"), index=False)
        summary["train_pages_cer"] = train_by_cat.round(4).to_dict()
    typed = scores[scores.label_source == "human"]
    if len(typed):
        summary["dev_cer_expert_typed_pages"] = typed.groupby("category")["cer"].mean().round(4).to_dict()
    summary["dev_problems"] = {c: {"too short": int((g.problem == "too short").sum()),
                                   "too long": int((g.problem == "too long").sum()),
                                   "read twice": int(g.read_twice.sum()), "pages": len(g)}
                               for c, g in scores.groupby("category")}
    if len(curve) > 1:
        best = min(curve, key=lambda p: p["mean"])
        summary["best_epoch"] = best["epoch"]
        summary["best_epoch_dev_cer"] = best["mean"]
    with open(os.path.join(args.out_dir, "curve.json"), "w") as f:
        json.dump({"dev_cer_by_epoch": curve, "loss": loss_log}, f, indent=2)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
