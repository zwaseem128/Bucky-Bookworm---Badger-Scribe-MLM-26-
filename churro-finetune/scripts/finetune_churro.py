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

10. Dominy backs (a third of its test pages): an address or docket written sideways or
    upside down on an otherwise blank sheet. Short readings are redone at the other three
    rotations and the reading the model is surest of is kept (--rotations); training
    images are turned at random so the model learns to read them (--rotate_train); the
    catalog record, which names the people, joins the prompt (--rich_context).
11. Dense Dominy ledgers that stop early: the amount of ink on the page says how much
    text to expect, and a reading far below it is redone in halves, twice if need be
    (--ink_check).

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
# Pages differ in size, so the GPU's memory gets cut into odd-sized pieces; growable segments let
# PyTorch reuse them (the first Qwen3-VL-8B run stopped with 1.2 GB free but in pieces).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BLOCK_MAX_ZOOM,
    DENSE_CHARS,
    MAX_PIXELS,
    MIN_PIXELS,
    MODEL_ID,
    MODELS,
    add_postmark_guess,
    augment,
    block_turns,
    build_prompt,
    catalog_fix,
    catalog_names,
    catalog_context,
    clean_output,
    cut_at_repeat,
    full_messages,
    ink_blocks,
    ink_chars,
    join_halves,
    load_image,
    load_metadata,
    looks_wrong,
    split_survey_label,
    stopped_early,
    top_bottom,
    training_text,
    user_messages,
)

# How pages are read. Saved in summary.json, so reading the test pages later
# (transcribe.py) works exactly the way the dev pages were scored.
READING_SETTINGS = ("max_pixels", "max_new_tokens", "upscale", "enhance", "retry", "postmark_guess",
                    "repetition_penalty", "retry_pick", "cut_repeats", "repeat_chars",
                    "no_repeat_ngram_size", "retry_repetition_penalty", "compare_adapter", "compare_ratio",
                    "rotations", "rotation_margin", "ink_check", "rich_context", "blocks", "catalog_fix",
                    "vote", "vote_samples", "vote_temperature")

# --rotations: a reading shorter than this, or a page with less ink than this many characters,
# is also read at 90, 180 and 270 degrees. Every labelled Dominy back is under both. A dense
# page isn't turned: a short reading of it stopped early, which the ink check handles.
ROTATE_BELOW_CHARS = 250
ROTATE_BELOW_INK = 350
# A turned reading may be this long at most. Only pages with under 700 characters' worth of ink
# are turned (about 300 tokens), and an untrained model reading a page the wrong way up tends to
# write until it is stopped: at the full limit, three turned readings of every short page made
# the first scoring pass of a run take over 20 minutes.
TURNED_MAX_TOKENS = 400
# --blocks: what a cut-out patch is told, and how long its reading may be (a patch is an address
# or a docket: a few lines at most).
BLOCK_HINT = ("This image is a small piece cut out of a larger sheet: one patch of writing, such as an "
              "address, a docket or an amount. Write only the words and figures in this piece.")
BLOCK_MAX_TOKENS = 120
_CATALOG_NAMES = {}


def catalog_names_for(data_dir, collections):
    """{doc_id: names in its catalog record} for --catalog_fix, read once per data folder."""
    key = (data_dir, collections)
    if key not in _CATALOG_NAMES:
        _CATALOG_NAMES[key] = {doc: catalog_names(rec) for doc, rec in
                               load_metadata(os.path.join(data_dir, "metadata.csv"), collections).items()}
        if not _CATALOG_NAMES[key]:
            raise SystemExit(f"catalog_fix = {collections!r} needs the competition's metadata.csv in {data_dir} "
                             "(it comes with the Kaggle download). Upload it there, then run this again.")
    return _CATALOG_NAMES[key]


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
    p.add_argument("--rotate_train", default="",
                   help="collections whose training images are turned by 90, 180 or 270 degrees half the time, "
                        "so the model learns to read sideways and upside-down writing, e.g. dominy_accounts")
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
    p.add_argument("--no_repeat_ngram_size", type=int, default=0,
                   help="never write the same run of this many tokens twice (0 = off; try 10 or more, since short "
                        "runs like 'u. die' really do repeat in a letter)")
    p.add_argument("--retry_repetition_penalty", type=float, default=0.0,
                   help="repetition penalty for the second reading of a part that looked wrong (0 = same as the first)")
    p.add_argument("--compare_adapter", default="",
                   help="with --epochs 0: also read every page with this second adapter (e.g. another epoch), and "
                        "keep its reading when the main one is much longer (a runaway)")
    p.add_argument("--compare_ratio", type=float, default=1.5,
                   help="how many times longer the main reading must be before the compare reading is kept")
    p.add_argument("--rotations", default="",
                   help="collections whose short readings are redone at 90, 180 and 270 degrees, keeping the "
                        "reading the model is surest of, e.g. dominy_accounts")
    p.add_argument("--rotation_margin", type=float, default=0.1,
                   help="how much surer (mean log-probability per token) a turned reading must be to replace "
                        "the upright one")
    p.add_argument("--ink_check", default="",
                   help="collections where 'stopped early' is judged from the ink on the page instead of the "
                        "collection's usual length, and such pages are redone in halves, e.g. dominy_accounts")
    p.add_argument("--rich_context", default="",
                   help="collections whose prompt gets the catalog record from <data_dir>/metadata.csv (title, "
                        "creator, date, summary), e.g. dominy_accounts. No other collection gets one.")
    p.add_argument("--blocks", default="",
                   help="collections whose sparse pages (backs) are read patch by patch: each patch of writing is "
                        "found from its ink, turned the way its lines run, cut out, enlarged and read on its own, "
                        "e.g. dominy_accounts")
    p.add_argument("--catalog_fix", default="",
                   help="collections whose readings get near-miss names respelled from the catalog record "
                        "(metadata.csv) after reading, e.g. dominy_accounts. Unlike --rich_context, the record "
                        "never joins the prompt.")
    p.add_argument("--vote", default="",
                   help="collections whose sparse pages (backs) are read several ways (whole and upright, turned, "
                        "patch by patch, with the contrast stretch flipped, and a few sampled readings of each) "
                        "and the reading that agrees most with the others is kept, e.g. dominy_accounts")
    p.add_argument("--vote_samples", type=int, default=4,
                   help="--vote: sampled readings of the whole page and of the patches, each (0 = greedy ones only)")
    p.add_argument("--vote_temperature", type=float, default=0.7, help="--vote: how freely the samples are drawn")
    p.add_argument("--init_adapter", default="",
                   help="start training from this saved adapter (e.g. a run's epoch3) instead of a new one, "
                        "with a fresh learning-rate schedule")
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


def load_model(model_id, max_pixels, adapter_dir=None, compare_dir=None):
    """The base model (CHURRO-3B, or another Qwen-VL such as Qwen3-VL-8B; see MODELS in common.py)
    and an adapter on top, if given, on the GPU, with its processor.
    compare_dir loads a second adapter beside the first, named "compare"."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODELS.get(model_id, model_id)
    dtype = pick_dtype()
    check_cudnn(dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {model_id} in {dtype}", flush=True)
    # Load straight onto the GPU. Loading to the CPU first and then moving it needs the whole model in
    # RAM: Qwen3-VL-8B is about 17.5 GB in bf16, more than the 16 GB an ml.g5.xlarge has.
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa",
                                                        device_map={"": device})
    processor = AutoProcessor.from_pretrained(model_id)
    # Our own load_image already shrinks every image to max_pixels; this keeps the processor from
    # resizing it again. Qwen3-VL's processor counts pixels in its "size" dict, Qwen2.5-VL's
    # (CHURRO's) in min_pixels / max_pixels, so both are set.
    ip = processor.image_processor
    ip.min_pixels, ip.max_pixels = MIN_PIXELS, max_pixels
    if isinstance(getattr(ip, "size", None), dict):
        ip.size = {"shortest_edge": MIN_PIXELS, "longest_edge": max_pixels}
    if adapter_dir:
        print(f"adding the adapter in {adapter_dir}", flush=True)
        model = PeftModel.from_pretrained(model, adapter_dir)
    if compare_dir:
        if not adapter_dir:
            raise SystemExit("compare_adapter needs a main adapter to compare with (adapter_dir).")
        print(f"adding a second adapter to compare with, from {compare_dir}", flush=True)
        model.load_adapter(compare_dir, adapter_name="compare")
        model.set_adapter("default")
    return model.eval(), processor, dtype


# ---------------------------------------------------------------------------
# Reading pages
# ---------------------------------------------------------------------------
def parse_box(box):
    return tuple(int(v) for v in box.split(",")) if isinstance(box, str) and box else None


def flat_len(text):
    return len(" ".join(str(text).split()))


def patch_reading_ok(text, chars):
    """A patch's reading is kept only if it has something in it, isn't stuck repeating, and isn't
    far longer than the patch's ink: a long reading of a small patch is a guess, however sure."""
    return bool(str(text).strip()) and not looks_wrong(text, 0) and flat_len(text) <= 4 * chars + 30


@torch.no_grad()
def generate(model, processor, jobs, opts, batch, penalty=None, sureness=None, max_tokens=None, temperature=0.0):
    """Read a list of images: jobs are (key, image path, box, prompt, category) and, optionally,
    a rotation in degrees and a pixel cap for that image. Returns {key: text}.

    sureness, if given, is a dict that gets {key: the mean log-probability of the tokens written}:
    how sure the model was of this reading. 0 is certain; a guess is well below.
    max_tokens, if given, replaces the run's max_new_tokens for these readings.
    temperature above 0 samples each token instead of taking the likeliest (--vote), so the same
    job twice gives two different readings."""
    sampling = {"do_sample": True, "temperature": temperature, "top_p": 0.95} if temperature > 0 \
        else {"do_sample": False}
    out_texts = {}
    jobs = sorted(jobs, key=lambda j: (j[2] is not None, j[3]))   # similar images together: less padding
    processor.tokenizer.padding_side = "left"
    extra, stops = {}, set()
    if sureness is not None:
        from transformers import LogitsProcessor, LogitsProcessorList

        class Sureness(LogitsProcessor):
            """Notes, at every step, the log-probability of the token greedy reading is about to pick."""

            def __init__(self):
                self.steps = []

            def __call__(self, input_ids, scores):
                self.steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.cpu())
                return scores

        for cfg in (getattr(model, "generation_config", None), processor.tokenizer):
            for name in ("eos_token_id", "pad_token_id"):
                ids = getattr(cfg, name, None)
                stops.update(ids if isinstance(ids, (list, tuple)) else [ids])
        stops.discard(None)
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        texts = [processor.apply_chat_template(user_messages(j[3]), tokenize=False, add_generation_prompt=True)
                 for j in chunk]
        images = [load_image(j[1], j[2], min(opts["max_pixels"], j[6]) if len(j) > 6 else opts["max_pixels"],
                             bool(opts["upscale"]), listed(opts["enhance"], j[4]), rotate=j[5] if len(j) > 5 else 0)
                  for j in chunk]
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(model.device)
        if sureness is not None:
            watch = Sureness()
            extra = {"logits_processor": LogitsProcessorList([watch])}
        out = model.generate(**enc, max_new_tokens=int(max_tokens or opts["max_new_tokens"]), **sampling,
                             repetition_penalty=penalty or opts.get("repetition_penalty", 1.05),
                             no_repeat_ngram_size=int(opts.get("no_repeat_ngram_size", 0) or 0), use_cache=True,
                             **extra)
        written = out[:, enc["input_ids"].shape[1]:]
        for row, (j, text) in enumerate(zip(chunk, processor.batch_decode(written, skip_special_tokens=True))):
            out_texts[j[0]] = text.strip()
            if sureness is not None:
                tokens = written[row].tolist()
                n = next((i + 1 for i, t in enumerate(tokens) if t in stops), len(tokens))   # up to its end mark
                n = max(1, min(n, len(watch.steps)))
                sureness[j[0]] = float(torch.stack(watch.steps[:n])[:, row].mean()) if watch.steps else 0.0
    processor.tokenizer.padding_side = "right"
    return out_texts


def read_pages(model, processor, page_ids, parts, data_dir, doc_ctx, categories, opts, expected, batch, notes=None):
    """{page_id: text} and the set of pages that needed a second reading.

    Each page is read whole or, for a survey spread, as two halves (left, then
    right). A part whose reading looks wrong (nothing, far too short, or stuck
    repeating) is read again as its top and bottom halves; the second reading
    is kept if it looks right. Then the text is cleaned up, and small Kade cards
    get the postmark guess.

    Collections named in --rotations: a short reading, or a page with little ink, is also read
    turned by 90, 180 and 270 degrees, and the reading the model is surest of is kept.
    Collections named in --ink_check: "far too short" is judged from the ink on the page, and
    such a page is read in halves, and a half that is still too short in halves again.

    notes, if given, is a dict that gets {page_id: {rotation, sureness, ink_chars, ...}} for the
    page's first part, for the scores file."""
    turn_set, ink_set = opts.get("rotations", ""), opts.get("ink_check", "")
    jobs, info = [], {}
    for pid in page_ids:
        cat = categories[pid]
        turned, inky = listed(turn_set, cat), listed(ink_set, cat)
        prompt = build_prompt(cat, doc_ctx.get(pid.rsplit("_", 1)[0], ""), sideways=turned)
        for order, r in enumerate(parts[parts.page_id == pid].to_dict("records")):
            box = parse_box(r["box"])
            key = (pid, order)
            path = os.path.join(data_dir, r["image"])
            jobs.append((key, path, box, prompt, cat))
            # With the ink check, the collection's usual length isn't used: a back with only an
            # address is rightly short, and used to be read again in halves for it.
            info[key] = {"expected": 0 if inky else expected.get(f"{cat}/{'half' if box else 'page'}", 0),
                         "size": (int(r.get("width") or 0), int(r.get("height") or 0)),
                         "turned": turned, "inky": inky, "rotation": 0,
                         "ink": ink_chars(path, box) if (turned or inky or listed(opts.get("blocks", ""), cat)
                                                         or listed(opts.get("vote", ""), cat))
                         else 0.0}
    sure = {} if any(i["turned"] for i in info.values()) else None
    texts = generate(model, processor, jobs, opts, batch, sureness=sure)
    upright = dict(texts)   # for --vote
    retried = set()

    if sure is not None:
        again = [((key, deg), path, box, prompt, cat, deg)
                 for key, path, box, prompt, cat in jobs
                 if info[key]["turned"] and info[key]["ink"] < DENSE_CHARS
                 and (flat_len(texts[key]) < ROTATE_BELOW_CHARS or info[key]["ink"] < ROTATE_BELOW_INK)
                 for deg in (90, 180, 270)]
        if again:
            other = generate(model, processor, again, opts, batch, sureness=sure,
                             max_tokens=min(int(opts["max_new_tokens"]), TURNED_MAX_TOKENS))
            margin = float(opts.get("rotation_margin", 0.1))
            for key in sorted({j[0][0] for j in again}):
                best, best_score = 0, (sure[key] if texts[key].strip() else float("-inf"))
                for deg in (90, 180, 270):
                    text = other[(key, deg)]
                    if text.strip() and not looks_wrong(text, 0) and sure[(key, deg)] - margin > best_score:
                        best, best_score = deg, sure[(key, deg)] - margin
                info[key]["tried"] = {0: round(sure[key], 3), **{d: round(sure[(key, d)], 3) for d in (90, 180, 270)}}
                if best:
                    texts[key] = other[(key, best)]
                    sure[key] = sure[(key, best)]
                    info[key]["rotation"] = best
                    retried.add(key[0])

    # Sparse pages (--blocks): each patch of writing read on its own, the way its lines run. A back
    # often has an address one way and a docket another; turning the whole sheet once loses one.
    whole = dict(texts)     # the whole page, the way it was kept (for --vote)
    picked = {}             # {key: [the job of each patch kept, at the way it was read]} (for --vote)
    block_set = opts.get("blocks", "")
    if block_set:
        again, found = [], {}
        for key, path, box, prompt, cat in jobs:
            if not listed(block_set, cat) or box is not None or info[key]["ink"] >= ROTATE_BELOW_INK:
                continue
            patches = ink_blocks(path)
            if not patches:
                continue
            found[key] = patches
            piece_prompt = build_prompt(cat) + "\n" + BLOCK_HINT    # no catalog record: it got copied
            for n, (x0, y0, x1, y1, _) in enumerate(patches):
                cap = (x1 - x0) * (y1 - y0) * BLOCK_MAX_ZOOM ** 2
                again += [((key, "block", n, deg), path, (x0, y0, x1, y1), piece_prompt, cat, deg, cap)
                          for deg in block_turns(x0, y0, x1, y1)]
        if again:
            piece_sure = {}
            pieces_read = generate(model, processor, again, opts, batch, sureness=piece_sure,
                                   max_tokens=min(int(opts["max_new_tokens"]), BLOCK_MAX_TOKENS))
            job_by_key = {j[0]: j for j in again}
            for key, patches in found.items():
                kept = []
                for n, (x0, y0, x1, y1, chars) in enumerate(patches):
                    options = [(piece_sure[(key, "block", n, deg)], pieces_read[(key, "block", n, deg)], deg)
                               for deg in block_turns(x0, y0, x1, y1)]
                    # a reading far longer than the patch's ink is a guess, however sure the model is
                    options = [o for o in options if o[1].strip() and patch_reading_ok(o[1], chars)]
                    if options:
                        best = max(options)
                        kept.append(best[1].strip())
                        picked.setdefault(key, []).append((job_by_key[(key, "block", n, best[2])], chars))
                if kept:
                    texts[key] = "\n".join(kept)
                    info[key]["blocks"] = len(kept)
                    retried.add(key[0])

    # Sparse pages (--vote): the readings above, plus the whole page with the contrast stretch
    # flipped and a few sampled readings of the whole page and of the patches, and the one closest
    # to all the others is kept. Writing that is really there comes out much the same in every
    # reading; a guess ("Mr Dominy / Easthampton / $25") comes out different each time.
    vote_set = opts.get("vote", "")
    voters = [j for j in jobs if listed(vote_set, j[4]) and j[2] is None and info[j[0]]["ink"] < ROTATE_BELOW_INK] \
        if vote_set else []
    if voters:
        n_samples = int(opts.get("vote_samples", 4) or 0)
        heat = float(opts.get("vote_temperature", 0.7) or 0.7)
        cap = min(int(opts["max_new_tokens"]), TURNED_MAX_TOKENS)
        flip = {}
        for enhanced in (True, False):   # the same pages, with the contrast stretch the other way
            group = [(key, path, box, prompt, cat, info[key]["rotation"]) for key, path, box, prompt, cat in voters
                     if bool(listed(opts["enhance"], cat)) == enhanced]
            if group:
                flip.update(generate(model, processor, group, dict(opts, enhance="" if enhanced else "all"), batch,
                                     max_tokens=cap))
        sampled = {}
        if n_samples:
            torch.manual_seed(1234)   # the same samples every time the page is read
            sampled = generate(model, processor, [((key, "whole", i), path, box, prompt, cat, info[key]["rotation"])
                                                  for key, path, box, prompt, cat in voters for i in range(n_samples)],
                               opts, batch, max_tokens=cap, temperature=heat)
            patch_jobs = [((key, "patch", n, i), *job[1:]) for key, *_ in voters
                          for n, (job, _) in enumerate(picked.get(key, [])) for i in range(n_samples)]
            if patch_jobs:
                sampled.update(generate(model, processor, patch_jobs, opts, batch, temperature=heat,
                                        max_tokens=min(int(opts["max_new_tokens"]), BLOCK_MAX_TOKENS)))
        _, normalize = page_cer_function()
        import metric   # importable now: page_cer_function put the repository on the path

        def apart(a, b):
            """0 = the same text, 1 = nothing alike. Both ways the same: the competition's CER divides
            by the label's length and stops at 1, so as a judge it would favour a short reading of one
            patch over a full reading of the page."""
            a, b = normalize(a), normalize(b)
            return metric.levenshtein(a, b) / max(1, len(a), len(b))

        for key, path, box, prompt, cat in voters:
            ink = info[key]["ink"]
            rot = info[key]["rotation"]
            kept_as = "patches" if info[key].get("blocks") else f"turned {rot}" if rot else "whole"
            readings = [(kept_as, texts[key])]                       # first: it wins a tie
            if info[key].get("blocks"):
                readings.append((f"turned {rot}" if rot else "whole", whole[key]))
            if rot:
                readings.append(("upright", upright[key]))
            readings.append(("contrast flipped", flip[key]))
            readings += [("whole sample", sampled[(key, "whole", i)]) for i in range(n_samples)]
            for i in range(n_samples if key in picked else 0):
                bits = [sampled[(key, "patch", n, i)].strip() for n, (_, chars) in enumerate(picked[key])
                        if patch_reading_ok(sampled[(key, "patch", n, i)], chars)]
                readings.append(("patch sample", "\n".join(b for b in bits if b)))
            ok =[(name, t) for name, t in readings if t.strip() and not looks_wrong(t, 0)
                  and (not ink or flat_len(t) <= 4 * ink + 30)]
            if len(ok) < 3:
                continue
            # How far each reading is from the others. The whole-page readings and the patch readings
            # count half each, however many of each there are, so neither way of reading outvotes the
            # other by numbers alone. The kept reading wins a tie, since it comes first.
            view = ["patch" if name.startswith("patch") else "page" for name, _ in ok]
            far = []
            for i, (_, t) in enumerate(ok):
                means = [sum(apart(t, o) for j, (_, o) in enumerate(ok) if j != i and view[j] == v) / n
                         for v in ("page", "patch")
                         for n in [sum(1 for j in range(len(ok)) if j != i and view[j] == v)] if n]
                far.append(sum(means) / len(means))
            best = min(range(len(ok)), key=lambda i: (far[i], i))
            info[key]["voted"] = f"{ok[best][0]} ({far[best]:.2f} apart, {len(ok)} readings)"
            if best != 0:
                texts[key] = ok[best][1]
                retried.add(key[0])

    if opts["retry"]:
        again = []
        for key, path, box, prompt, cat in jobs:
            if info[key]["rotation"] == 0 and not info[key].get("blocks") and not info[key].get("voted") \
                    and looks_wrong(texts[key], info[key]["expected"]):
                w, h = info[key]["size"]
                for half, half_box in zip(("top", "bottom"), top_bottom(box, w, h)):
                    again.append(((key, half), path, half_box, prompt, cat))
        if again:
            second = generate(model, processor, again, opts, batch, opts.get("retry_repetition_penalty") or None)
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

    # Dense pages that stopped early (--ink_check): read in halves, and a half that is still far
    # short of its ink in halves again. Each piece is then about as long as the pages the model
    # reads well, and gets the whole pixel budget to itself.
    job_of = {j[0]: j for j in jobs}
    pieces = {key: [(job_of[key][2], texts[key])] for key in job_of
              if info[key]["inky"] and info[key]["rotation"] == 0 and not info[key].get("blocks")
              and not info[key].get("voted") and all(info[key]["size"])}
    for depth in range(2):
        again, short = [], []
        for key, plist in pieces.items():
            _, path, _, prompt, cat = job_of[key]
            w, h = info[key]["size"]
            for n, (box, text) in enumerate(plist):
                ink = info[key]["ink"] if len(plist) == 1 else ink_chars(path, box)
                if stopped_early(text, ink):
                    halves = top_bottom(box, w, h, overlap=0.03)
                    short.append((key, n, halves))
                    again += [((key, depth, n, half), path, half_box, prompt, cat)
                              for half, half_box in zip(("top", "bottom"), halves)]
        if not again:
            break
        second = generate(model, processor, again, opts, batch)
        split = False
        for key, n, halves in sorted(short, key=lambda x: -x[1]):   # last piece first, so positions hold
            top, bottom = second[(key, depth, n, "top")], second[(key, depth, n, "bottom")]
            joined = join_halves(top, bottom)
            if flat_len(joined) > 1.2 * flat_len(pieces[key][n][1]) and not looks_wrong(joined, 0):
                pieces[key][n:n + 1] = [(halves[0], top), (halves[1], bottom)]
                retried.add(key[0])
                split = True
        if not split:
            break
    for key, plist in pieces.items():
        if len(plist) > 1:
            text = plist[0][1]
            for _, more in plist[1:]:
                text = join_halves(text, more)
            texts[key] = text

    if opts.get("compare_adapter"):
        # A second reading with another adapter (e.g. another epoch). Two adapters rarely run away
        # on the same page, so when the main reading is much longer, it is most likely the runaway.
        model.set_adapter("compare")
        other = generate(model, processor, jobs, opts, batch)
        model.set_adapter("default")
        ratio = float(opts.get("compare_ratio", 1.5))
        for key in texts:
            a, b = len(" ".join(texts[key].split())), len(" ".join(other[key].split()))
            if b > 0 and a > ratio * b:
                texts[key] = other[key]
                retried.add(key[0])

    result = {}
    for pid in page_ids:
        keys = sorted(k for k in texts if k[0] == pid)
        text = clean_output("\n".join(texts[k] for k in keys), categories[pid])
        if listed(opts.get("catalog_fix", ""), categories[pid]):
            names = catalog_names_for(data_dir, opts["catalog_fix"]).get(pid.rsplit("_", 1)[0], set())
            text = catalog_fix(text, names)
        if listed(opts.get("cut_repeats", ""), categories[pid]):
            text = cut_at_repeat(text, int(opts.get("repeat_chars", 22)))
        if opts["postmark_guess"] and keys:
            w, h = info[keys[0]]["size"]
            text = add_postmark_guess(text, categories[pid], w, h)
        result[pid] = text
        if notes is not None and keys:
            first = info[keys[0]]
            notes[pid] = {"rotation": first["rotation"], "ink_chars": round(first["ink"]),
                          "sureness": round(sure[keys[0]], 3) if sure is not None and keys[0] in sure else "",
                          "rotations_tried": json.dumps(first["tried"]) if "tried" in first else "",
                          "blocks": first.get("blocks", ""), "vote": first.get("voted", "")}
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
    notes = {}
    preds, retried = read_pages(model, processor, list(dev.page_id), parts, args.data_dir, doc_ctx,
                                dict(zip(dev.page_id, dev.category)), opts, expected, args.eval_batch, notes)
    rows = []
    for r in dev.to_dict("records"):
        pred, ref = preds[r["page_id"]], r["text"]
        ratio = len(normalize(pred)) / max(1, len(normalize(ref)))
        rows.append({"page_id": r["page_id"], "doc_id": r.get("doc_id", ""), "category": r["category"],
                     "label_source": r.get("label_source", ""),
                     "cer": page_cer(pred, ref), "cer_no_postmark": page_cer(without_postmark(pred), without_postmark(ref)),
                     "length_ratio": round(ratio, 2),
                     "problem": "too short" if ratio < 0.7 else "too long" if ratio > 1.1 else "",
                     "read_twice": r["page_id"] in retried, **notes.get(r["page_id"], {}),
                     "prediction": pred, "reference": ref})
    df = pd.DataFrame(rows)
    by_cat = df.groupby("category")["cer"].mean()
    print(f"[dev {label}] {len(df)} pages in {(time.time() - start) / 60:.1f} min: "
          + ", ".join(f"{c} {v:.3f}" for c, v in by_cat.items()) + f"; mean of collections {by_cat.mean():.3f}"
          + (f"; {len(retried)} read twice" if retried else ""), flush=True)
    if "rotation" in df.columns and (df.rotation != 0).any():
        print("    kept a turned reading: " + ", ".join(f"{r.page_id} ({int(r.rotation)} degrees)"
                                                        for r in df[df.rotation != 0].itertuples()), flush=True)
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
    def __init__(self, items, data_dir, doc_ctx, opts, augment_setting, seed, rotate_setting=""):
        self.rows = items.to_dict("records")
        self.data_dir, self.doc_ctx, self.opts = data_dir, doc_ctx, opts
        self.augment_setting, self.rotate_setting = augment_setting, rotate_setting
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        cat = r["category"]
        # Half the time, a page of these collections is turned on its side or upside down, with
        # the same label: the model has to read the writing whichever way up it is, which the
        # backs of folded letters need, and can't lean on where a remembered page's lines were.
        turn = self.rng.choice((90, 180, 270)) if listed(self.rotate_setting, cat) and self.rng.random() < 0.5 else 0
        img = load_image(os.path.join(self.data_dir, r["image"]), parse_box(r["box"]), self.opts["max_pixels"],
                         bool(self.opts["upscale"]), listed(self.opts["enhance"], cat), rotate=turn)
        if listed(self.augment_setting, cat):
            img = augment(img, self.rng, cat)
        prompt = build_prompt(cat, self.doc_ctx.get(r["doc_id"], ""),
                              sideways=listed(self.opts.get("rotations", ""), cat))
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


def as_training_items(pages, parts):
    """Rows like train_items.csv for labelled pages that prepare_data.py left out of it (the dev
    pages): a survey spread whose label splits becomes two images, as there."""
    rows = []
    for r in pages.to_dict("records"):
        prs = parts[parts.page_id == r["page_id"]].to_dict("records")
        if not prs:
            continue
        text = training_text(r["text"], r["category"])
        halves = split_survey_label(text) if len(prs) == 2 else None
        base = {"page_id": r["page_id"], "doc_id": r["doc_id"], "category": r["category"]}
        if halves:
            rows += [{**base, "image": p["image"], "box": p["box"], "text": h} for p, h in zip(prs, halves)]
        else:
            rows.append({**base, "image": prs[0]["image"], "box": "", "text": text})
    return pd.DataFrame(rows, columns=["page_id", "doc_id", "category", "image", "box", "text"])


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
    doc_ctx = catalog_context(args.data_dir, args.rich_context) if args.use_metadata else {}
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
        # The usual dev pages of those collections aren't scored now, so the ones outside the
        # held-out documents are trained on. (Every Kade dev page is in kade_112, which each Kade
        # split holds out, so this changes nothing there; it matters for Dominy, where the
        # held-out documents are chosen to look like the test pages.)
        freed = dev[dev.category.isin(set(held_pages.category)) & ~dev.doc_id.isin(held)]
        if len(freed):
            items = pd.concat([items, as_training_items(freed, parts)], ignore_index=True)
            print(f"training on {len(freed)} of the usual dev pages, since other documents are held out: "
                  + ", ".join(freed.page_id), flush=True)
        dev = pd.concat([dev[~dev.category.isin(set(held_pages.category))],
                         held_pages[["page_id", "doc_id", "category", "label_source", "text"]]], ignore_index=True)
        print(f"held out: {', '.join(f'{d} ({n} pages)' for d, n in held_pages.doc_id.value_counts().sort_index().items())}",
              flush=True)
    if args.categories != "all":
        # .astype(bool): with no rows left (every document held out), map() gives an empty column
        # that isn't true/false, and pandas would take it as "no columns".
        items = items[items.category.map(lambda c: listed(args.categories, c)).astype(bool)]
        dev = dev[dev.category.map(lambda c: listed(args.categories, c)).astype(bool)]
        if not len(items) and args.epochs > 0:
            raise SystemExit(f"No training images in collections '{args.categories}'.")
    if args.train_on_dev:
        if args.holdout_docs:
            raise SystemExit("--train_on_dev and --holdout_docs don't go together: the held-out pages are the dev pages.")
        extra = as_training_items(dev, parts)
        items = pd.concat([items, extra], ignore_index=True)
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
        raise SystemExit("--adapter_dir only scores a saved adapter; use it with --epochs 0. "
                         "To train further from an adapter, use --init_adapter.")
    if args.compare_adapter and args.epochs > 0:
        raise SystemExit("--compare_adapter is for scoring (--epochs 0), not training.")
    if args.init_adapter:
        if args.epochs <= 0:
            raise SystemExit("--init_adapter starts training from an adapter; it needs --epochs above 0.")
        if not os.path.exists(os.path.join(args.init_adapter, "adapter_config.json")):
            raise SystemExit(f"No adapter in {args.init_adapter}")
        if args.train_vision:
            raise SystemExit("--train_vision 1 changes which layers the adapter has, so it can't continue a saved "
                             "adapter. Start a fresh run for it (no --init_adapter).")
    if args.epochs > 0 and os.path.exists(os.path.join(args.out_dir, "summary.json")):
        # A finished run. Launching it again used to "resume" from its last checkpoint, train for
        # 0 minutes, and overwrite its summary and curve with empty ones.
        raise SystemExit(f"This run already finished ({args.out_dir}). Its results are kept. To train again, "
                         "change a setting, or delete the folder to start it over.")
    model, processor, dtype = load_model(args.model_id, args.max_pixels, args.adapter_dir or None,
                                         args.compare_adapter or None)
    summary = {"model_id": args.model_id, "epochs": args.epochs, "lr": args.lr, "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
               "train_vision": args.train_vision, "augment": args.augment, "rotate_train": args.rotate_train,
               "oversample": args.oversample, "effective_batch": args.batch_size * args.grad_accum,
               "train_images_per_epoch": len(items), "dev_pages": len(dev),
               "categories": args.categories, "holdout_docs": args.holdout_docs,
               "trained_on_docs": sorted(set(items.doc_id)), "adapter_dir": args.adapter_dir,
               "train_on_dev": args.train_on_dev, "init_adapter": args.init_adapter,
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
        if args.init_adapter:
            # Carry on from what a saved adapter already knows. Its layers and size come from the
            # saved adapter, so lora_r, lora_alpha and train_vision don't apply.
            from peft import PeftModel
            print(f"training on from the adapter in {args.init_adapter}", flush=True)
            model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        else:
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
        class LabelOnlyLoss(Trainer):
            """The same loss as the model's own, computed only where there is something to learn.
            The model's own forward scores every position (prompt, ~1000 image pieces and the
            transcription) against the whole vocabulary, in float32: about 1.1 GB per page for
            Qwen3-VL's 152k words, and that is where the first 8B run ran out of memory. Only the
            transcription's positions count towards the loss, so only those are scored here."""

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                labels = inputs.pop("labels")
                lm = model.get_base_model() if hasattr(model, "get_base_model") else model
                hidden = lm.model(**inputs, use_cache=False).last_hidden_state
                targets = labels[:, 1:]            # position t predicts token t + 1
                keep = targets != -100
                logits = lm.lm_head(hidden[:, :-1][keep]).float()
                loss = torch.nn.functional.cross_entropy(logits, targets[keep], reduction="sum")
                # Like the model's own loss: per token, over every token of the accumulated batch.
                loss = loss / (num_items_in_batch if num_items_in_batch is not None else keep.sum().clamp(min=1))
                return (loss, None) if return_outputs else loss

        trainer = LabelOnlyLoss(model=model, args=targs,
                          train_dataset=PageDataset(items, args.data_dir, doc_ctx, opts, args.augment, args.seed,
                                                    args.rotate_train),
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
