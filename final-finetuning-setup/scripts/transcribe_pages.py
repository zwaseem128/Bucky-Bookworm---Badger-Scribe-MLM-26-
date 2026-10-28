"""Transcribe whole Kade pages with a fine-tuned adapter: read every line, join them.

Run by 3_transcribe.ipynb:

    python transcribe_pages.py --lines_dir <data>/kade_lines --pages pages.csv --out predictions.csv \
        [--adapter_dir <run>/model]

pages.csv needs a page_id column, and optionally a text column with the right
answer: then each page's CER is computed too. Leave out --adapter_dir to use
the Kurrent adapter as it was before our training. Pages already in --out are
skipped, so an interrupted run continues where it stopped.
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finetune_qwen_kurrent_lora import load_for_inference, transcribe_lines  # noqa: E402
from kurrent_common import BASE_MODEL, START_ADAPTER, load_line_image, page_cer  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines_dir", required=True)
    p.add_argument("--pages", required=True, help="CSV with a page_id column (and optionally text)")
    p.add_argument("--out", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--start_adapter", default=START_ADAPTER)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()

    pages = pd.read_csv(args.pages, keep_default_na=False, encoding="utf-8-sig")
    done = set(pd.read_csv(args.out, keep_default_na=False)["page_id"]) if os.path.exists(args.out) else set()
    todo = pages[~pages.page_id.isin(done)]
    print(f"{len(pages)} pages, {len(done)} already done, {len(todo)} to do", flush=True)
    if todo.empty:
        return

    model, processor = load_for_inference(args.adapter_dir, args.base_model, args.start_adapter)
    has_ref = "text" in pages.columns
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["page_id", "text"] + (["cer"] if has_ref else []))
        for i, r in enumerate(todo.to_dict("records"), 1):
            start = time.time()
            folder = os.path.join(args.lines_dir, r["page_id"])
            crops = sorted(c for c in os.listdir(folder) if c.endswith(".png")) if os.path.isdir(folder) else []
            if not crops:
                print(f"  {r['page_id']}: no line images, written as blank", flush=True)
            lines = transcribe_lines(model, processor, [load_line_image(os.path.join(folder, c)) for c in crops],
                                     args.max_new_tokens, args.batch)
            text = "\n".join(lines)
            row = [r["page_id"], text]
            note = ""
            if has_ref:
                cer = page_cer(text, r["text"])
                row.append(round(cer, 4))
                note = f", CER {cer:.3f}"
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(todo)}] {r['page_id']}: {len(crops)} lines in {time.time() - start:.0f}s{note}", flush=True)


if __name__ == "__main__":
    main()
