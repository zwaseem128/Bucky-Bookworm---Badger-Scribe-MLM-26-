"""Read pages with CHURRO-3B (and a trained adapter) and save the text.

Run by 3_predict_test.ipynb:

    python transcribe.py --data_dir <data> --pages <csv with page_id> --out <csv> [--adapter_dir <run folder>]

Pages are read exactly as in training and dev scoring (survey spreads in two
halves, the same prompt, the same clean-up). Pages already in --out are
skipped, so an interrupted run carries on where it stopped. Without
--adapter_dir it reads with CHURRO as it is.
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MAX_PIXELS, MODEL_ID, category_of, load_metadata  # noqa: E402
from finetune_churro import load_model, read_pages  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pages", required=True, help="CSV with a page_id column")
    p.add_argument("--out", required=True)
    p.add_argument("--adapter_dir", default=None)
    p.add_argument("--model_id", default=MODEL_ID)
    p.add_argument("--max_pixels", type=int, default=MAX_PIXELS)
    p.add_argument("--max_new_tokens", type=int, default=1536)
    p.add_argument("--batch", type=int, default=4, help="images read at once")
    p.add_argument("--chunk", type=int, default=12, help="pages read between saves")
    args = p.parse_args()

    pages = list(pd.read_csv(args.pages, keep_default_na=False, encoding="utf-8-sig")["page_id"])
    done = set(pd.read_csv(args.out, keep_default_na=False)["page_id"]) if os.path.exists(args.out) else set()
    todo = [pid for pid in pages if pid not in done]
    print(f"{len(pages)} pages, {len(done)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    parts = pd.read_csv(os.path.join(args.data_dir, "parts.csv"), keep_default_na=False)
    doc_ctx = load_metadata(os.path.join(args.data_dir, "metadata.csv"))
    model, processor, _ = load_model(args.model_id, args.max_pixels, args.adapter_dir)
    new = not os.path.exists(args.out)
    start = time.time()
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["page_id", "text"])
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s:s + args.chunk]
            texts = read_pages(model, processor, chunk, parts, args.data_dir, doc_ctx,
                               {pid: category_of(pid) for pid in chunk}, args.max_pixels,
                               args.max_new_tokens, args.batch, log_every=0)
            for pid in chunk:
                writer.writerow([pid, texts[pid]])
            f.flush()
            n = min(s + args.chunk, len(todo))
            left = (time.time() - start) / n * (len(todo) - n)
            print(f"[{n}/{len(todo)}] pages read (about {left / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
