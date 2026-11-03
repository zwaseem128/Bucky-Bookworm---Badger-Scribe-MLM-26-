"""Read pages with a run from Notebook 2 and save the text.

Run by 3_predict_test.ipynb:

    python transcribe.py --data_dir <data> --pages <csv with page_id> --out <csv> --run_dir <run folder>

Pages are read exactly the way that run's dev pages were scored: its adapter
(if it has one; the baseline run doesn't), its reading settings from
summary.json (image size, enlarging small images, contrast, the second
reading of a part that looks wrong, the postmark guess), survey spreads in two
halves, the same prompt and the same clean-up. Pages already in --out are
skipped, so an interrupted run carries on where it stopped.
"""

import argparse
import csv
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import category_of, load_metadata  # noqa: E402
from finetune_churro import load_model, read_pages  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--pages", required=True, help="CSV with a page_id column")
    p.add_argument("--out", required=True)
    p.add_argument("--run_dir", required=True, help="a run folder from Notebook 2")
    p.add_argument("--model_id", default=None, help="default: the run's model")
    p.add_argument("--batch", type=int, default=4, help="images read at once")
    p.add_argument("--chunk", type=int, default=12, help="pages read between saves")
    args = p.parse_args()

    summary = json.load(open(os.path.join(args.run_dir, "summary.json")))
    opts = summary["reading"]
    has_adapter = os.path.exists(os.path.join(args.run_dir, "adapter_config.json"))
    print(f"reading as run {os.path.basename(args.run_dir)} scored its dev pages: {opts}", flush=True)

    pages = list(pd.read_csv(args.pages, keep_default_na=False, encoding="utf-8-sig")["page_id"])
    done = set(pd.read_csv(args.out, keep_default_na=False)["page_id"]) if os.path.exists(args.out) else set()
    todo = [pid for pid in pages if pid not in done]
    print(f"{len(pages)} pages, {len(done)} already done, {len(todo)} to do", flush=True)
    if not todo:
        return

    parts = pd.read_csv(os.path.join(args.data_dir, "parts.csv"), keep_default_na=False)
    expected_path = os.path.join(args.data_dir, "expected_chars.json")
    expected = json.load(open(expected_path)) if os.path.exists(expected_path) else {}
    doc_ctx = load_metadata(os.path.join(args.data_dir, "metadata.csv"))
    model, processor, _ = load_model(args.model_id or summary["model_id"], opts["max_pixels"],
                                     args.run_dir if has_adapter else None)
    new = not os.path.exists(args.out)
    start, read_twice = time.time(), 0
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["page_id", "text"])
        for s in range(0, len(todo), args.chunk):
            chunk = todo[s:s + args.chunk]
            texts, retried = read_pages(model, processor, chunk, parts, args.data_dir, doc_ctx,
                                        {pid: category_of(pid) for pid in chunk}, opts, expected, args.batch)
            read_twice += len(retried)
            for pid in chunk:
                writer.writerow([pid, texts[pid]])
            f.flush()
            n = min(s + args.chunk, len(todo))
            left = (time.time() - start) / n * (len(todo) - n)
            print(f"[{n}/{len(todo)}] pages read, {read_twice} needed a second reading "
                  f"(about {left / 60:.0f} min left)", flush=True)


if __name__ == "__main__":
    main()
