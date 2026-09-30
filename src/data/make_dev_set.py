"""Pick the dev set: a fixed handful of labelled train pages we hold back for scoring.

Every idea we try is scored on these same pages, so results are comparable
over the whole competition. Run once, commit src/data/dev_ids.csv, and never
regenerate it. If dev_ids.csv already exists, the script keeps it and only
rebuilds your local answer key for those pages; --force re-picks.

How pages are chosen, per category:
  * Whole documents go into the dev set, never part of one, so pages of one
    letter can't be split between dev and training. This is the same rule
    Kaggle uses for its test set.
  * Exception: a category that is a single document (the survey notes are one
    bound notebook) is split page by page, again matching Kaggle.
  * Aim for --per-category pages, but never take more than --max-fraction of
    a category, and always leave at least one document for training.

Only train.csv is read. Test pages are never touched.

    python src/data/make_dev_set.py --train data/train.csv

Also writes data/dev_solution.csv (dev pages with their labels), so any
predictions file can be scored on the dev set with Kaggle's metric:

    python metric.py --solution data/dev_solution.csv --submission my_preds.csv
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO / "data" / "train.csv"
DEFAULT_OUT = REPO / "src" / "data" / "dev_ids.csv"
DEFAULT_SOLUTION = REPO / "data" / "dev_solution.csv"
DEV_COLUMNS = ["page_id", "doc_id", "category", "label_source"]


def read_train(path):
    # csv, not pandas: pandas would read a transcribed "NA" as a missing value.
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"page_id", "doc_id", "text", "category", "label_source"} - set(rows[0] if rows else {})
    if missing:
        sys.exit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def pick_dev_pages(rows, per_category=15, max_fraction=0.25, seed=0):
    """Return the set of page_ids that go in the dev set."""
    rng = random.Random(seed)
    pages_by_doc = defaultdict(list)
    docs_by_cat = defaultdict(set)
    for r in rows:
        pages_by_doc[r["doc_id"]].append(r["page_id"])
        docs_by_cat[r["category"]].add(r["doc_id"])

    dev = set()
    for cat in sorted(docs_by_cat):
        # Sort before shuffling so the result doesn't depend on row order in train.csv.
        docs = sorted(docs_by_cat[cat])
        n_pages = sum(len(pages_by_doc[d]) for d in docs)
        cap = max(1, math.floor(max_fraction * n_pages))
        target = min(per_category, cap)

        if len(docs) == 1:
            pages = sorted(pages_by_doc[docs[0]])
            if len(pages) < 2:
                print(f"warning: {cat} has only {len(pages)} page; nothing held out", file=sys.stderr)
                continue
            dev.update(rng.sample(pages, min(target, len(pages) - 1)))
            continue

        rng.shuffle(docs)
        chosen, count = [], 0
        for d in docs[:-1]:  # the last shuffled doc always stays in training
            size = len(pages_by_doc[d])
            if count + size <= cap:
                chosen.append(d)
                count += size
            if count >= target:
                break
        if not chosen:
            # Every document is bigger than the cap; take the smallest one anyway.
            chosen = [min(docs, key=lambda d: (len(pages_by_doc[d]), d))]
        for d in chosen:
            dev.update(pages_by_doc[d])
    return dev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN, help="path to Kaggle's train.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--solution-out", type=Path, default=DEFAULT_SOLUTION)
    ap.add_argument("--per-category", type=int, default=15)
    ap.add_argument("--max-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing dev_ids.csv")
    args = ap.parse_args(argv)

    rows = read_train(args.train)
    reuse = args.out.exists() and not args.force
    if reuse:
        with open(args.out, newline="", encoding="utf-8") as f:
            dev = {r["page_id"] for r in csv.DictReader(f)}
        unknown = dev - {r["page_id"] for r in rows}
        if unknown:
            sys.exit(f"{len(unknown)} dev pages are not in {args.train}, e.g. {sorted(unknown)[0]}. "
                     "Is this the right train.csv?")
    else:
        dev = pick_dev_pages(rows, args.per_category, args.max_fraction, args.seed)
    dev_rows = sorted((r for r in rows if r["page_id"] in dev), key=lambda r: r["page_id"])

    if not reuse:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DEV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(dev_rows)

    args.solution_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.solution_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page_id", "text", "category"], extrasaction="ignore")
        w.writeheader()
        w.writerows(dev_rows)

    print(f"{'category':<20}{'train pages':>12}{'dev pages':>11}{'dev docs':>10}{'human':>7}{'silver':>8}")
    for cat in sorted({r["category"] for r in rows}):
        in_cat = [r for r in rows if r["category"] == cat]
        d = [r for r in dev_rows if r["category"] == cat]
        human = sum(r["label_source"] == "human" for r in d)
        print(f"{cat:<20}{len(in_cat):>12}{len(d):>11}{len({r['doc_id'] for r in d}):>10}"
              f"{human:>7}{len(d) - human:>8}")
    if reuse:
        print(f"\nkept the existing dev set in {args.out.name} ({len(dev_rows)} pages); "
              f"rebuilt {args.solution_out.name}")
    else:
        print(f"\nwrote {args.out.name} ({len(dev_rows)} pages) and {args.solution_out.name}")


if __name__ == "__main__":
    main()
