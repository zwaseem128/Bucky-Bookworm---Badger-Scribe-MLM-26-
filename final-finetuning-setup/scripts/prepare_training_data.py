"""Pair each line crop with its line of the label, ready for training.

Run by 1_prepare_lines.ipynb, after segment_lines.py. Reads <data>/kade_lines/
and train.csv, and writes into <data>:

    line_counts.csv   page_id, kraken_lines, label_lines, match   (every Kade page with a label)
    train_lines.csv   page_id, doc_id, crop, text                 one row per training line
    dev_pages.csv     page_id, text                               the dev pages and their full labels

Only pages where kraken found exactly as many lines as the label has are used
for training: there, crop i and label line i are the same line. The dev pages
are scored as whole pages, so they don't need to match.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kurrent_common import label_lines  # noqa: E402

# The team's dev set (src/data/dev_ids.csv on lokesh/data-dev-set): for Kade it
# is the whole kade_112 document. These pages are never trained on.
DEV_DOCS = ("kade_112",)


def build(data_dir, train_csv, dev_docs=DEV_DOCS):
    lines_dir = os.path.join(data_dir, "kade_lines")
    labels = pd.read_csv(train_csv, keep_default_na=False, encoding="utf-8-sig")
    kade = labels[labels["category"] == "kade_letters"]

    counts, rows = [], []
    for r in kade.to_dict("records"):
        folder = os.path.join(lines_dir, r["page_id"])
        crops = sorted(f for f in os.listdir(folder) if f.endswith(".png")) if os.path.isdir(folder) else []
        ref = label_lines(r["text"])
        is_dev = r["doc_id"] in dev_docs
        counts.append({"page_id": r["page_id"], "doc_id": r["doc_id"], "dev": is_dev,
                       "kraken_lines": len(crops), "label_lines": len(ref),
                       "match": bool(crops) and len(crops) == len(ref)})
        if not is_dev and counts[-1]["match"]:
            rows += [{"page_id": r["page_id"], "doc_id": r["doc_id"],
                      "crop": f"kade_lines/{r['page_id']}/{c}", "text": t}
                     for c, t in zip(crops, ref)]

    counts = pd.DataFrame(counts)
    counts.to_csv(os.path.join(data_dir, "line_counts.csv"), index=False)
    pd.DataFrame(rows, columns=["page_id", "doc_id", "crop", "text"]).to_csv(
        os.path.join(data_dir, "train_lines.csv"), index=False)
    kade[kade["doc_id"].isin(dev_docs)][["page_id", "text"]].to_csv(
        os.path.join(data_dir, "dev_pages.csv"), index=False)

    train = counts[~counts.dev]
    dev = counts[counts.dev]
    missing = train[train.kraken_lines == 0]
    print(f"Training pages: {int(train.match.sum())} of {len(train)} usable "
          f"({train.match.mean():.0%}), {len(rows)} lines.")
    print(f"  Line counts don't match: {int((~train.match).sum()) - len(missing)} pages. "
          f"Not segmented: {len(missing)} pages.")
    print(f"Dev pages: {len(dev)}, {int((dev.kraken_lines > 0).sum())} segmented.")
    if (dev.kraken_lines == 0).any():
        print(f"  WARNING: no lines for {', '.join(dev[dev.kraken_lines == 0].page_id)}. "
              "They will score as blank (CER 1.0).")
    return counts
