"""Pair each image from make_crops.py with its part of the transcription.

Run by 1_prepare_lines.ipynb, after make_crops.py. Reads <data>/kade_lines/
(the images, crops.csv and crop_settings.json) and train.csv, and writes into
<data>:

    line_counts.csv   page_id, kraken_lines, label_lines, match   (every Kade page with a label)
    train_lines.csv   page_id, doc_id, crop, text                 one row per training image
    dev_pages.csv     page_id, text                               the dev pages and their full labels

How the text is matched depends on how the pages were cut:

    pages    each page image gets its whole transcription, so every page is used.
    lines    image i gets line i of the transcription. That's only safe when kraken
             found exactly as many lines as the transcription has, so other pages
             are left out of training.
    groups   the same, with each image getting the lines it holds (e.g. lines 4-6).

The dev pages are scored as whole pages, so they don't need to match.
"""

import json
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
    mode = json.load(open(os.path.join(lines_dir, "crop_settings.json")))["mode"]
    crops = pd.read_csv(os.path.join(lines_dir, "crops.csv"), keep_default_na=False)
    labels = pd.read_csv(train_csv, keep_default_na=False, encoding="utf-8-sig")
    kade = labels[labels["page_id"].str.startswith("kade_")]

    counts, rows = [], []
    for r in kade.to_dict("records"):
        page_crops = crops[crops.page_id == r["page_id"]]
        ref = label_lines(r["text"])
        if mode == "pages":
            found = len(ref) if len(page_crops) else 0   # nothing to count: the page is one image
        else:
            found = int(page_crops["n_lines"].astype(int).sum()) if len(page_crops) else 0
        is_dev = r["doc_id"] in dev_docs
        match = bool(len(page_crops)) and found == len(ref)
        counts.append({"page_id": r["page_id"], "doc_id": r["doc_id"], "dev": is_dev,
                       "kraken_lines": found, "label_lines": len(ref), "match": match})
        if is_dev or not match:
            continue
        for c in page_crops.to_dict("records"):
            if mode == "pages":
                text = "\n".join(ref)
            else:
                first, n = int(c["first_line"]), int(c["n_lines"])
                text = "\n".join(ref[first:first + n])
            rows.append({"page_id": r["page_id"], "doc_id": r["doc_id"],
                         "crop": f"kade_lines/{c['crop']}", "text": text})

    counts = pd.DataFrame(counts)
    counts.to_csv(os.path.join(data_dir, "line_counts.csv"), index=False)
    pd.DataFrame(rows, columns=["page_id", "doc_id", "crop", "text"]).to_csv(
        os.path.join(data_dir, "train_lines.csv"), index=False)
    kade[kade["doc_id"].isin(dev_docs)][["page_id", "text"]].to_csv(
        os.path.join(data_dir, "dev_pages.csv"), index=False)

    train = counts[~counts.dev]
    dev = counts[counts.dev]
    missing = train[train.kraken_lines == 0]
    print(f"Cut as '{mode}'. Training pages: {int(train.match.sum())} of {len(train)} usable "
          f"({train.match.mean():.0%}), {len(rows)} training images.")
    if mode != "pages":
        print(f"  Line counts don't match: {int((~train.match).sum()) - len(missing)} pages. "
              f"Not segmented: {len(missing)} pages.")
    print(f"Dev pages: {len(dev)}, {int((dev.kraken_lines > 0).sum())} ready.")
    if (dev.kraken_lines == 0).any():
        print(f"  WARNING: no images for {', '.join(dev[dev.kraken_lines == 0].page_id)}. "
              "They will score as blank (CER 1.0).")
    return counts
