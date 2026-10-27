"""Build the training job's input folder from the kraken line crops.

Run from the notebooks. It takes kade_lines.zip (made by
notebooks/kade_line_segmentation.ipynb) and train.csv, and writes:

    lines.zip         the crops, unchanged
    train_lines.csv   page_id, doc_id, crop, text   crop i of a page paired with label line i
    dev_pages.csv     page_id, doc_id, text         pages that can be scored, with their full labels
    test_pages.csv    page_id                       Kade pages with crops but no label (Kaggle test)

Only pages where kraken found exactly as many lines as the label has are used
for training: there, crop i and label line i are the same line. Scored pages
are read as whole pages, so they don't need to match.

Which documents are scored and which are trained on is decided by the training
script (--holdout_docs), so one folder serves both the experiments and the
final run. The one fixed rule is here: the team's dev document is never in
train_lines.csv.
"""

import os
import shutil
import sys
import zipfile
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from kurrent_common import label_lines  # noqa: E402

# The dev set from src/data/dev_ids.csv: for Kade it is the whole kade_112 document.
DEV_DOCS = ("kade_112",)
# A second document held out while we experiment. The Kaggle test pages come from
# five Kade documents that have no pages in train.csv at all, so the question
# that matters is how well the model reads a document it has never seen.
# kade_112 alone (11 small postcards) is too thin to answer that; kade_007 is
# the smallest of the three big documents (77 pages, full-size scans).
HOLDOUT_DOCS = ("kade_007",)


def build(zip_path, train_csv, out_dir, dev_docs=DEV_DOCS, holdout_docs=HOLDOUT_DOCS):
    with zipfile.ZipFile(zip_path) as z:
        crops = defaultdict(list)
        for name in z.namelist():
            parts = name.split("/")
            # kade_lines/<page_id>/<NNN>.png; skips line_counts.csv and unfinished .partial folders
            if len(parts) == 3 and parts[0] == "kade_lines" and parts[2].endswith(".png") \
                    and not parts[1].endswith(".partial"):
                crops[parts[1]].append(name)
    labels = pd.read_csv(train_csv, keep_default_na=False, encoding="utf-8-sig")
    kade = labels[labels["category"] == "kade_letters"]

    rows, matched, mismatched, missing = [], 0, [], []
    for r in kade[~kade["doc_id"].isin(dev_docs)].to_dict("records"):
        page_crops = sorted(crops.get(r["page_id"], []))
        ref = label_lines(r["text"])
        if not page_crops:
            missing.append(r["page_id"])
        elif len(page_crops) != len(ref):
            mismatched.append(r["page_id"])
        else:
            matched += 1
            rows += [{"page_id": r["page_id"], "doc_id": r["doc_id"], "crop": c, "text": t}
                     for c, t in zip(page_crops, ref)]
    scored = kade[kade["doc_id"].isin(tuple(dev_docs) + tuple(holdout_docs))][["page_id", "doc_id", "text"]]
    no_scored_crops = sorted(set(scored["page_id"]) - set(crops))
    # Every Kade page that was cut into lines but has no label is a test page.
    test_pages = sorted(set(crops) - set(kade["page_id"]))

    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(zip_path) != os.path.abspath(os.path.join(out_dir, "lines.zip")):
        shutil.copy(zip_path, os.path.join(out_dir, "lines.zip"))
    train = pd.DataFrame(rows, columns=["page_id", "doc_id", "crop", "text"])
    train.to_csv(os.path.join(out_dir, "train_lines.csv"), index=False)
    scored.to_csv(os.path.join(out_dir, "dev_pages.csv"), index=False)
    pd.DataFrame({"page_id": test_pages}).to_csv(os.path.join(out_dir, "test_pages.csv"), index=False)

    train_pages = matched + len(mismatched) + len(missing)
    print(f"training pages: {matched} of {train_pages} usable ({matched / max(train_pages, 1):.0%}), "
          f"{len(rows)} lines")
    print(f"  line counts don't match: {len(mismatched)} pages; not segmented yet: {len(missing)} pages")
    by_doc = train.groupby("doc_id").page_id.nunique().to_dict() if len(train) else {}
    print(f"  usable pages by document: {by_doc}")
    print(f"pages for scoring: {len(scored)} ({len(scored) - len(no_scored_crops)} segmented), "
          f"from {', '.join(sorted(scored['doc_id'].unique()))}")
    if no_scored_crops:
        print(f"  WARNING: no crops for {len(no_scored_crops)} of them; they will score as blank (CER 1.0)")
    print(f"test pages ready to predict: {len(test_pages)} (Kaggle has 141 Kade test pages)")
    return {"matched": matched, "mismatched": mismatched, "missing": missing, "lines": len(rows),
            "test_pages": len(test_pages)}
