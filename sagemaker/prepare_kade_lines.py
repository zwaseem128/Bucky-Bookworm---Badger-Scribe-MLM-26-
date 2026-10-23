"""Build the training job's input folder from the kraken line crops.

Run from notebooks/sagemaker_kade_lora.ipynb. It takes kade_lines.zip (made by
notebooks/kade_line_segmentation.ipynb) and train.csv, and writes:

    lines.zip         the crops, unchanged
    train_lines.csv   page_id, doc_id, crop, text   crop i of a page paired with label line i
    dev_pages.csv     page_id, text                 the dev pages, with their full labels

Only pages where kraken found exactly as many lines as the label has are used
for training: there, crop i and label line i are the same line. Dev pages are
scored as whole pages, so they don't need to match.
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


def build(zip_path, train_csv, out_dir, dev_docs=DEV_DOCS):
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
    dev = kade[kade["doc_id"].isin(dev_docs)][["page_id", "text"]]
    no_dev_crops = sorted(set(dev["page_id"]) - set(crops))

    os.makedirs(out_dir, exist_ok=True)
    if os.path.abspath(zip_path) != os.path.abspath(os.path.join(out_dir, "lines.zip")):
        shutil.copy(zip_path, os.path.join(out_dir, "lines.zip"))
    pd.DataFrame(rows, columns=["page_id", "doc_id", "crop", "text"]).to_csv(
        os.path.join(out_dir, "train_lines.csv"), index=False)
    dev.to_csv(os.path.join(out_dir, "dev_pages.csv"), index=False)

    train_pages = matched + len(mismatched) + len(missing)
    print(f"training pages: {matched} of {train_pages} usable ({matched / max(train_pages, 1):.0%}), "
          f"{len(rows)} lines")
    print(f"  line counts don't match: {len(mismatched)} pages; not segmented yet: {len(missing)} pages")
    print(f"dev pages: {len(dev)} ({len(dev) - len(no_dev_crops)} segmented)")
    if no_dev_crops:
        print(f"  WARNING: no crops for {', '.join(no_dev_crops)}; they will score as blank (CER 1.0)")
    return {"matched": matched, "mismatched": mismatched, "missing": missing, "lines": len(rows)}
