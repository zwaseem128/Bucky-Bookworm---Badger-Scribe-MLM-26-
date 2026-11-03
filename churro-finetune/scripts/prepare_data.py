"""Decide how every page is read, and pair the training images with their text.

Run by 1_prepare_data.ipynb. Reads <data>/train.csv, <data>/test.csv and
<data>/images/, plus dev_ids.csv (the team's dev set), and writes into <data>:

    parts.csv        page_id, part, image, box, width, height
                                                   how each page is read: whole, or, for a
                                                   survey spread, left half then right half
    expected_chars.json                            how long a page (or half) of each collection
                                                   usually is, to catch readings that stopped early
    train_items.csv  page_id, doc_id, category, image, box, text
                                                   one row per training image (dev pages left out)
    dev_pages.csv    page_id, doc_id, category, label_source, text
                                                   the dev pages and their full labels, for scoring
    test_pages.csv   page_id, category             the pages Kaggle scores us on

A survey spread whose label splits at the right-hand page number becomes two
training images, one per page; one whose label doesn't split is trained on
whole. Either way, every survey spread is read in two halves.
"""

import json
import os
import sys

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import category_of, find_fold, is_spread, split_survey_label, spread_halves, training_text  # noqa: E402


def build(data_dir, dev_ids_csv, split_spreads=True):
    images_dir = os.path.join(data_dir, "images")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"), keep_default_na=False, encoding="utf-8-sig")
    test = pd.read_csv(os.path.join(data_dir, "test.csv"), keep_default_na=False, encoding="utf-8-sig")
    dev_ids = set(pd.read_csv(dev_ids_csv, keep_default_na=False)["page_id"])
    for df in (train, test):
        if "category" not in df.columns:
            df["category"] = df["page_id"].map(category_of)

    # How each page is read: whole, or a survey spread as two halves.
    parts, boxes, missing = [], {}, []
    for r in pd.concat([train, test])[["page_id", "category"]].drop_duplicates("page_id").to_dict("records"):
        image = os.path.join("images", r["page_id"] + ".jpg")
        path = os.path.join(data_dir, image)
        if not os.path.exists(path):
            missing.append(r["page_id"])
            continue
        with Image.open(path) as img:
            size = {"width": img.width, "height": img.height}
            if split_spreads and r["category"] == "survey_notes" and is_spread(*img.size):
                left, right = spread_halves(img.width, img.height, find_fold(img))
                boxes[r["page_id"]] = (left, right)
                parts += [{"page_id": r["page_id"], "part": "left", "image": image, "box": ",".join(map(str, left)), **size},
                          {"page_id": r["page_id"], "part": "right", "image": image, "box": ",".join(map(str, right)), **size}]
                continue
        parts.append({"page_id": r["page_id"], "part": "page", "image": image, "box": "", **size})
    pd.DataFrame(parts).to_csv(os.path.join(data_dir, "parts.csv"), index=False)

    # Training images: everything in train.csv except the dev pages.
    items, split_count, whole_spreads = [], 0, 0
    for r in train.to_dict("records"):
        if r["page_id"] in dev_ids or r["page_id"] in missing:
            continue
        image = os.path.join("images", r["page_id"] + ".jpg")
        text = training_text(r["text"], r["category"])
        halves = split_survey_label(text) if r["page_id"] in boxes else None
        if halves:
            split_count += 1
            for box, half in zip(boxes[r["page_id"]], halves):
                items.append({"page_id": r["page_id"], "doc_id": r["doc_id"], "category": r["category"],
                              "image": image, "box": ",".join(map(str, box)), "text": half})
        else:
            whole_spreads += r["page_id"] in boxes
            items.append({"page_id": r["page_id"], "doc_id": r["doc_id"], "category": r["category"],
                          "image": image, "box": "", "text": text})
    items = pd.DataFrame(items)
    items.to_csv(os.path.join(data_dir, "train_items.csv"), index=False)

    # How long each kind of image's text usually is (whole pages and half-spreads
    # separately), so a reading far shorter than that can be caught and redone.
    lengths = items.assign(kind=items.box.ne("").map({True: "half", False: "page"}),
                           chars=items.text.map(lambda t: len(" ".join(t.split()))))
    expected = {f"{c}/{k}": float(g.chars.median()) for (c, k), g in lengths.groupby(["category", "kind"])}
    with open(os.path.join(data_dir, "expected_chars.json"), "w") as f:
        json.dump(expected, f, indent=2)

    if "label_source" not in train.columns:
        train["label_source"] = ""
    dev = train[train["page_id"].isin(dev_ids)][["page_id", "doc_id", "category", "label_source", "text"]]
    dev.to_csv(os.path.join(data_dir, "dev_pages.csv"), index=False)
    test[["page_id", "category"]].to_csv(os.path.join(data_dir, "test_pages.csv"), index=False)

    # What we have, per collection.
    print(f"{'collection':<17}{'train pages':>12}{'train images':>14}{'dev pages':>11}{'test pages':>12}")
    for cat in ("kade_letters", "survey_notes", "dominy_accounts"):
        n_train = int(((train.category == cat) & ~train.page_id.isin(dev_ids)).sum())
        n_items = int((items.category == cat).sum()) if len(items) else 0
        print(f"{cat:<17}{n_train:>12}{n_items:>14}{int((dev.category == cat).sum()):>11}"
              f"{int((test.category == cat).sum()):>12}")
    print(f"\nSurvey spreads cut at the fold: {len(boxes)}. Training labels split into left and right pages: "
          f"{split_count}; trained on whole: {whole_spreads}.")
    absent = sorted({"kade_letters", "survey_notes", "dominy_accounts"} - set(train.category))
    if absent:
        print(f"WARNING: train.csv has no {', '.join(absent)} pages. This looks like the Kade-only class copy; "
              "use the full competition data (README.md, 'Where the data comes from').")
    if missing:
        print(f"WARNING: {len(missing)} pages have no image in {images_dir}, e.g. {', '.join(missing[:3])}.")
    return items
