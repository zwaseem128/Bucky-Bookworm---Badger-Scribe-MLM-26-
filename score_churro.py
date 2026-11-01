"""
Score the CHURRO transcriptions with the official metric.py, without changing it.

metric.py is imported straight from the Badger Scribe shared drive, so the
numbers here come from the same code as the Kaggle leaderboard.

    python score_churro.py        score the pages from the last eval run
    python score_churro.py all    score every transcription in "output data"

By default only the pages listed in "output data"/last_eval_pages.csv are
scored. churro_simple.py writes that list at the end of an eval run, so
older transcriptions in the same folder (from an "all" run, or from an
earlier prompt) don't get mixed into the score.

Pages are scored against their label in train.csv, whether a person wrote it
(human) or it is an unchecked machine draft (silver_claude); page_scores.csv
has a label_source column to tell them apart. Pages that aren't in
train.csv, such as test pages, can't be scored and are listed as "no label".

Results go to "output data"/scores/:
  page_scores.csv  one row per page: CER, WER, and both texts side by side
  summary.txt      the per-category table and the overall score
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Same drive location as churro_simple.py, including the BADGER_DRIVE
# override for teammates whose Google Drive isn't mounted as G:.
DRIVE = Path(os.environ.get("BADGER_DRIVE", r"G:\Shared drives\Badger Scribe"))
if not DRIVE.exists():
    raise SystemExit(f"Can't find {DRIVE}. Is Google Drive for Desktop running "
                     "and signed in to the account that has the shared drive?")

# Don't leave a __pycache__ folder on the shared drive when importing metric.py.
sys.dont_write_bytecode = True
sys.path.insert(0, str(DRIVE))
import metric  # noqa: E402  (the shared drive copy, unmodified)

LABELS = DRIVE / "train.csv"
OUT = DRIVE / "output data"
EVAL_LIST = OUT / "last_eval_pages.csv"
SCORES = OUT / "scores"

score_all = len(sys.argv) > 1 and sys.argv[1].lower() == "all"

txts = sorted(OUT.glob("*.txt"))
if not score_all:
    if not EVAL_LIST.exists():
        raise SystemExit(f"There's no {EVAL_LIST.name} in {OUT} yet. Run churro_simple.py "
                         "and type 'eval' first, or run 'python score_churro.py all' "
                         "to score every transcription in the folder.")
    wanted = set(pd.read_csv(EVAL_LIST, keep_default_na=False, dtype=str)["page_id"])
    txts = [p for p in txts if p.stem in wanted]
if not txts:
    raise SystemExit(f"No transcriptions to score in {OUT}.")

# keep_default_na=False for the same reason metric.py uses it: a page that
# reads "NA" must stay text, not turn into a missing value.
labels = pd.read_csv(LABELS, keep_default_na=False)
preds = pd.DataFrame(
    [{"page_id": p.stem, "text": p.read_text(encoding="utf-8")} for p in txts]
)

# Score against only the pages we've transcribed. Given the whole train.csv,
# metric.py would count every untouched page as a blank answer (CER 1.0).
labeled = labels[labels["page_id"].isin(preds["page_id"])]
solution = labeled[["page_id", "text", "category"]]
unlabeled = sorted(set(preds["page_id"]) - set(labels["page_id"]))
if solution.empty:
    raise SystemExit("None of these pages are in train.csv, so there's nothing to score.")

pages = solution.merge(preds, on="page_id", suffixes=("_reference", "_churro"))
pages.insert(2, "cer", [metric.page_cer(p, r) for p, r in zip(pages["text_churro"], pages["text_reference"])])
pages.insert(3, "wer", [metric.page_wer(p, r) for p, r in zip(pages["text_churro"], pages["text_reference"])])
pages.insert(4, "label_source", pages["page_id"].map(labeled.set_index("page_id")["label_source"]))
SCORES.mkdir(parents=True, exist_ok=True)
pages.to_csv(SCORES / "page_scores.csv", index=False, encoding="utf-8")

by_cat = metric.per_category_cer(solution, preds, "page_id")
overall = metric.score(solution, preds, "page_id")

lines = [f"{'category':<24}{'pages':>6}{'CER':>9}"]
counts = solution.groupby("category")["page_id"].count()
for cat in sorted(by_cat.index):
    lines.append(f"{cat:<24}{counts[cat]:>6}{by_cat[cat]:>9.4f}")
lines.append("-" * 39)
lines.append(f"{'macro CER (overall)':<30}{overall:>9.4f}")
lines.append("")
lines.append("CER = share of characters wrong (0 is perfect, 1 is all wrong).")
lines.append(f"Scored {len(solution)} of {len(preds)} transcribed pages "
             f"({'every transcription in the folder' if score_all else 'from the last eval run'}).")
silver = int((pages["label_source"] != "human").sum())
if silver:
    lines.append(f"{silver} of them are scored against unchecked machine drafts (silver_claude).")
if unlabeled:
    lines.append(f"No label in train.csv, so not scored: {', '.join(unlabeled)}")
summary = "\n".join(lines)
(SCORES / "summary.txt").write_text(summary + "\n", encoding="utf-8")

print(summary)
print(f"\nSaved to {SCORES}")
