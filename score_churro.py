"""
Score the CHURRO transcriptions with the official metric.py, without changing it.

metric.py is imported straight from the Badger Scribe shared drive, so the
numbers here come from the same code as the Kaggle leaderboard. Only pages
that have a human transcription in train.csv can be scored; the rest are
listed as "no label".

Results go to churro_output/scores/:
  page_scores.csv  one row per page: CER, WER, and both texts side by side
  summary.txt      the per-category table and the overall score
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"G:\Shared drives\Badger Scribe")
import metric  # noqa: E402  (the shared drive copy, unmodified)

LABELS = Path.home() / "Desktop" / "BuckyBookworm" / "data" / "train.csv"
OUT = Path.home() / "Downloads" / "Bucky Bookworm Images" / "churro_output"
SCORES = OUT / "scores"
SCORES.mkdir(exist_ok=True)

# keep_default_na=False for the same reason metric.py uses it: a page that
# reads "NA" must stay text, not turn into a missing value.
labels = pd.read_csv(LABELS, keep_default_na=False)
preds = pd.DataFrame(
    [{"page_id": p.stem, "text": p.read_text(encoding="utf-8")}
     for p in sorted(OUT.glob("*.txt"))]
)

# Score against only the pages we've transcribed. Given the whole train.csv,
# metric.py would count every untouched page as a blank answer (CER 1.0).
solution = labels[labels["page_id"].isin(preds["page_id"])][["page_id", "text", "category"]]
unlabeled = sorted(set(preds["page_id"]) - set(labels["page_id"]))

pages = solution.merge(preds, on="page_id", suffixes=("_reference", "_churro"))
pages.insert(2, "cer", [metric.page_cer(p, r) for p, r in zip(pages["text_churro"], pages["text_reference"])])
pages.insert(3, "wer", [metric.page_wer(p, r) for p, r in zip(pages["text_churro"], pages["text_reference"])])
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
lines.append(f"Scored {len(solution)} of {len(preds)} transcribed pages.")
if unlabeled:
    lines.append(f"No label in train.csv, so not scored: {', '.join(unlabeled)}")
summary = "\n".join(lines)
(SCORES / "summary.txt").write_text(summary + "\n", encoding="utf-8")

print(summary)
print(f"\nSaved to {SCORES}")
