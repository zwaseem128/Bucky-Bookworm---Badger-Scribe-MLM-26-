"""Turn a finished Kade LoRA run into numbers we can stand behind.

Used by both notebooks after a run. `run_dir` is the folder holding the run's
train_summary.json and eval_epoch<N>.csv files (the job's model.tar.gz,
unpacked, or the in-notebook run's model/ folder).

    report(run_dir)                 prints the before/after table and returns the numbers
    results_row(run_dir, name)      one Markdown row for docs/kade-lora-results.md
    merge_into_submission(...)      put the Kade predictions into a full Kaggle submission
"""

import json
import os

import numpy as np
import pandas as pd


def _load(run_dir):
    summary = json.load(open(os.path.join(run_dir, "train_summary.json")))
    before = pd.read_csv(os.path.join(run_dir, "eval_epoch0.csv"), keep_default_na=False)
    after = pd.read_csv(os.path.join(run_dir, f"eval_epoch{summary['kept_epoch']}.csv"), keep_default_na=False)
    pages = before.merge(after, on=["page_id", "doc_id"], suffixes=("_before", "_after"))
    return summary, pages


def paired_bootstrap(before, after, n=5000, seed=0):
    """How sure are we that the CER really dropped? Resample the pages with
    replacement many times, and see how much the average drop moves. Returns
    the mean drop and a 95% interval for it. If the interval doesn't include
    0, the improvement is very unlikely to be luck in which pages were scored."""
    drop = np.asarray(before, dtype=float) - np.asarray(after, dtype=float)
    rng = np.random.default_rng(seed)
    means = drop[rng.integers(0, len(drop), size=(n, len(drop)))].mean(axis=1)
    return float(drop.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report(run_dir):
    summary, pages = _load(run_dir)
    k = summary["kept_epoch"]
    print(f"Kept epoch {k} of {summary['epochs']:g}; trained on {', '.join(summary['train_docs'])} "
          f"({summary['train_pages']} pages, {summary['train_lines']} lines).\n")

    rows = []
    for doc, g in list(pages.groupby("doc_id")) + [("all scored pages", pages)]:
        rows.append({"document": doc, "pages": len(g),
                     "CER before": g.cer_before.mean(), f"CER epoch {k}": g.cer_after.mean(),
                     "reading CER before": g.cer_reading_before.mean(),
                     f"reading CER epoch {k}": g.cer_reading_after.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    drop, lo, hi = paired_bootstrap(pages.cer_before, pages.cer_after)
    better = int((pages.cer_after < pages.cer_before - 0.005).sum())
    worse = int((pages.cer_after > pages.cer_before + 0.005).sum())
    print(f"\nAverage CER drop: {drop:.3f} (95% interval {lo:.3f} to {hi:.3f}).")
    print(f"Pages better: {better}, worse: {worse}, about the same: {len(pages) - better - worse} "
          f"(of {len(pages)}; 'about the same' = within 0.005).")
    if lo > 0:
        print("The whole interval is above 0: the improvement holds up.")
    elif hi < 0:
        print("The whole interval is below 0: training made these pages worse.")
    else:
        print("The interval includes 0: on these pages we can't tell this run from no training.")

    pages["gain"] = pages.cer_before - pages.cer_after
    top = pages.sort_values("gain").iloc[-1]
    print(f"\nThe page that improved most: {top.page_id} ({top.cer_before:.3f} -> {top.cer_after:.3f})")
    print("REFERENCE\n" + top.text_ref_before[:500])
    print("\nBEFORE\n" + top.text_pred_before[:500])
    print(f"\nEPOCH {k}\n" + top.text_pred_after[:500])
    return {"summary": summary, "pages": pages, "drop": drop, "ci": (lo, hi), "better": better, "worse": worse}


def results_row(run_dir, run_name, cost=None):
    """One row for the results table in docs/kade-lora-results.md. The last
    cell is left empty for the leaderboard score, filled in by hand."""
    summary, pages = _load(run_dir)
    drop, lo, hi = paired_bootstrap(pages.cer_before, pages.cer_after)
    by_doc = pages.groupby("doc_id").cer_after.mean()
    docs = ", ".join(f"{d} {c:.3f}" for d, c in by_doc.items())
    settings = (f"lr {summary['lr']:g}, r {summary['lora_r']}, epochs {summary['epochs']:g}, "
                f"min_px {summary['min_pixels']}, seed {summary['seed']}, start {summary['start_adapter']}")
    cost_s = f"${cost:.2f}" if cost is not None else ""
    return (f"| {run_name} | {settings} | {summary['dev_cer_before']:.3f} | {pages.cer_after.mean():.3f} "
            f"(epoch {summary['kept_epoch']}) | {pages.cer_reading_after.mean():.3f} | {docs} | "
            f"{drop:.3f} [{lo:.3f}, {hi:.3f}] | {summary['train_minutes']} min {cost_s} | |")


def merge_into_submission(base_csv, kade_csv, out_csv):
    """Replace the Kade rows of a full Kaggle submission (for example the
    CHURRO run, which covers all three categories) with our predictions.
    Pages we have no prediction for keep the base file's text."""
    base = pd.read_csv(base_csv, keep_default_na=False, encoding="utf-8-sig")
    kade = pd.read_csv(kade_csv, keep_default_na=False).set_index("page_id")["text"]
    hit = base.page_id.isin(kade.index)
    base.loc[hit, "text"] = base.loc[hit, "page_id"].map(kade)
    base.to_csv(out_csv, index=False)
    n_kade = int(base.page_id.str.startswith("kade_").sum())
    print(f"{int(hit.sum())} of {n_kade} Kade rows replaced -> {out_csv}")
    if hit.sum() < n_kade:
        print("  The rest keep the base file's text. Check that every Kade test page was segmented.")
