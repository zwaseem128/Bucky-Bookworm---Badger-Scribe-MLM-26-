"""Scoring metric for BadgerScribe: Archival Document Transcription Challenge.

Metric = Character Error Rate (CER), macro-averaged across document
categories. Lower is better. This exact code scores the Kaggle leaderboard
against the hidden test set (the optional Usage column carries the
Public/Private leaderboard split); participants run it locally against the
released train set (run this file from the command line) to iterate
between submissions.

    page CER     = levenshtein(prediction, reference) / len(reference), capped at 1.0
    category CER = mean of page CERs within the category
    score        = mean of category CERs

Comparison is verbatim — casing, punctuation, and historical spelling all
count. The only normalization applied to BOTH sides before comparison:
  * literal two-character "\\n" escapes become newlines,
  * formatting tags (<u>, <del>, <ins>) are removed from both sides,
  * Unicode NFC normalization,
  * all whitespace runs collapse to a single space; leading/trailing stripped.
See transcription_conventions.md for the ground-truth conventions.

Programmatic usage: `score(solution, submission, "page_id")`, where
  solution   columns: page_id, text, category  (+ optional Usage)
  submission columns: page_id, text

Self-test: `python metric.py` with no arguments (also what Kaggle's
notebook harness runs, harmlessly)
"""

import re
import sys
import unicodedata

import pandas as pd


class ParticipantVisibleError(Exception):
    """Raised for submission problems; the message is shown to the participant."""


_TAG_RE = re.compile(r"</?(?:u|del|ins)>")


def normalize_text(text) -> str:
    """Apply the (only) normalizations scoring allows. Everything else is verbatim."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text)
    # Line breaks may be submitted as literal backslash-n escapes.
    text = text.replace("\\n", "\n")
    # Formatting markup is not scored: <u>/<del>/<ins> tags are removed from
    # BOTH prediction and reference — transcribe the characters, not the styling.
    text = _TAG_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def levenshtein(a: str, b: str) -> int:
    """Edit distance, two-row DP; O(len(a) * len(b)) time, O(min) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (ca != cb),  # substitution
            ))
        prev = curr
    return prev[-1]


ILLEGIBLE = "#"


def _levenshtein_ref(pred: str, ref: str) -> int:
    """Edit distance where ILLEGIBLE in the REFERENCE matches any character free.

    Same DP as levenshtein() but it cannot swap the arguments, because the
    wildcard is one-sided: a `#` in the ground truth marks a character the
    human transcriber could not read, and matches whatever the prediction put
    there at zero cost. A `#` in a *prediction* is an ordinary character and
    is scored normally, so emitting `#` everywhere gains nothing.

    Without this, `#` inverts the incentive it exists for: a pipeline that
    correctly reads the character is penalised for reading it, while one that
    guesses `#` scores perfectly having read nothing.
    """
    if not pred:
        return len(ref)
    if not ref:
        return len(pred)
    prev = list(range(len(ref) + 1))
    for i, cp in enumerate(pred, start=1):
        curr = [i]
        for j, cr in enumerate(ref, start=1):
            sub = 0 if (cr == ILLEGIBLE or cr == cp) else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub))
        prev = curr
    return prev[-1]


def page_cer(prediction: str, reference: str) -> float:
    """Per-page CER against the normalized reference, capped at 1.0.

    `#` in the reference marks an illegible character and is not scored — see
    _levenshtein_ref. It still counts toward reference length, so a page is
    not made artificially easy by being hard to read.

    An empty reference would divide by zero; reference pages are curated to be
    non-empty, but guard anyway: empty reference scores 0.0 for an empty
    prediction and 1.0 otherwise.
    """
    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    if ILLEGIBLE in ref:
        return min(1.0, _levenshtein_ref(pred, ref) / len(ref))
    return min(1.0, levenshtein(pred, ref) / len(ref))


def page_wer(prediction: str, reference: str) -> float:
    """Word Error Rate — diagnostic only, not part of the score."""
    pred = normalize_text(prediction).split()
    ref = normalize_text(reference).split()
    if not ref:
        return 0.0 if not pred else 1.0
    # Reuse levenshtein over word sequences via a sentinel join.
    dist = _seq_levenshtein(pred, ref)
    return min(1.0, dist / len(ref))


def _seq_levenshtein(a, b) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        curr = [i]
        for j, tb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ta != tb)))
        prev = curr
    return prev[-1]


def per_category_cer(solution: pd.DataFrame, submission: pd.DataFrame,
                     row_id_column_name: str = "page_id") -> "pd.Series":
    """Mean page CER per category; index = category. Used by score() and the CLI."""
    for col in (row_id_column_name, "text", "category"):
        if col not in solution.columns:
            raise ValueError(f"solution is missing required column '{col}'")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission is missing the '{row_id_column_name}' column.")
    if "text" not in submission.columns:
        raise ParticipantVisibleError("Submission is missing the 'text' column.")

    sub = submission.drop_duplicates(subset=row_id_column_name, keep="last")
    if len(sub) < len(submission):
        raise ParticipantVisibleError("Submission contains duplicate page_id rows.")
    merged = solution.merge(sub[[row_id_column_name, "text"]],
                            on=row_id_column_name, how="left",
                            suffixes=("_ref", "_pred"))
    missing = merged["text_pred"].isna() & merged["text_ref"].notna()
    # Missing rows are allowed but score CER 1.0 (same as an empty prediction).
    merged.loc[missing, "text_pred"] = ""

    merged["cer"] = [
        page_cer(p, r) for p, r in zip(merged["text_pred"], merged["text_ref"])
    ]
    return merged.groupby("category")["cer"].mean()


def score(solution: pd.DataFrame, submission: pd.DataFrame,
          row_id_column_name: str) -> float:
    """Kaggle entry point. Returns macro-averaged CER (lower is better).

    row_id_column_name deliberately has NO default: Kaggle inspects this
    signature and rejects the metric with "row_id_column_name must not have a
    default value" if it does. Pass "page_id" when calling it yourself.
    """
    if "Usage" in solution.columns:
        solution = solution.drop(columns=["Usage"])
    return float(per_category_cer(solution, submission, row_id_column_name).mean())


def _cli(argv=None) -> int:
    """Score a predictions CSV against a labelled CSV, and print the breakdown.

    Lives here rather than in a separate script so the number you get locally
    and the number on the leaderboard come from one file, not two
    implementations that can drift apart.

        python metric.py --solution train.csv --submission my_predictions.csv

    With no arguments it runs the self-tests below instead.
    """
    import argparse

    ap = argparse.ArgumentParser(prog="metric.py", description=_cli.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solution", required=True, help="labelled CSV (page_id, text, category)")
    ap.add_argument("--submission", required=True, help="predictions CSV (page_id, text)")
    args = ap.parse_args(argv)

    # keep_default_na=False matters: without it pandas reads a transcribed
    # "NA" or "null" as a missing value and you score against NaN.
    solution = pd.read_csv(args.solution, keep_default_na=False)
    submission = pd.read_csv(args.submission, keep_default_na=False)

    try:
        by_cat = per_category_cer(solution, submission)
        overall = score(solution, submission, "page_id")
    except ParticipantVisibleError as e:
        print(f"submission error: {e}", file=sys.stderr)
        return 1

    merged = solution.merge(submission.drop_duplicates("page_id", keep="last"),
                            on="page_id", how="left", suffixes=("_ref", "_pred"))
    merged["text_pred"] = merged["text_pred"].fillna("")
    wer_by_cat = merged.assign(
        wer=[page_wer(p, r) for p, r in zip(merged["text_pred"], merged["text_ref"])]
    ).groupby("category")["wer"].mean()

    counts = solution.groupby("category")["page_id"].count()
    print(f"{'category':<24}{'pages':>6}{'CER':>9}{'WER':>9}")
    for cat in sorted(by_cat.index):
        print(f"{cat:<24}{counts[cat]:>6}{by_cat[cat]:>9.4f}{wer_by_cat[cat]:>9.4f}")
    print("-" * 48)
    print(f"{'macro CER (overall)':<30}{overall:>9.4f}")
    return 0


# Kaggle runs this file as a NOTEBOOK cell, where __name__ == "__main__" and
# sys.argv is populated by papermill. Dispatching the CLI on `len(sys.argv) > 1`
# therefore fired on the leaderboard, argparse found no --solution, and the
# metric validation run died with SystemExit 2. The guard has to identify OUR
# invocation specifically: argv[0] is this file, and our flag is present.
def _invoked_as_cli() -> bool:
    return (bool(sys.argv) and sys.argv[0].endswith("metric.py")
            and "--solution" in sys.argv)


if __name__ == "__main__":
    if _invoked_as_cli():
        raise SystemExit(_cli())

    # Anything else - including Kaggle's notebook harness - runs the self-test.
    sol = pd.DataFrame({
        "page_id": ["a1", "a2", "b1", "b2"],
        "text": ["Meandered the Lake", "N 40 E 12.50", "Lieber Bruder,\nes geht", "To 3 chairs 0.7.6"],
        "category": ["survey_notes", "survey_notes", "kade_letters", "dominy_accounts"],
    })

    perfect = sol[["page_id", "text"]].copy()
    assert score(sol, perfect, "page_id") == 0.0

    # Whitespace layout is free: \n escapes and extra spaces score identically.
    ws = perfect.copy()
    ws.loc[2, "text"] = "Lieber   Bruder, \\n es geht"
    assert score(sol, ws, "page_id") == 0.0

    # Case matters (verbatim scoring).
    cased = perfect.copy()
    cased.loc[0, "text"] = "meandered the lake"
    assert score(sol, cased, "page_id") > 0.0

    # Garbage and empty predictions both cap at CER 1.0 for the page.
    garbage = perfect.copy()
    garbage.loc[1, "text"] = "x" * 500
    empty = perfect.copy()
    empty.loc[1, "text"] = ""
    assert score(sol, garbage, "page_id") == score(sol, empty, "page_id")

    # Macro average: one bad page in a 2-page category = 0.5 category CER / 3 categories.
    assert abs(score(sol, empty, "page_id") - (0.5 / 3)) < 1e-9

    # Missing row scores like an empty prediction, not an error.
    blank_b2 = perfect.assign(text=perfect["text"].where(perfect["page_id"] != "b2", ""))
    assert score(sol, perfect.iloc[:3], "page_id") == score(sol, blank_b2, "page_id")

    # Duplicate ids are rejected.
    try:
        score(sol, pd.concat([perfect, perfect.iloc[:1]]), "page_id")
        raise AssertionError("duplicate ids should raise")
    except ParticipantVisibleError:
        pass

    # Kaggle's notebook harness must NOT be mistaken for a CLI invocation.
    # This is the failure that took down the metric validation run: argv is
    # populated there, so a bare `len(sys.argv) > 1` fired argparse and the
    # run died with SystemExit 2 before scoring anything.
    _real_argv = sys.argv
    for fake in (["/usr/local/lib/python3.12/dist-packages/ipykernel_launcher.py",
                  "-f", "/root/.local/share/jupyter/runtime/kernel-17.json"],
                 ["__notebook__.ipynb"],
                 ["metric.py"]):
        sys.argv = fake
        assert not _invoked_as_cli(), f"would have run the CLI for argv={fake}"
    sys.argv = ["tools/metric.py", "--solution", "a.csv", "--submission", "b.csv"]
    assert _invoked_as_cli()
    sys.argv = _real_argv

    # Kaggle rejects the metric if row_id_column_name carries a default.
    import inspect
    _p = inspect.signature(score).parameters["row_id_column_name"]
    assert _p.default is inspect.Parameter.empty, \
        "row_id_column_name must not have a default value (Kaggle rejects it)"

    assert levenshtein("kitten", "sitting") == 3
    assert page_wer("the cat sat", "the cat sat") == 0.0
    assert abs(page_wer("the dog sat", "the cat sat") - 1 / 3) < 1e-9

    print("all metric self-tests passed")
