# Badger Scribe — MVP plan

**Goal of the MVP:** one model, one prompt, one local scorer, one valid Kaggle submission, all
runnable by any of us in under an hour. Not the best score — the *baseline* every later
component is A/B'd against.

**Verdict on the original draft:** the kernel (run existing models zero-shot → submit → compare)
is right. Everything else in the draft is either a later A/B (orientation, contrast) or not an
MVP at all (per-character statistical classifier). See "What we cut" at the bottom.

MLM26 runs Sept 9 – Dec 9. Steps 0–5 are week one. If we're not on step 5 by **Sept 24**, we
are over-building.

## Definitions

| term | meaning |
|---|---|
| **input** | one page image + its metadata row (at minimum: id, language) |
| **output** | one UTF-8 transcription string per page id, in the sample-submission format |
| **dev split** | 50 labelled train pages, fixed and committed (`configs/dev_ids.csv`), stratified by language |
| **score** | our local implementation of the Kaggle metric, run on the dev split. **Never** the leaderboard — we get a handful of submissions a day; the leaderboard confirms, it doesn't iterate |
| **baseline** | the best row in `docs/experiments.md` after step 6. Every PR after that reports its delta against it |

## Steps

### 0. Read the rules and the metric — day 1, everyone, 30 min
Open the competition Overview + Data tabs together. Write into this file, verbatim:
- the metric (CER? WER? normalized how? case-folded? whitespace-collapsed?)
- submissions per day, and whether the public LB is a subset of test
- whether pretrained/external models and internet are allowed (they must be — the whole plan is hosted VLMs)
- train / test counts, image format, every metadata column

**Works when:** the four bullets above are filled in and nobody disagrees with them.

### 1. Look at the data — [LP], by Sept 19
`notebooks/lp-data-look.ipynb`. Twenty random train pages side-by-side with their labels. Count
languages, image sizes, and — by eye — how many pages are rotated, faint, multi-column, or have
marginalia. Note whether labels are page-level or line-level and whether line breaks are preserved.

**Works when:** a ten-line summary lands in `docs/data-notes.md` with counts, and it answers
"do we even have an orientation problem?" with a number.

### 2. Local scorer — [LD], by Sept 19
`badger_scribe/eval/metric.py` implementing exactly the metric from step 0.
`tests/eval/test_metric.py` with: perfect prediction → 0 error; empty prediction → 1.0 (or
whatever the metric gives); one known edit → the hand-computed value.

**Works when:** tests pass, and after step 5 the dev score and the public LB score for the same
model are in the same ballpark. If they're wildly different our normalization is wrong — fix
before anything else.

### 3. Dev split — [LD], with step 2
`scripts/make_dev_split.py --seed 0 --n 50` → `configs/dev_ids.csv`. Stratified by language.
Committed. Nobody re-samples it.

**Works when:** the file is on `main` and `score.py` refuses to run on anything else by default.

### 4. Pipeline skeleton with a dummy model — [ZW], by Sept 22
```
scripts/transcribe.py --model dummy --split dev   →  artifacts/preds/dummy-dev.csv
scripts/score.py artifacts/preds/dummy-dev.csv    →  prints score
scripts/transcribe.py --model dummy --split test  →  artifacts/submissions/dummy.csv
```
Model adapters live in `badger_scribe/models/` and implement one function:
`transcribe(image: PIL.Image, meta: dict) -> str`. The dummy returns `""`.
Cache every model call to disk keyed by (model, prompt, image id) so re-runs are free.

**Works when:** `dummy.csv` is **accepted by Kaggle** and appears on the leaderboard with a
terrible score. That's submission #1 and the pipeline is real. Until this happens, nothing
else counts as progress.

### 5. First real model: CHURRO — [CO], by Sept 24
CHURRO is a ~3B open-weight VLM fine-tuned specifically for historical text recognition
(page-level, many languages, centuries of scripts; claims to beat frontier APIs at far lower
cost). It is on BadgerBrain, it's cheap, and it's the model most likely to just work. Adapter
`badger_scribe/models/churro.py`, prompt = whatever the CHURRO paper uses, no language hint yet.

Run dev → score → run test → submit. Add the first row to `docs/experiments.md`:

| date | model | prompt | preprocessing | dev score | LB score | notes |
|---|---|---|---|---|---|---|

**Works when:** dev score and LB score are both recorded, and the dev/LB gap is small enough
that we trust the dev score (step 2's check). **This row is the baseline.**

### 6. Second model + language hint — [CO]/[ZW], by Sept 26
Same adapter interface, `badger_scribe/models/qwen_vl.py` against BadgerBrain's Qwen VL model.
Then one prompt variant for each model: add `"The text is in {language}."` from metadata.
That's four rows in the experiments table.

**Works when:** four rows exist, run on the same dev split with the same scorer. The best one
becomes the baseline. Whatever the answer, we now know whether the language metadata is worth
anything — the original draft assumed it; this measures it.

### 7. Error analysis on the baseline — everyone, Sept 29 meeting
Sort dev pages by per-page error. Everyone reads the worst ten. Tag each page: `orientation`,
`faint/contrast`, `layout` (columns, margins, tables), `language`, `hallucination`,
`omission`, `label-noise`. Count.

**Works when:** `docs/error-analysis.md` has the counts. The biggest bucket is the first A/B.
If `orientation` is 0/10, we never build orientation detection.

### 8. A/B components — from Oct, one branch + one PR + one experiments row each
Candidates, ordered by likely payoff per hour (re-order after step 7):
1. Line/region segmentation before the VLM (VLMs skip and merge lines on dense pages)
2. Contrast / binarization preprocessing (only if `faint` was a big bucket)
3. Orientation detection (only if `orientation` was a big bucket; try the VLM first, then a 4-way rotation classifier)
4. Few-shot: one solved page of the same language in the prompt
5. Ensembling / majority vote across models
6. Fine-tuning CHURRO on our train set — when AWS credits arrive, not before

**Each works when:** dev score beats the baseline by more than the run-to-run noise (measure
noise once by running the baseline twice), *then* one LB submission confirms it. Only then
does it become the new baseline.

## What we cut from the draft, and why

- **Per-character statistical classification with a language-specific charset.** Requires
  character segmentation of 19th-century handwriting, which is the hard part of HTR; VLMs do
  it implicitly and better. Weeks of work for a worse baseline.
- **Orientation and contrast first.** They're preprocessing A/Bs, not the MVP. Step 7 decides
  whether they're worth anything; step 1 gives us a first guess for free.
- **Manual orientation fixing.** Only if step 7 says orientation is the biggest bucket *and*
  the VLM can't fix it. Even then, a rotation classifier is a one-day build.
- **"Compare how each scores in Kaggle."** Compare on the dev split with our scorer; use Kaggle
  to confirm. Otherwise we burn the daily submission cap on things we could have measured locally.

## Owners (proposal — reassign at the next meeting)

| | steps |
|---|---|
| LP | 1, 7 lead |
| CO | 5, 6 |
| LD | 2, 3, scorer/LB gap check |
| ZW | 4, submission plumbing, 6 |

Everyone: step 0, step 7.
