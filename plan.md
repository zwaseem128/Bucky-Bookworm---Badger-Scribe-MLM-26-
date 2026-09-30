# Badger Scribe — MVP plan

**What we're building first:** the simplest pipeline that reads a page image, sends it to an
open-weight model, writes the text out, and gets a score on Kaggle. Not the best score. A
**baseline** — the thing every later idea is measured against.

**Deadline:** the competition closes in ~3 months (early Dec). Steps 0–5 are due the week of
**Sept 24**.

Everything here follows [CONTRIBUTING.md](CONTRIBUTING.md): one branch and one PR per step,
one reviewer, `CLAIM:` a file before editing it.

## The challenge in one paragraph

Transcribe 19th-century handwritten pages from UW Libraries — German immigrant letters in
**Kurrent script**, craftsmen's account books, and surveyors' field notes — **verbatim**: original
spelling, casing and punctuation, no cleanup. Scored by **character error rate (CER)**, averaged
per category and then across categories, so we must be decent at *all three*; the German letters
alone are 141 of the 182 test pages at launch. The final pipeline must run on **one GPU ≤ 96 GB**
using **open-weight models only**. A **writeup** in our repo is part of the submission.

## Hard rules (everyone, every agent, no exceptions)

1. **Never send a test image to Claude, GPT, Gemini or any closed model** — not for
   transcription, not for "post-correction", not to "have a look". That disqualifies us.
   Coding agents included: do not open files under `data/images/` for test page_ids.
   Train images are fine.
2. **Nobody hand-transcribes a test page.** Contributing transcriptions to the organizers
   is encouraged, but only train pages.
3. **No crawlers on UW library catalogs.** Everything we need is in the Kaggle dataset.
4. Submission cap is **2 per day**. Test locally first; the leaderboard confirms.

## Words we'll use

| word | plain meaning |
|---|---|
| **VLM** | a model that looks at an image and answers in text. `churro-3b` on BadgerBrain is one. An *embedding* model is not — it turns images into numbers for search and cannot transcribe. |
| **CER** | character error rate: edits needed to turn our text into the truth, divided by the truth's length, capped at 1.0. Lower is better. Empty prediction = 1.0. |
| **macro CER** | our actual score: CER averaged within each category, then averaged across categories. |
| **category** | `kade_letters` (German), Dominy account books, survey notes — from `train.csv`/`test.csv`. |
| **dev set** | training pages we hold out and never change; every idea is scored on the same dev set. |
| **baseline** | best result so far. A new idea must beat it on the dev set to be kept. |
| **LB** | Kaggle leaderboard. A fixed subset of test pages; rescored as the organizers add verified labels. |
| **silver** | `label_source = silver_claude`: a machine draft nobody has checked. All 436 German train pages are human-verified; the English ones are mostly silver (≈0.06–0.08 CER, and they over-modernise spelling). |

## Where things go (per CONTRIBUTING.md)

| what | where |
|---|---|
| the scorer — **copied verbatim from Kaggle, never edited** | `src/shared/metric.py` |
| loading images + CSVs, making the dev set | `src/data/` |
| BadgerBrain client + one file per model | `src/services/badgerbrain.py`, `src/services/models/churro.py`, `…/qwen_vl.py` |
| the pipeline (image → model → text → submission file) | `src/features/transcribe/` |
| tests | `tests/`, mirroring `src/` |
| notes, results table, error analysis, notebooks | `docs/`, `docs/notebooks/` |
| the Kaggle download (308 MB) and our predictions/submissions | `data/`, `outputs/` — **both gitignored**. Two new top-level folders: approve in team chat at step 0, as CONTRIBUTING requires. |
| the writeup (required for submission) | `WRITEUP.md` at repo root, from Kaggle's `WRITEUP_TEMPLATE.md` |

## Steps — at a glance

| # | step | who | done by | you know it works when |
|---|---|---|---|---|
| 0 | Download data, read the conventions, approve folders | all | Sept 18 | everyone has `data/` locally; `transcription_conventions.md` read |
| 1 | Look at the data | LP | Sept 19 | `docs/data-notes.md` has counts per category |
| 2 | Wire up Kaggle's `metric.py` | LD | Sept 19 | `sample_submission.csv` scores 1.0, train labels score 0.0 |
| 3 | Make the dev set (split by document) | LD | Sept 19 | `src/data/dev_ids.csv` on `main`, all 3 categories present |
| 4 | Pipeline with a fake model | ZW | Sept 22 | Kaggle accepts our submission (scores ~1.0) |
| 5 | First real model: CHURRO | CO | Sept 24 | dev + LB macro CER in `docs/results.md` → **baseline** |
| 6 | Second model + metadata in the prompt | CO, ZW | Sept 26 | 4 rows in `docs/results.md` |
| 7 | Look at the worst pages, per category | all | Sept 29 | `docs/error-analysis.md` has counts |
| 8 | Improvements one at a time — fine-tuning first | all | Oct → | each beats the baseline on dev, then LB |
| 9 | Writeup + tagged submission | all | ongoing; final by close | `WRITEUP.md` + `git tag v1.0-submission` |

## Steps — details

### 0. Download data, read the conventions (everyone, Sept 18)
- Everyone joins the competition and downloads the dataset into `data/` (gitignored).
- Everyone reads `transcription_conventions.md` — it defines what "correct" means
  (line breaks don't matter; `#` in the truth matches anything; everything else verbatim).
- Approve `data/` and `outputs/` as gitignored top-level folders in team chat.
- Open the Kaggle **starter notebooks**. If one already runs a VLM zero-shot end-to-end,
  step 5 starts from it instead of from scratch.
- Everyone has a BadgerBrain key and has run the 5-line example in the
  [gateway quickstart](https://github.com/qualiaMachine/RunAI_apps/blob/main/docs/gateway-quickstart.md).
  Gateway: `https://llm-gw01.doit.wisc.edu/v1`, OpenAI-style API. What it hosts (Sept 2026):

  | model | can it transcribe a page? |
  |---|---|
  | `churro-3b` | **yes** — the only vision model. Our step-5 model. |
  | `qwen3.8-27b` | no — text only. Could post-correct text later (open-weight, so allowed). |
  | `qwen3-vl-embedding-8b` | **no** — returns vectors, not text. Not a transcription model. |

  BadgerBrain has one vision model, so **someone asks Chris at step 0** whether a second one
  (e.g. `Qwen/Qwen2.5-VL-7B-Instruct`, the model in Kaggle's example writeup) can be hosted.
  Otherwise step 6 runs it on Kaggle's free GPU. Fine-tuning (step 8) also isn't something
  a gateway key can do — it needs a Run:ai account, Kaggle's GPU, or AWS credits.

Facts we now know (from the Overview/Data tabs, Sept 17):
- Files: `train.csv` (page_id, doc_id, text, category, label_source), `test.csv` (same minus
  text), `sample_submission.csv` (page_id, text), `images/<page_id>.jpg`, `metadata.csv`
  (per doc_id: title, creator, date, summary, collection, holding, rights_url, item_url),
  `metric.py`, `RESOURCES.md`, `WRITEUP_TEMPLATE.md`, `transcription_conventions.md`.
- Test at launch: 182 pages, 141 German. Data grows during the fall; LB gets rescored.
- Reference points: best open model the organizers tried scores ≈0.29 CER on German,
  ≈0.20 on English. The fine-tuning starter goes 0.23 → 0.06 on a local holdout in ~1 GPU-hour.

### 1. Look at the data (LP, Sept 19)
Notebook `docs/notebooks/lp-data-look.ipynb`. For **each category**: page count, human vs
silver count, 5 sample pages next to their labels, image sizes. By eye: how many pages are
rotated, faint, tabular, or have margin notes. Look hard at the Kurrent letters — that's the
script none of us can read.

**Works when:** `docs/data-notes.md` has the per-category table and answers
"how many of 15 sampled pages are rotated?" with a number.

### 2. Wire up the scorer (LD, Sept 19)
Copy Kaggle's `metric.py` to `src/shared/metric.py` unchanged — **it is the exact code that
scores the leaderboard, so our local number and LB number can't disagree by implementation.**
Add `src/shared/score.py` that calls it on a predictions file and prints macro CER + per-category CER.
`tests/shared/test_score.py`: `sample_submission.csv` scores 1.0; `train.csv` text as
predictions scores 0.0.

**Works when:** both tests pass.

### 3. Make the dev set (LD, Sept 19)
`src/data/make_dev_set.py` → `src/data/dev_ids.csv`. Rules:
- **Split by `doc_id`, not by page** — pages of one letter must not straddle dev/train,
  same as Kaggle does. Exception: the survey notes are one document; split those by page.
- All three categories present, roughly 15–20 pages each, fixed random seed.
- English dev labels are silver — note that next to any English dev number.

**Works when:** the CSV is on `main` and `score.py` reads it by default. **Nobody regenerates it.**

### 4. Pipeline with a fake model (ZW, Sept 22)
Every model is one function: `def transcribe(image, metadata) -> str`. The fake returns `""`.
`src/features/transcribe/` does:
1. load dev or test pages + their `metadata.csv` row (`src/data/`)
2. call `transcribe` on each — cache each result on disk so re-runs are free
3. write `outputs/predictions/<model>-dev.csv` and score it
4. write `outputs/submissions/<model>.csv` with `page_id,text`, same row order as
   `sample_submission.csv`

**Works when:** Kaggle **accepts** the fake submission (it scores ~1.0). That's our first
submission and proves the plumbing. Nothing counts as progress before this.

### 5. First real model: CHURRO (CO, Sept 24)
CHURRO is a small open model built specifically for historical handwriting across many scripts,
on BadgerBrain as `churro-3b`. `src/services/models/churro.py` sends the page as a base64 image
with the quickstart's prompt, `"Transcribe this page."`, no metadata yet. Smoke-test on **one
German train page (human label) and one English page (silver label)** before running the dev set. Run dev → score → run test → submit. Start `docs/results.md`:

| date | model | prompt | preprocessing | dev macro CER | dev per-category | LB CER | notes |
|---|---|---|---|---|---|---|---|

**Works when:** the row has dev and LB scores and they're in the same ballpark. **Baseline.**
Also start `WRITEUP.md` from the template and log this as the first thing we tried.

### 6. Second model + metadata in the prompt (CO and ZW, Sept 26)
`src/services/models/qwen_vl.py` for `Qwen/Qwen2.5-VL-7B-Instruct` — on BadgerBrain if Chris
adds it, otherwise in a Kaggle GPU notebook (same `transcribe()` shape either way). Then one
variant per model that adds the catalog metadata to the prompt: language (from category), plus `title`,
`creator`, `date`, `summary` from `metadata.csv`. The organizers say names, dates and subject
nouns are exactly what models misread and that metadata is worth feeding in.

**Works when:** 4 rows in `docs/results.md`, same dev set. Best row is the new baseline.

### 7. Look at the worst pages, per category (everyone, Sept 29 meeting)
For each category, sort dev pages by CER, worst first; everyone reads the 5 worst. Tag each:
`rotated` · `faint` · `stain/shadow/fold` · `underline/strikethrough` · `layout` (columns,
tables, margins, wrong reading order) · `script` (Kurrent vs print vs cursive) ·
`modernised spelling` · `translated instead of transcribed` · `made-up text` · `skipped text` ·
`bad label`. Count.

**Works when:** `docs/error-analysis.md` has the counts per category. **The biggest tag in
the worst category is what we fix first.** If `rotated` is 0, we never build rotation detection.

### 8. Improvements, one at a time (from Oct)
One idea = one branch (`chloe/data-contrast-preproc`) = one PR = one row in `docs/results.md`.
Order, to be re-sorted after step 7:

1. **Fine-tune** on the human-labelled train pages (the starter recipe; ~1 GPU-hour on Kaggle's
   free GPU, a Run:ai account, or AWS credits — a BadgerBrain key alone can't do this). Train on `human` rows first — silver rows teach modernised
   spelling, which the metric punishes. Add the Alfred Escher Kurrent set from `RESOURCES.md`.
2. Cut the page into lines/regions before the model (VLMs skip lines on dense pages)
3. Contrast / binarization — only if `faint` was a big tag
4. Rotation — only if `rotated` was a big tag; ask the VLM first, build a classifier second
5. Few-shot: one solved train page of the same category in the prompt
6. Two models vote; disagreement also flags pages for human review (the organizers ask for this)

**Each works when:** beats the baseline on dev macro CER by more than the noise (run the
baseline twice once to measure noise), *then* one LB submission confirms it.

### 9. Writeup and tagged submission (ongoing, final before close)
`WRITEUP.md` (≤2,500 words) is updated every time `docs/results.md` gets a row — the
organizers grade the learning journey, not just the number. For the final: `git tag
v1.0-submission && git push origin v1.0-submission`, fill the submission card (code_url,
models, peak_vram, leaderboard_cer, external_data, hardware, eval_wall_clock), post it in the
"Writeup Index" discussion thread. Organizers re-run our repo at that tag on one ≤96 GB GPU.

**Works when:** the tag URL opens in a private browser window and the card is posted.

## The prompt is part of the pipeline

One standard prompt for every model, kept in `src/features/transcribe/prompt.py`, changed only
via a PR with a results row. It must say what `transcription_conventions.md` says: transcribe
**verbatim**, keep original spelling, casing and punctuation, keep reading order, one line per
line, **do not translate or modernise** (Kurrent → modern German is translation, not
transcription — prior-work threads about Sütterlin→Hochdeutsch are about a different task).

## What we dropped from the first draft, and why

- **Recognizing characters one at a time with a per-language character set.** That means
  cutting handwriting into letters, which is the hardest part of the problem. VLMs do it for
  us. Weeks of work for a worse result.
- **Fixing orientation and contrast first.** Might matter, might not. Steps 1 and 7 tell us.
- **Fixing orientation by hand.** Only if step 7 says it's the top problem *and* the VLM can't.
- **Writing our own scorer.** Kaggle ships `metric.py`. Use it.
- **Comparing models on Kaggle.** 2 submissions a day. Compare on dev; Kaggle confirms.

Owners are a proposal — reassign at the next meeting.
