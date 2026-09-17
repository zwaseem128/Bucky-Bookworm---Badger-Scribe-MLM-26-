# Badger Scribe — MVP plan

**What we're building first:** the simplest pipeline that reads a manuscript image, sends it to a
model, writes the text out, and gets a score on Kaggle. Not the best score. A **baseline** —
the thing every later idea is measured against.

**Deadline:** MLM26 ends Dec 9. Steps 0–5 below are due the week of **Sept 24**.

Everything here follows [CONTRIBUTING.md](CONTRIBUTING.md): one branch and one PR per step,
one reviewer, `CLAIM:` a file before editing it.

## Words we'll use

| word | plain meaning |
|---|---|
| **VLM** | a model that looks at an image and answers in text. CHURRO and Qwen-VL on BadgerBrain are VLMs. |
| **CER / metric** | the number Kaggle uses to score us. Probably "character error rate" — fraction of characters we got wrong. **Step 0 confirms this.** Lower is better (if it's CER). |
| **dev set** | 50 training pages we set aside and never change. We score every idea on these same 50 pages so results are comparable. |
| **baseline** | the best result we have so far. Any new idea has to beat it on the dev set to be kept. |
| **LB** | the Kaggle leaderboard. We only get a few submissions per day, so we test locally first and use the LB to confirm. |

## Where things go (per CONTRIBUTING.md)

| what | where |
|---|---|
| the scoring function | `src/shared/metric.py` |
| loading images + metadata, making the dev set | `src/data/` |
| talking to BadgerBrain, one file per model | `src/services/badgerbrain.py`, `src/services/models/churro.py`, `…/qwen_vl.py` |
| the pipeline (image → model → text → submission file) | `src/features/transcribe/` |
| tests | `tests/`, next to what they test |
| notes, results table, error analysis, notebooks | `docs/`, `docs/notebooks/` |
| predictions and submission files | `outputs/` — **gitignored**. This is one new top-level folder; approve it in team chat at step 0 as CONTRIBUTING requires. |

## Steps — at a glance

| # | step | who | done by | you know it works when |
|---|---|---|---|---|
| 0 | Read the Kaggle rules and metric together | all | Sept 18 | the facts are written into this file |
| 1 | Look at the data | LP | Sept 19 | `docs/data-notes.md` has counts |
| 2 | Write the scorer | LD | Sept 19 | tests pass |
| 3 | Make the dev set | LD | Sept 19 | `src/data/dev_ids.csv` is on `main` |
| 4 | Pipeline with a fake model | ZW | Sept 22 | Kaggle accepts the submission |
| 5 | First real model: CHURRO | CO | Sept 24 | dev score + LB score in the results table → **baseline** |
| 6 | Second model + language hint | CO, ZW | Sept 26 | 4 rows in the results table |
| 7 | Look at the worst pages | all | Sept 29 | `docs/error-analysis.md` has counts |
| 8 | Try improvements one at a time | all | Oct → | each beats the baseline on dev, then on LB |

## Steps — details

### 0. Read the rules and the metric (everyone, 30 min, Sept 18)
Open the Kaggle Overview and Data tabs together and write the answers here, word for word:
- Metric: ____ (CER? WER? is case ignored? is whitespace collapsed?)
- Submissions per day: ____
- Train pages: ____ Test pages: ____ Image format: ____
- Metadata columns: ____
- Pretrained models and internet allowed? ____ (must be yes — the plan depends on BadgerBrain)

Also approve the `outputs/` folder in team chat.

### 1. Look at the data (LP, Sept 19)
Notebook in `docs/notebooks/lp-data-look.ipynb`. Show 20 random training pages next to their
correct text. Count: languages, image sizes, and — just by looking — how many pages are rotated,
faint, in columns, or have writing in the margins. Note whether the correct text keeps line breaks.

**Works when:** `docs/data-notes.md` has those counts. In particular it answers "how many of 20
pages are rotated?" with a number, so we know whether orientation is even a problem.

### 2. Write the scorer (LD, Sept 19)
`src/shared/metric.py` — the exact metric from step 0, nothing fancier.
`tests/shared/test_metric.py` — three tests: perfect answer scores 0; empty answer scores 1
(or whatever the metric says); one deliberate typo scores the value you computed by hand.

**Works when:** the tests pass. Later (step 5) our dev score and the LB score should be close.
If they aren't, this file is wrong — fix it before doing anything else.

### 3. Make the dev set (LD, Sept 19)
`src/data/make_dev_set.py` picks 50 training pages, spread across languages, with a fixed random
seed, and writes `src/data/dev_ids.csv`. Commit the CSV. **Nobody regenerates it.**

**Works when:** the CSV is on `main` and the scorer reads it by default.

### 4. Pipeline with a fake model (ZW, Sept 22)
Build the plumbing before any real model. Every model is one Python function:

```python
def transcribe(image, metadata) -> str
```

The fake model returns `""`. The pipeline in `src/features/transcribe/` does:

1. load dev or test images + metadata (`src/data/`)
2. call `transcribe` on each — save each result to disk so re-runs are free
3. write `outputs/predictions/<model>-dev.csv` and score it
4. write `outputs/submissions/<model>.csv` in Kaggle's format

**Works when:** the fake model's submission is **accepted by Kaggle** and shows up on the LB
with a terrible score. That's our first submission. Nothing counts as progress before this.

### 5. First real model: CHURRO (CO, Sept 24)
CHURRO is a small open model built specifically to read old handwritten documents. It's on
BadgerBrain and cheap, so it's the one most likely to just work. Write
`src/services/models/churro.py`, use the prompt from its paper, no language hint yet.

Run dev → score → run test → submit. Start `docs/results.md`:

| date | model | prompt | preprocessing | dev score | LB score | notes |
|---|---|---|---|---|---|---|

**Works when:** the row has both scores and they're close. **This row is the baseline.**

### 6. Second model + language hint (CO and ZW, Sept 26)
`src/services/models/qwen_vl.py` for BadgerBrain's Qwen vision model, same function shape.
Then, for both models, one variant that adds `"The text is in {language}."` from the metadata.

**Works when:** `docs/results.md` has 4 rows on the same dev set. Best one is the new baseline.
Now we *know* whether the language metadata helps instead of assuming it.

### 7. Look at the worst pages (everyone, Sept 29 meeting)
Sort the dev pages by score, worst first. Everyone reads the ten worst. Tag each one:
`rotated` · `faint` · `layout` (columns, margins, tables) · `wrong language` · `made-up text` ·
`skipped text` · `bad label`. Count the tags.

**Works when:** `docs/error-analysis.md` has the counts. **The biggest tag is what we fix first.**
If `rotated` is 0 out of 10, we never build orientation detection.

### 8. Try improvements, one at a time (from Oct)
One idea = one branch (`chloe/data-contrast-preproc`) = one PR = one row in `docs/results.md`.
Rough order, to be re-sorted after step 7:

1. Cut the page into lines or regions before sending to the model (VLMs skip lines on dense pages)
2. Contrast / binarization — only if `faint` was a big tag
3. Detect rotation — only if `rotated` was a big tag; ask the VLM first, build a classifier second
4. Show the model one solved page in the same language (few-shot)
5. Run both models and vote
6. Fine-tune CHURRO on our training pages — when AWS credits arrive, not before

**Each works when:** it beats the baseline on the dev set by more than the noise (run the
baseline twice once to measure noise), *then* one LB submission confirms it.

## What we dropped from the first draft, and why

- **Recognizing characters one at a time with a per-language character set.** That means
  cutting handwriting into individual letters, which is the hardest part of the whole problem.
  VLMs do it for us. Weeks of work for a worse result.
- **Fixing orientation and contrast first.** Might matter, might not. Step 1 and step 7 tell us.
- **Fixing orientation by hand.** Only if step 7 says it's the top problem *and* the VLM can't
  handle it. Even then a rotation classifier is a one-day build.
- **Comparing models on Kaggle.** Compare on the dev set. Kaggle confirms. Otherwise we burn our
  daily submissions on things we could have measured in seconds.

Owners above are a proposal — reassign at the next meeting.
