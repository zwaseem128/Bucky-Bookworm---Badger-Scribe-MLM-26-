# Fine-tuning CHURRO on all three collections

This folder fine-tunes **CHURRO-3B**, an open handwriting model trained on 100,000 historical
pages, on whole pages from all three Badger Scribe collections at once: the Kade letters, the
survey field notes and the Dominy account books. It runs entirely in a SageMaker notebook in your
browser, so it works the same on a Mac or Windows.

It starts from the recipe in the SageMaker lesson (which took the Kade letters from a CER of
0.267 to 0.065 for $0.75 of GPU) and **tailors it to every problem in Lokesh's dev-set audit**
(`docs/dev-set-error-audit.md` on `lokesh/data-dev-set`). Each fix below is also a setting, so
you can turn it off and measure what it's worth.

## How each problem from the dev-set audit is tackled

The numbers are the audit's own problem numbers.

| Audit problem | What this folder does about it | Where |
|---|---|---|
| **1. Survey: the two-page spreads.** CHURRO stopped early, read a page twice, or wove the two pages together on 12 of 35 dev pages, and that was 77% of the survey error. | Each spread is cut at the fold and read one page at a time, left then right. Labels are split to match where they can (146 of 288 split at the right-hand page number). A reading that comes out empty, far shorter than that kind of page usually is, or stuck repeating itself, is read again in two parts, top and bottom, and the better reading is kept. | Notebook 1; `retry` |
| **2. Kade: small, faint pencil postcards.** The dev postcards are 693 × 1151 pixels, half the detail of most Kade scans. | Small images are enlarged to the same pixel budget as the others, so the writing gets as much detail. In training, Kade images are sometimes shrunk and enlarged again and faded, so the model also learns from pages that look like the postcards. | `upscale`, `augment` |
| **3. Kade: a `Postmark` line that isn't on the page** (10 of 11 dev labels end with one). | It's taken out of what the model learns, so it never invents dates. Then, measured on the training pages (not the dev letter): 15 of 22 small Kade images (under 1 megapixel: postcards) have a postmark line (68%), but only 54 of 262 larger ones (21%). So on small cards only, `Postmark , 19` is added: the part every postmark line shares. It saves about 13 characters of error when the label has one and costs 14 when it doesn't, so at 68% it gains about 4 characters per card on average. | `postmark_guess` |
| **4. Kade: dialect spelling** (*jestern*, *bei die factry*). | Nothing corrects spelling, anywhere; the labels are learned exactly as written. | always |
| **5. Dominy: stains, bleed-through, and `#` in the labels.** | `#` (what the transcriber couldn't read) is never something the model learns to write. Training images sometimes get a faint mirror image of the page on top, like ink from the back showing through. An optional contrast stretch for faded pages. | always; `augment`; `enhance` |
| **6. Dominy: ledger symbols the labels leave out.** | No training label contains a check mark or `×`, so they're removed from the output; curly quotes and long dashes become plain ones; fractions are written ½ ¼ ¾ as the labels do. | always |
| **7. Few pages, short pages.** Each collection is a third of the score, but Dominy has only 27 training pages. | Dominy's training images are used 4 times per epoch, each time with different random changes. | `oversample` |
| **8. Survey: vocabulary and numbers.** | Learned from the 365 survey training images (pages and half-spreads). | training |
| **9. Machine-drafted labels.** 42 of 44 English dev labels are drafts, about 6–8% wrong themselves. | The scores on the expert-typed dev pages are shown separately. | scoring |

Two more things help with all of them:

- **Honest scores.** Every run is scored on the team's 55 dev pages (`dev_ids.csv`): whole
  documents set aside, never trained on. Scoring uses the competition's own `metric.py`, per
  collection, plus the audit's length check: how many pages came out too short (under 0.7 of the
  label's length) or too long (over 1.1). The lesson held out random pages from letters it also
  trained on, which flatters the score.
- **Separating the effects.** Notebook 2 has a `"plain"` run (CHURRO with none of the reading
  fixes) and a `"baseline"` run (with them, but no training), so you can see what the fixes are
  worth on their own and what training adds on top.

## What's in the folder

| File | What it does |
|---|---|
| `1_prepare_data.ipynb` | Downloads the pages from S3, sets the dev pages aside, cuts the survey spreads. Run once. |
| `2_finetune.ipynb` | Trains the adapter and scores it on the dev pages, per collection. |
| `3_predict_test.ipynb` | Reads the 208 test pages with your best run and writes `submission.csv` for Kaggle. |
| `dev_ids.csv` | The team's dev pages (from `lokesh/data-dev-set`). |
| `scripts/` | The code the notebooks run. |

## How long it takes

On an **`ml.g5.xlarge`** (one A10G GPU, about $1.41 an hour), the recommended instance:

| Step | Time | What takes the time |
|---|---|---|
| Notebook 1 | 15–20 min | Downloading about 2.5 GB of page images; finding the fold in 319 survey spreads |
| Notebook 2, `"smoke"` | 10–15 min | Downloading the 7.5 GB model (first time only) |
| Notebook 2, `"plain"` | 15–30 min | Reading the 55 dev pages (90 images, with the survey halves) |
| Notebook 2, `"baseline"` | 15–35 min | The same, plus second readings of parts that look wrong |
| Notebook 2, `"full"` | 1–1.5 hours | Training: about 760 images per epoch (Dominy's used 4 times) × 2 epochs, ~40 min. A few dev pages after each epoch, ~10 min. All 55 dev pages at the end, 15–35 min. |
| Notebook 3 | 40–60 min | Reading the 208 test pages (239 images, with the survey halves) |
| **All of it, once** | **about 3.5–4 hours** | **about $5–6** |

Each extra `"full"` run with different settings adds 1 to 1.5 hours.

On an **`ml.g4dn.xlarge`** (T4 GPU, about $0.74 an hour), expect about 2.5 times as long, so
roughly 9 hours for everything, at about the same cost. The T4 can't train in bf16, and this model
in fp16 can fail with a `nan` loss, so the g5 (or `ml.g6.xlarge`, $1.13 an hour) is the safer
choice.

These times are estimates scaled from the lesson's measurements on the same model and instance
(about 1.45 seconds per page per epoch of training, about 21 seconds to read a page one at a
time; reading several at once is faster). Your smoke run gives the real speed.

## Part 1: The notebook instance

Create a notebook instance in the SageMaker console (Notebooks → Create notebook instance):

- **Instance type:** `ml.g5.xlarge` (or `ml.g6.xlarge`).
- **Volume size** (under Additional configuration): **50 GB**. The default 5 GB can't hold the
  model and the images.
- **IAM role:** the class role.

If it fails with a **quota** message, ask the class organisers, or request
"ml.g5.xlarge for notebook instance usage" under Service Quotas → Amazon SageMaker.

**It bills every hour it's running, even with the browser closed. Stop it when you're done**
(Notebooks → select it → Actions → Stop). Your files under `~/SageMaker` survive a stop.

## Part 2: Get this branch

Open JupyterLab, then **File → New → Terminal**, paste these two lines (Cmd+V on a Mac, Ctrl+V on
Windows) and press Enter:

```
cd ~/SageMaker
git clone -b lokesh/fine-tune-churro https://github.com/zwaseem128/Bucky-Bookworm---Badger-Scribe-MLM-26-.git churro
```

Then open `churro/churro-finetune/` in the file browser. To get updates later:
`cd ~/SageMaker/churro && git pull`.

## Where the data comes from

Notebook 1 needs the **full competition data** in S3, in one folder:

```
<folder>/train.csv
<folder>/test.csv
<folder>/images/kade_007_p005.jpg, survey_002_p0010.jpg, dominy_001_p002.jpg, ...
(optional) <folder>/metadata.csv, <folder>/sample_submission.csv
```

The class bucket `ml-marathon-shared` has only the Kade letters, so upload the Kaggle download to
your own bucket: get it from the competition's **Data** tab (Download All) and unzip it; in the
S3 console, open your bucket, **Create folder** `badger-scribe`, open it, **Upload** the CSV files
and **Add folder** for `images`. Then type your bucket's name in Notebook 1, Step 2. One person
doing this is enough; teammates use the same bucket name if their role can read it.

## Part 3: Run the notebooks

Open each notebook and pick the kernel **conda_pytorch_p310** (or another `conda_pytorch` kernel;
not `conda_python3`). Run the cells from top to bottom with **Shift + Enter**.

**Notebook 1** (once):
1. Set up the folders.
2. Download the pages. Type your bucket name first. If the files aren't where you said, it
   searches the bucket and tells you where they are.
3. Prepare the pages. Read the table: every collection should have training, dev and test pages.
   It also says how many survey spreads were cut, and how many labels were split into left and
   right pages (about half split at the right-hand page number; the rest are trained whole).
4. **Check a spread by eye.** A red line shows where it was cut; it should run down the fold, not
   through writing. Look at a few. If it's wrong on many, tell the team before training.
5. **See what the model trains on.** One page per collection, as it is and with three random
   changes. They should look like other scans of the same page, never like damage.

**Notebook 2** (each run):
1. Install the libraries (after every start of the instance, before any other cell).
2. Set up the folders.
3. Check the GPU, disk and data. It warns if the GPU is a T4.
4. Choose the run. Do `"smoke"` first, then `"plain"`, `"baseline"` and `"full"`. The step also
   has a table of the audit's problems and the setting for each fix.
5. Train and score. You can close the browser; it keeps going. `train.log` in the run's folder
   shows progress.
6. The scores: dev CER per collection and their mean (Kaggle's score), next to `"plain"` and
   `"baseline"`; the expert-typed pages on their own; and the length check (pages too short, too
   long, read twice) per collection.
7. The pages the length check flagged, then one page in detail with the model's mistakes coloured
   in.
8. Save the run to S3.

**Notebook 3** (when you have a good run):
1–2. Install and set up, as before.
3. Pick the run: the best by mean dev CER is picked for you.
4. Read the 208 test pages. If it stops, run the cell again; it carries on.
5. Make `submission.csv`: one row per test page, in Kaggle's order and format.
6. Download it (right-click → Download) and upload it on the competition's Submit Prediction page.

## The settings you can change

In Notebook 2, Step 4. Change **one at a time**, and write each run's settings and its three dev
scores in the team's results table.

| Setting | Starts at | In plain words | Change it when |
|---|---|---|---|
| `epochs` | 2 | How many times training goes through all the images. The lesson saw small gains up to 3. | The curve after each epoch is still falling: try 3. |
| `lr` | `2e-4` | How big each correction is. | Scores get worse after epoch 1: try `1e-4`. |
| `lora_r`, `lora_alpha` | 16, 32 | How big and how strong the adapter is. Keep alpha at twice r. | Training loss stays high: try 32 and 64. |
| `grad_accum` | 8 | How many images it sees before each correction. | Rarely. |
| `augment` | `"all"` | Random changes to training images (problems 2, 5 and 7): a tilt, brightness, contrast, sharpness; postcard-like shrinking and fading for Kade; faint bleed-through for Dominy. | Try `"none"` to measure what it's worth. |
| `oversample` | `"dominy_accounts:4"` | Uses a collection's images several times per epoch (problem 7). | Dominy's dev score lags the others: try `:6`. Try `""` to measure it. |
| `train_vision` | 0 | 1 also trains the part of the model that looks at the image, not just the part that writes. Might help with Kurrent letterforms. | Try once the others are settled. |
| `max_pixels` | about 1 megapixel | Image detail. More is slower and, per the lesson, rarely helps. | Small characters (fractions like ½) are misread. |
| `upscale` | 1 | Enlarges small images to the full pixel budget (problem 2). | Set 0 to measure it. |
| `enhance` | `""` | A contrast stretch for the named collections (problem 5), e.g. `"dominy_accounts"`. Off by default: the audit says to judge it on the dev scores. | Try `"dominy_accounts"`. |
| `retry` | 1 | Reads a part again, top and bottom separately, when its first reading is empty, far too short, or stuck repeating (problems 1 and 7). | Set 0 to measure it. |
| `postmark_guess` | 1 | Adds `Postmark , 19` on small Kade cards (problem 3). | Set 0 if the team decides against it. |

The reading settings (`max_pixels`, `upscale`, `enhance`, `retry`, `postmark_guess`) are saved
with each run, and Notebook 3 reads the test pages with exactly the chosen run's settings.

**Reading the scores.** The dev set is small: 11 Kade pages from one letter, 35 survey pages from
one notebook, 9 Dominy pages. A difference of 0.01 to 0.02 may be luck; run the same settings
twice to see how much scores move by themselves. The survey and Dominy labels are mostly machine
drafts that are about 6–8% wrong themselves, so scores much below 0.08 there mostly measure how
closely the model copies the drafting model.

## Troubleshooting

| What you see | What to do |
|---|---|
| `No module named 'PIL'` (or `'torch'`) | Wrong kernel. Kernel → Change Kernel → conda_pytorch_p310, then run from Step 1. |
| `AutoVideoProcessor requires the Torchvision library` | Run Step 1 again; it installs the right torchvision. Or switch to conda_pytorch_p310. |
| `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` | The script notices and carries on without cuDNN (a little slower). conda_pytorch_p310 avoids it. |
| `The training loss became NaN` | fp16 on a T4. Use an `ml.g5.xlarge` or `ml.g6.xlarge` instance. |
| `CUDA out of memory` | Lower `max_pixels` to `1024 * 28 * 28`, or check no other notebook is using the GPU (Kernel → Shut Down Kernel there). |
| Notebook 1: "This notebook isn't allowed to read <bucket>" | The bucket can stay private. The message prints a bucket policy naming only this notebook's role: in the S3 console open the bucket → **Permissions** → **Bucket policy** → **Edit**, paste it (or add its statement to the policy already there), **Save changes**, and run the cell again. Only the bucket's owner can do this. |
| Notebook 1: "There is no badger-scribe/train.csv" | The data isn't at that place in the bucket; the message says where `train.csv` is, or that it isn't there. |
| Notebook 1: a collection has 0 training pages | `train.csv` is the Kade-only class copy. Use the full competition data. |
| The red line in Step 4 cuts through writing | The fold wasn't found on that spread. Tell the team which spreads; the cut can be turned off with `prepare_data.build(DATA, "dev_ids.csv", split_spreads=False)`. |
| Many pages "read twice" in the length check | Their first readings came out empty, far too short or looping. If the second readings don't fix them, look at those pages in Step 7. |
| Want to start a run over | Delete its folder in `~/SageMaker/churro-work/runs/` and run Notebook 2 from Step 4. |

## Credits

The training script is adapted from `finetune_churro_lora.py` and `badger_common.py` in the
[ML with AWS SageMaker](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html)
lesson (MIT licence). The output clean-up and collection hints come from Lokesh's
`churro_simple.py`; the dev set and every problem tackled here from Lokesh's dev-set audit. CHURRO-3B is by
Stanford OVAL ([paper](https://arxiv.org/abs/2509.19768)), under the Qwen research licence
(non-commercial use).
