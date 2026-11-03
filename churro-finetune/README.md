# Fine-tuning CHURRO on all three collections

This folder fine-tunes **CHURRO-3B**, an open handwriting model trained on 100,000 historical
pages, on whole pages from all three Badger Scribe collections at once: the Kade letters, the
survey field notes and the Dominy account books. It runs entirely in a SageMaker notebook in your
browser, so it works the same on a Mac or Windows.

It is the "strong baseline" from the team's evaluation of fine-tuning options: the recipe from
the SageMaker lesson (which took the Kade letters from a CER of 0.267 to 0.065 for $0.75 of GPU),
with three changes:

- **Honest scores.** Runs are scored on the team's 55 dev pages (`dev_ids.csv`: whole documents
  set aside, never trained on), with the competition's own `metric.py`, per collection. The lesson
  held out random pages from letters it also trained on, which makes scores look better than the
  leaderboard will.
- **Survey spreads read one page at a time.** Each survey image is an open notebook, two pages side
  by side. Reading across the fold mixes the pages up, which caused most of the survey errors in
  Lokesh's audit. Spreads are cut at the fold and read left page, then right page.
- **Cleaner labels.** The Kade `Postmark` line (on the back of the card, not in the image) is taken
  out of what the model learns, and the model is never trained to write `#`.

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
| Notebook 2, `"baseline"` | 15–30 min | Reading the 55 dev pages (90 images, with the survey halves) |
| Notebook 2, `"full"` | 1–1.5 hours | Training: about 680 images × 2 epochs, ~35 min. A few dev pages after each epoch, ~10 min. All 55 dev pages at the end, 15–30 min. |
| Notebook 3 | 40–60 min | Reading the 208 test pages (239 images, with the survey halves) |
| **All of it, once** | **about 3–3.5 hours** | **about $4–5** |

Each extra `"full"` run with different settings adds 1 to 1.5 hours.

On an **`ml.g4dn.xlarge`** (T4 GPU, about $0.74 an hour), expect about 2.5 times as long, so
roughly 8 hours for everything, at about the same cost. The T4 can't train in bf16, and this model
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

**Notebook 2** (each run):
1. Install the libraries (after every start of the instance, before any other cell).
2. Set up the folders.
3. Check the GPU, disk and data. It warns if the GPU is a T4.
4. Choose the run. Do `"smoke"` first, then `"baseline"`, then `"full"`.
5. Train and score. You can close the browser; it keeps going. `train.log` in the run's folder
   shows progress.
6. The scores: dev CER per collection and their mean (Kaggle's score), next to the baseline.
7. One page in detail, with the model's mistakes coloured in.
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
| `augment` | `"dominy_accounts"` | Small random changes to training images (a slight tilt, brightness, contrast, blur), so the 27 Dominy training pages go further. | Try `"all"`, or `""` to compare without. |
| `train_vision` | 0 | 1 also trains the part of the model that looks at the image, not just the part that writes. Might help with Kurrent letterforms. | Try once the others are settled. |
| `max_pixels` | about 1 megapixel | Image detail. More is slower and, per the lesson, rarely helps. | Small characters (fractions like ½) are misread. |

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
| Notebook 1: "There is no badger-scribe/train.csv" | The data isn't at that place in the bucket; the message says where `train.csv` is, or that it isn't there. |
| Notebook 1: a collection has 0 training pages | `train.csv` is the Kade-only class copy. Use the full competition data. |
| The red line in Step 4 cuts through writing | The fold wasn't found on that spread. Tell the team which spreads; the cut can be turned off with `prepare_data.build(DATA, "dev_ids.csv", split_spreads=False)`. |
| Want to start a run over | Delete its folder in `~/SageMaker/churro-work/runs/` and run Notebook 2 from Step 4. |

## Credits

The training script is adapted from `finetune_churro_lora.py` and `badger_common.py` in the
[ML with AWS SageMaker](https://carpentries-incubator.github.io/ML_with_AWS_SageMaker/09_Finetuning-BadgerScribe.html)
lesson (MIT licence). The output clean-up and collection hints come from Lokesh's
`churro_simple.py`; the dev set and the survey findings from Lokesh's dev-set work. CHURRO-3B is by
Stanford OVAL ([paper](https://arxiv.org/abs/2509.19768)), under the Qwen research licence
(non-commercial use).
