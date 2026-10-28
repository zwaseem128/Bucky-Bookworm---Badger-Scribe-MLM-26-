# Final fine-tuning setup: Kade letters, all on SageMaker

This folder is everything we need to fine-tune a handwriting model for the Kade letters, and
it runs entirely in a **SageMaker notebook in your web browser**. Nothing gets installed on
your own computer, so it works the same whether you're on a Mac or Windows. You only need a
browser and access to the class's AWS account.

It's a standalone version of the work in the rest of this repository. The older folders
(`sagemaker/`, `notebooks/`, `docs/`) are still there, but you don't need them: everything
this folder uses is inside it.

## What's in the folder

| File | What it does |
|---|---|
| `1_prepare_lines.ipynb` | Downloads the Kade pages and cuts every page into one image per line with kraken. Run it once. |
| `2_finetune.ipynb` | The training. Run it as many times as you like, with different settings. |
| `3_transcribe.ipynb` | Uses your best training run to transcribe the test pages for Kaggle. |
| `scripts/` | The code the notebooks run. You don't need to open it. |
| `requirements-training.txt` | The library versions training needs. Notebooks 2 and 3 install them. |

## The idea, in a few paragraphs

Our models scored a character error rate (CER) of about 0.5 on the Kade letters: about half
the characters wrong. The letters are in **Kurrent**, an old German handwriting that most
models can't read.

We don't start from scratch. **Qwen3-VL-4B** is a general model that reads images. Researchers
at the University of Bern trained an add-on for it, the **Kurrent adapter**, on almost
900,000 lines of 19th-century German handwriting. We start from that, and teach it one more
thing: *these* letters. That means Anna Seifert's handwriting, her own spelling (*jestern*,
*bei die factry*), the English words she mixes in, and the way the Kaggle answers are typed.

Teaching a model this way is called **fine-tuning**. We show it a picture of a line together
with the correct text, and it adjusts itself to make that text a little more likely. Then it
does the same for the next line, thousands of times. We use **LoRA**, a way of fine-tuning
that keeps the original model untouched and trains only a small add-on of its own: about 33
million numbers, next to the model's 4.4 billion. That's what makes it fit on one affordable
GPU, and the result is a file of about 130 MB.

The Bern adapter reads **one line at a time**, so first we cut every page into lines with a
program called **kraken** (Notebook 1). To score the model on a whole page, we read each of
its lines and put them back together.

We keep 11 pages aside that the model never trains on: letter `kade_112`, the team's **dev
set**. After every round of training, the model reads those 11 pages and we compute the CER,
the same way Kaggle does. Lower is better. That's how we know whether training helped.

## What it costs

You pay for the notebook instance (the computer running Jupyter) **for every hour it is
running**, whether or not you're using it, and even with the browser closed.

| Instance | GPU | About per hour |
|---|---|---|
| `ml.g4dn.xlarge` | NVIDIA T4, 16 GB | $0.74 |
| `ml.g5.xlarge` | NVIDIA A10G, 24 GB | $1.41 |

A first full pass (Notebook 1, a smoke test, a baseline, one full training run, and Notebook 3)
takes about 3 hours on a g4dn, so about $2–3. The real risk is forgetting to stop the
instance: a g5 left on over a weekend is about $70. **Stop the instance every time you're
done** (see "Stopping and starting").

Both instances work. The g5 is about twice as fast, so a run costs about the same on either. The
T4 in the g4dn can't use the number format we'd normally train in, so it uses an older one that,
once in a while, makes training fail with a `loss` of `nan`. If that happens, switch to a g5.

## Part 1: Create the notebook instance (once)

1. Sign in to the AWS console for the class account, and check the region in the top-right
   corner is the one the class uses.
2. Open **Amazon SageMaker AI**. In the left menu, under **Applications and IDEs**, click
   **Notebooks**, then **Create notebook instance**.
3. Fill in:
   - **Notebook instance name:** something with your name, like `badger-finetune-yourname`.
   - **Notebook instance type:** `ml.g4dn.xlarge` (or `ml.g5.xlarge`).
   - Open **Additional configuration** and set **Volume size in GB** to **50**. The default 5
     GB is far too small: the model alone is 9 GB.
   - **IAM role:** choose the role the class set up (its name usually starts with
     `AmazonSageMaker-ExecutionRole`).
4. Click **Create notebook instance**. After a few minutes its status changes from *Pending*
   to *InService*.

If creating it fails with a message about a **quota** or **limit**, the account isn't allowed
to run that instance type yet. Ask the class organisers, or request it under
**Service Quotas → Amazon SageMaker → "ml.g4dn.xlarge for notebook instance usage"**.
Approval can take a day or more.

## Part 2: Get this folder onto the instance (once)

1. Next to your instance, click **Open JupyterLab**.
2. In JupyterLab, open a terminal: **File → New → Terminal**.
3. Copy these two lines, paste them into the terminal (Cmd+V on a Mac, Ctrl+V on Windows), and
   press Enter.

   ```
   cd ~/SageMaker
   git clone -b claude/bold-lamport-kdtrph https://github.com/zwaseem128/Bucky-Bookworm---Badger-Scribe-MLM-26-.git badger-scribe
   ```

4. In the file browser on the left, open `badger-scribe`, then `final-finetuning-setup`.

To get updates later, run `cd ~/SageMaker/badger-scribe && git pull` in a terminal.

## Part 3: Run the notebooks

Open each notebook by double-clicking it. If JupyterLab asks for a **kernel**, choose
**conda_pytorch_p310** (or whichever `conda_pytorch` kernel is listed). Run the cells one at a
time, from top to bottom, with **Shift + Enter** (the same on Mac and Windows). Each step
explains itself in the notebook; here's the overview.

### Notebook 1: `1_prepare_lines.ipynb` (once, about an hour)

1. **Set up the folders.** Everything goes in `~/SageMaker/badger-finetune`, which survives
   stopping the instance.
2. **Install kraken.** kraken gets its own separate Python environment, because it needs a
   different version of PyTorch than training does. About 3 minutes the first time; after
   that it's already there.
3. **Download** the Kade page images and the page lists from the class bucket.
4. **Cut the pages into lines.** Try 5 pages first (`LIMIT = 5`), check them, then set
   `LIMIT = None` for all 436, which takes about 45 minutes. If it stops, just run the cell again:
   finished pages are skipped.
5. **Look at the result.** One page next to its line images. Each image should hold exactly
   one whole line.
6. **Pair lines with the transcription.** Wherever kraken found as many lines as the
   transcription has, image 1 goes with line 1, image 2 with line 2, and so on. Other pages are
   left out of training. Read the first line it prints: we're hoping for 70% of pages or more.
7. **Save to S3** (optional), so teammates can skip this notebook.

### Notebook 2: `2_finetune.ipynb` (each run 10 minutes to 1.5 hours)

1. **Install the training libraries.** Run this every time you start the instance, because
   installed libraries are wiped when it stops.
2. **Set up the folders**, the same as Notebook 1.
3. **Check** the GPU, the disk space, and that Notebook 1 is done.
4. **Choose the run.** Do three, in order:
   - `"smoke"`: a tiny run, just to prove everything works.
   - `"baseline"`: no training; scores the Kurrent adapter as it is. This is the number to beat.
   - `"full"`: the real training, with the settings below.
5. **Train.** The progress appears under the cell. You can close the browser; training keeps
   going while the instance runs.
6. **See the results.** The dev score before training and after each round, a chart, and the
   page that improved most.
7. **Save the run to S3**, so it's safe and the team can see it.

### Notebook 3: `3_transcribe.ipynb` (about 30 minutes)

1. **Install the libraries** (as in Notebook 2).
2. **Set up the folders.**
3. **Pick a run.** Your best one is chosen automatically.
4. **Check on the dev pages.** Their score should match what Notebook 2 reported.
5. **Transcribe the 141 test pages**, line by line.
6. **Save `kade_predictions.csv`** in Kaggle's format: `page_id`, `text`.
7. **Download it** (right-click the file in the file browser → **Download**) or save it to S3.

These predictions cover the Kade pages only. If the team's Kaggle file also has the other kinds
of pages, replace just its Kade rows with these.

## The settings you can change

They're in Notebook 2, Step 4. Change **one at a time**, and write down each run's settings
and dev score in the team's results table, so you know what made the difference.

| Setting | Starts at | In plain words | Change it when |
|---|---|---|---|
| `lr` (learning rate) | `1e-4` (0.0001) | How big each correction is. Too big, and the model forgets what Bern taught it; too small, and nothing changes. | The dev score gets better and then worse: lower it to `5e-5`. It barely moves: raise it to `2e-4`. |
| `epochs` | 3 | How many times training goes through all the lines. | The best round is always the last: try 4 or 5. The best is round 1: 2 is enough. |
| `lora_r` | 16 | How big our add-on is, so how much it can change. | Training loss stays high: try 32. The dev score lags behind the training loss: try 8. |
| `lora_alpha` | 32 | How strongly the add-on acts. Keep it at twice `lora_r`. | Whenever you change `lora_r`. |
| `lora_dropout` | 0.05 | A little randomness during training, so it can't simply memorise. | Training loss drops fast but the dev score doesn't follow: try 0.1. |
| `batch_size` and `grad_accum` | 4 and 4 | How many lines it looks at before each correction (4 × 4 = 16). | Only if the GPU runs out of memory: halve `batch_size` and double `grad_accum`. |
| `min_pixels` | 65536 | The smallest image size the model works with. The Kade postcards are small and faint, so bigger can mean more detail. | Try 131072 once the others are settled. Training gets slower. |

Two things to leave alone: the instruction the model is given (the Bern adapter only knows its
exact wording) and which pages are in the dev set (or scores stop being comparable).

**Reading the results.** Notebook 2 shows the dev CER for each round, where round 0 is before
any training. It might look like this (made-up numbers):

```
 round  dev CER
     0   0.21    <- the Kurrent adapter on its own
     1   0.15
     2   0.13    <- the best: this is the version that's kept
     3   0.14    <- slightly worse: it started memorising the training letters
```

Two cautions. The dev set is only 11 pages, so a difference of 0.01 or 0.02 may be luck: run
the same settings twice to see how much the score moves by itself. And all 11 pages are one
writer's pencil postcards, so check real improvements on the Kaggle leaderboard too.

## Stopping and starting

**To stop:** in the SageMaker console, under **Notebooks**, select your instance and choose
**Actions → Stop**. Do this every time you finish.

**What survives a stop:** everything in `~/SageMaker`: this repository, the page and line
images, the kraken environment, the downloaded model, and all your runs.

**What doesn't:** the training libraries (re-run Step 1 of Notebook 2 or 3 after each start).

**To start again:** select the instance, **Actions → Start**, wait for *InService*, then
**Open JupyterLab**.

**Deleting** the instance (only when the project is over) erases everything on it, so first
save what you need to S3 with the last step of each notebook. A stopped instance still costs
a little for its storage (about $5 a month for 50 GB).

**Changing the instance type**, for example from g4dn to g5: stop it, choose **Edit**, change
the type, save, and start it again. Your files stay.

## Troubleshooting

| What you see | What to do |
|---|---|
| `No GPU` in Notebook 2 or 3 | The instance type has no GPU. Stop it, **Edit** it to `ml.g4dn.xlarge` or `ml.g5.xlarge`, start it. |
| `No module named 'transformers'`, or an error naming `Qwen3VL` | Step 1 of the notebook didn't run, or ran after other cells. Kernel → Restart Kernel, then run from Step 1. |
| `Access Denied` when downloading from the class bucket | The instance's IAM role can't read `ml-marathon-shared`. Ask the class organisers, or ask a teammate to share their line images (Notebook 1, Step 7). |
| `No space left on device` | The volume is too small. Stop the instance, **Edit** it, raise the volume size to 50 GB, start it. |
| `CUDA out of memory` | Halve `batch_size` and double `grad_accum` in Notebook 2, Step 4. |
| `loss` shows `nan` | Happens occasionally on a g4dn. Switch the instance to `ml.g5.xlarge`. |
| `kraken didn't install` | Run the cell again. If it still fails, delete the `kraken-env` folder in `~/SageMaker/badger-finetune` and run it once more. |
| Fewer than 70% of training pages usable in Notebook 1 | kraken is struggling with the pages. Look at a few in Step 5 and tell the team before training. |
| The browser lost connection during training | Training continues. Open `train.log` in the run's folder under `badger-finetune/runs/` to see where it is. |
| Want to start a run over | Delete its folder in `~/SageMaker/badger-finetune/runs/` and run Notebook 2 from Step 4. Running the same settings again otherwise continues the old run. |
