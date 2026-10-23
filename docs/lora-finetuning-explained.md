# Fine-tuning for the Kade letters, explained

This note explains, without assuming a technical background, how we teach a handwriting model to
read the Kade letters better, and how we run that on AWS SageMaker. It goes with
`notebooks/sagemaker_kade_lora.ipynb` (the notebook you run) and `sagemaker/code/` (the code the
notebook sends to AWS).

## The problem

Our models scored a character error rate (CER) of about 0.5 on the Kade letters: about half the
characters wrong. The letters are German, written in Kurrent, an old German handwriting that looks
nothing like modern script. The general-purpose models we tried weren't trained on enough of it.

## The three layers

Think of the model we use as three layers stacked on top of each other.

1. **Qwen3-VL-4B**, the base model. A general "look at an image and write about it" model made by
   Alibaba. It knows a lot about images and text, but not much about Kurrent. It has about 4.4
   billion numbers ("weights") inside it, which is what "4B" means.

2. **The Kurrent adapter** (`dh-unibe/qwen3vl-german-xix-v2`), made by the University of Bern. They
   took Qwen3-VL-4B and taught it 19th-century German handwriting, Kurrent included, using almost
   900,000 lines of handwriting. That's far more than we could ever collect. We use their work as
   our starting point instead of starting from scratch.

3. **Our adapter.** We teach that model one more thing: *these* letters. Anna Seifert's
   handwriting, her farm-German spelling (*jestern*, *bei die factry*), the English words she mixes
   in, and the exact way the Kaggle labels are typed.

## What "fine-tuning" means

Fine-tuning is showing the model examples with the right answers, and nudging it towards those
answers. For us an example is a picture of one line of a letter plus the text of that line. The
model reads the picture and guesses the text. We compare the guess with the right text,
character by character, and nudge the model's numbers so that next time the right text is a little
more likely. Then we repeat this thousands of times.

The number that measures "how wrong was the guess" is called the **loss**. It should go down as
training goes on.

## What "LoRA" means

Changing all 4.4 billion numbers would need an expensive computer and a lot of examples, and it
would be easy to damage what the model already knows. LoRA (Low-Rank Adaptation) is a shortcut.
We **freeze** the whole model and add a small set of new numbers next to it, about 33 million
(0.7% of the model). Only those new numbers are trained. At the end we save just them: a file of
about 130 MB instead of a 9 GB copy of the whole model. That file is the "adapter".

The Kurrent adapter from Bern is exactly this kind of file. We merge theirs into the model first
(so its Kurrent knowledge is built in and stays put), and then train our own small one on top.

## Why we work with single lines

The Bern adapter was trained on pictures of **one line** of handwriting at a time, and its authors
warn that it gets confused on whole pages. So before training, we cut every Kade page into lines
with a program called kraken (`notebooks/kade_line_segmentation.ipynb`, run on Ubuntu).

Our labels are whole pages, though, so we need to know which line of the label goes with which
picture. When kraken finds exactly as many lines on the page as the label has, we pair them in
order: picture 1 with line 1, picture 2 with line 2, and so on. Pages where the counts don't
match are left out of training for now.

To read a whole page, we read each of its lines and put them back together, one per row.

Two clean-up rules for the labels:

- **The `Postmark June 14, 1932` line is removed** from what we train on. The postmark is on the
  back of the card, so it isn't in the picture. If we trained on it, the model would learn to
  make up dates.
- **Spelling is never corrected.** The labels keep Anna's spelling exactly, and so should the
  model.

## How we know if it worked: the dev set

We keep 11 Kade pages aside that the model never trains on: all of document `kade_112`, the
team's agreed dev set (`docs/dev-set.md` on Lokesh's branch). After every round of training, the
model reads those 11 pages and we compute the CER, the same way Kaggle does. Lower is better.

The score is computed **before** any training too ("epoch 0"). That's the Bern adapter as it is,
and it's the number our training has to beat.

## What happens on AWS

SageMaker splits the work across three AWS pieces:

| Piece | What it is | What it costs |
|---|---|---|
| **Notebook instance** | A small computer that runs Jupyter. You write and run the notebook here. It has no GPU; it only *sends* the work. | About $0.05/hour. Stop it when you're done, or it keeps charging. |
| **S3 bucket** | Online storage. The line pictures go up here, and the trained adapter comes back here. | Pennies. |
| **Training job** | A GPU computer that AWS starts just for this job, runs our training script on, and shuts down when the script ends. | Only billed while the job runs. |

The notebook says to SageMaker: "start a GPU computer of this type, copy this folder of code and
these files onto it, run `finetune_qwen_kurrent_lora.py` with these settings, save what it
produces to my bucket, and never run longer than X minutes." That instruction is called an
**estimator**, and launching it is `estimator.fit(...)`.

Inside the job, the script:

1. Installs the right library versions (`requirements.txt`).
2. Downloads Qwen3-VL-4B and the Bern adapter, and merges them.
3. Adds our new, empty adapter.
4. Reads the 11 dev pages and records the score (epoch 0).
5. Trains, reading every training line once per **epoch**. After each epoch it reads the dev
   pages again and keeps the adapter from whichever epoch scored best.
6. Saves the best adapter, the scores, and a summary to `model.tar.gz` in your bucket.

The notebook then downloads that file and shows you the before and after scores.

**Checkpoints:** at the end of every epoch the job saves its progress to your bucket. If the job
is stopped partway, a new job pointed at the same checkpoint folder picks up from there instead
of starting over.

## Which GPU

| Instance | GPU | Price/hour | Good for |
|---|---|---|---|
| `ml.g5.xlarge` | A10G, 24 GB memory | ~$1.41 | The default, and what the SageMaker lesson uses. |
| `ml.g4dn.xlarge` | T4, 16 GB memory | ~$0.74 | Cheaper per hour but about 2x slower, so a run costs about the same. The T4 can't do the number format we'd normally train in ("bf16"), so it uses an older one ("fp16") that can occasionally break training: the loss shows up as `nan`. If that happens, switch to g5. |

Our pictures are single lines, which are small, so even the T4's 16 GB is enough. (The lesson
says a T4 is too small, but it was training on whole pages.)

Before your first job, check you have a **quota** for the instance type "for training job usage"
(AWS console → Service Quotas → Amazon SageMaker). It is often 0 until you ask.

## The settings you can change

All of them are in step 3 of the notebook, in the `"full"` entry. Change **one at a time**, so
you know which change made the difference.

| Setting | Starts at | In plain words | Change it when |
|---|---|---|---|
| `lr` (learning rate) | `1e-4` (0.0001) | How big each nudge is. Too big and the model forgets what Bern taught it; too small and nothing changes. | The dev score gets better, then worse → lower it (`5e-5`). Barely moves → raise it (`2e-4`). |
| `epochs` | 3 | How many times training goes through all the lines. | The best epoch is always the last one → try 4 or 5. The best is epoch 1 → 2 is enough. |
| `lora_r` (rank) | 16 | How big our adapter is: how much it can change. | Loss stays high → try 32. Dev score worse than training suggests → try 8. |
| `lora_alpha` | 32 | How strongly the adapter's changes are applied. Keep it at 2 × `lora_r`. | Whenever you change `lora_r`. |
| `lora_dropout` | 0.05 | Randomly ignores a few of the adapter's numbers each step, so it can't just memorise. | Training loss falls fast but the dev score doesn't improve → 0.1. |
| `batch_size` × `grad_accum` | 4 × 4 = 16 | How many lines the model sees before each nudge. 16 is what Bern used. | Rarely. If the job runs out of memory, halve `batch_size` and double `grad_accum`. |
| `min_pixels` | 65536 | The smallest picture size the model works with. The Kade postcards are small and faint, so a bigger value means more detail. | Try 131072 once the other settings are settled. It makes training slower. |

Things **not** to change: the prompt (Bern's adapter only knows its exact wording) and which
pages are in the dev set (or scores stop being comparable).

## How to read the results

The notebook prints, for each epoch, the dev CER:

```
 epoch  dev_cer
     0   0.21     <- the Bern adapter as it is (made-up numbers for illustration)
     1   0.15
     2   0.13     <- best: this adapter is the one that's saved
     3   0.14     <- slightly worse: it started memorising the training letters
```

Then each dev page's score before and after, and the text of the page that improved most, so you
can see what changed.

Two cautions:

- **Small differences may be luck.** 11 pages is a small test. Before believing that a change
  helped by 0.01 or 0.02, run the same settings twice and see how much the score moves on its own.
- **The dev set is one writer.** `kade_112` is postcards in pencil from one writer, and smaller
  images than the rest. A setting that helps there should still be checked on the Kaggle
  leaderboard.

## Order of work

1. Cut the pages into lines on Ubuntu (`docs/ubuntu-setup.md`, then the segmentation notebook),
   with `LIMIT = None`. Zip the result.
2. In SageMaker, open `notebooks/sagemaker_kade_lora.ipynb` and run it with `RUN = "smoke"`: a
   tiny run (under a dollar) that proves everything works.
3. `RUN = "baseline"`: the Bern adapter's score, no training.
4. `RUN = "full"`: the real fine-tune. Roughly 45 minutes and about $1 on g5 (an estimate until
   the smoke test gives a real number).
5. Change one setting, run again, compare. Write each run's settings and score in the team's
   results table.

Turning the trained adapter into Kaggle predictions for the test pages is the next step after
this, and isn't in the notebook yet.
