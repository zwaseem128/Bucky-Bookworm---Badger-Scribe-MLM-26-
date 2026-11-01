# The dev set: what it is and how to use it

This note explains step 3 of our plan, the "dev set". It covers what a dev set is, why we need
one, how the pages were picked, and how you use it when you test a model. You don't need any
coding background to follow it.

## What a dev set is

Kaggle only lets us upload two answer files a day, and the leaderboard only tells us one
overall number. That is far too slow and too vague for trying out ideas. We need a way to grade
ourselves, as often as we like, on our own computers.

Kaggle gives us a set of pages with the correct text already typed out (the "training" pages,
in `train.csv`). The dev set is a small group of those pages that we put aside and agree never
to learn from. Think of it as a practice exam. We keep the answer key, we don't peek at the
questions while studying, and every time we try a new idea we sit the same practice exam and
compare scores.

The important word is *same*. If Chloe tests CHURRO on one set of pages and Zain tests Qwen on
a different set, we can't tell whether the difference in scores comes from the models or from
the pages. With one shared dev set, the only thing that changes between two scores is the idea
being tested.

## How the pages were picked

The script that makes the dev set is `src/data/make_dev_set.py`. It follows three rules.

**Rule 1: whole documents only.** Many of our documents run over several pages; a letter might
be four pages long. If page 1 of a letter were in the dev set and page 2 were in training, the
model would get to "study" the writer's handwriting, names, and topic, and then be tested on the
same letter. It would look better than it really is. So the script moves whole documents into
the dev set, never a page here and a page there. Kaggle uses the same rule for its hidden test
pages, so this keeps our practice exam close to the real one.

**Rule 2: the survey notebook is the exception.** All of the surveyors' field notes come from a
single bound notebook, which counts as one document. If we followed rule 1 strictly, we would
have to put the entire notebook either in the dev set or in training, and then we'd have no
survey pages in one of the two. So for any category that consists of only one document, the
script picks individual pages instead. Again, this is exactly what Kaggle does.

**Rule 3: don't take too much.** Every page in the dev set is a page we can't later use to teach
(fine-tune) a model. So the script aims for about 15 pages per category, but never takes more
than a quarter of a category's pages, and always leaves at least one document behind for
training.

The choice within those rules is random, but it's a fixed kind of random: the script uses a
"seed" (the number 0), which means it makes the same choice every time it runs on the same
`train.csv`. Nobody can accidentally reshuffle it.

## What gets saved

The script writes two files.

`src/data/dev_ids.csv` is the list of which pages are in the dev set. It has four columns:
the page name, which document it belongs to, its category, and whether its correct text was
typed by a person (`human`) or is an unchecked machine draft (`silver_claude`). It does not
contain the transcriptions themselves. This file goes on GitHub, so the whole team shares one
list.

`data/dev_solution.csv` is the answer key: the same pages, with their correct text. It stays
on your own computer and is not uploaded to GitHub, because the transcriptions belong to the
libraries that hold the documents and we shouldn't republish them.

## Making it (done once)

This only needs to happen once for the whole team. After that, the list is fixed.

1. Download the competition data from Kaggle and put `train.csv` in the `data` folder of the
   repo, so its path is `data/train.csv`. (If it lives somewhere else, like the shared Google
   Drive, that's fine too; see the second command below.)
2. Open a terminal in the repo folder and run:

   ```
   python src/data/make_dev_set.py
   ```

   or, if `train.csv` is somewhere else:

   ```
   python src/data/make_dev_set.py --train "G:\Shared drives\Badger Scribe\train.csv"
   ```

3. It prints a small table, something like this (these numbers are from a test run on made-up
   data, not our real pages):

   ```
   category             train pages  dev pages  dev docs  human  silver
   dominy_accounts               23          5         2      0       5
   kade_letters                 136         19         5     19       0
   survey_notes                  30          7         1      0       7
   ```

   Read it row by row. Every category should have some dev pages. The German letters
   (`kade_letters`) should come from several documents, not one. If `kade_letters` shows
   `dev docs` as 1, stop and tell the team, because it would mean every German page shares a
   single document number, and rule 1 wouldn't be protecting us.

4. Commit `src/data/dev_ids.csv` and open a pull request.

Once the list is on GitHub, running the script again will not pick new pages. It sees that
`src/data/dev_ids.csv` already exists and keeps it exactly as it is. That is deliberate:
changing the dev set halfway through would make all our earlier scores incomparable. If the team
ever agrees to change it (for example, when Kaggle adds more verified English pages), the script
can be told to pick again with `--force`, and everyone should re-score their models afterwards.

## Using it to score a model

Anyone who wants to score a model on the dev set needs their own copy of the answer key, since
it isn't on GitHub. Pull the latest code, make sure `train.csv` is in your `data` folder, and run
the same command as above:

```
python src/data/make_dev_set.py
```

Because the shared list already exists, the script leaves it alone and just builds
`data/dev_solution.csv` on your computer for exactly those pages. It says "kept the existing dev
set" to confirm. You only need to do this once, or again if Kaggle updates `train.csv`.

Then run your model on the dev pages and save its output as a CSV with exactly two columns,
`page_id` and `text`. Score it with the scoring file Kaggle gave us:

```
python metric.py --solution data/dev_solution.csv --submission your_predictions.csv
```

It prints an error rate for each category and one overall number. Lower is better: 0 means
perfect, 1 means every character was wrong. Put the overall number and the per-category numbers
in our results table.

One mistake to avoid: your predictions file must have only the two columns `page_id` and
`text`. If it has extra columns (such as `category`), Kaggle's scoring script crashes with a
confusing error that says `KeyError: 'category'`.

## Things to keep in mind

The English answer keys are mostly machine drafts, not checked by a person. Kaggle says these
drafts get roughly 6 to 8 characters in every 100 wrong, and they tend to modernise old
spelling. So an English score measures how close a model gets to the draft, which isn't quite
the truth. The German answer keys were all typed by experts, so German scores are the ones to
trust most. The `label_source` column in the dev list lets us separate the two if we need to.

The dev set is small, so scores will wobble a bit from run to run. A difference of a hundredth
or two between two ideas may just be noise. Plan step 8 says to run the baseline twice to see
how big that wobble is, before we believe any improvement.

Nothing here touches the test pages. The script only reads `train.csv`.
