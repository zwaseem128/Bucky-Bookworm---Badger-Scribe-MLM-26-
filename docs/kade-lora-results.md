# Kade LoRA: results

This page is where the fine-tuning runs get written down, and where they turn into something we
can show the rest of ML+X. `docs/lora-finetuning-explained.md` explains how the training works;
this page is about what we found.

## What we want to be able to say

A good progress update makes a few clear claims and backs each one with a number. The runs in the
notebooks are designed so that each one supports one of these claims:

1. **Fine-tuning helps on letters the model has never seen.** This comes from comparing the
   `baseline` run with the `full` run on `kade_007`. Kaggle's Kade test pages all come from five
   documents that aren't in the training data, so `kade_007`, which is also kept out of training,
   is the fairest preview of the leaderboard we have. `kade_112` is reported too, but it is only
   11 small postcards, so it shouldn't carry the argument on its own.

2. **The improvement is bigger than chance.** Two things support this. The first is the 95%
   interval that the notebook prints for every run: if the whole interval sits above zero, the
   drop isn't luck in which pages happened to be scored. The second is the `repeat` run, which
   trains again with nothing changed except the random seed. The gap between `full` and `repeat`
   shows how much a score moves on its own, and any setting we later call "better" should beat
   `full` by more than that gap.

3. **Starting from the Bern Kurrent adapter matters (or it doesn't).** The `no_kurrent` run
   fine-tunes plain Qwen3-VL on the same lines. If it ends up close to `full`, our own data is
   doing most of the work. If it ends up far behind, the 900,000 lines Bern trained on are what
   make this possible. Either answer is worth telling people.

4. **It carries over to Kaggle.** The `baseline` and `final` runs both read the 141 Kade test
   pages and produce a submission. Comparing their leaderboard scores shows whether the
   improvement we measured on our own pages also shows up on the real test.

## Runs

The notebook prints one line for this table at the end of every run. Paste it here and fill in
the last column once a submission has been scored.

- **CER before** is the Kurrent adapter with no training, scored exactly as Kaggle scores it.
- **CER after** is the same pages at the epoch we kept.
- **Reading CER** leaves out the `Postmark` line, which is on the back of the card and can't be
  read from the image. It shows how well the model reads what is actually there.
- **Drop [95% interval]** is how much the CER fell, and the range we can trust it to.

| Run | Settings | CER before | CER after | Reading CER after | CER by document | Drop [95% interval] | Time and cost | Leaderboard Kade CER |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Leaderboard

If the submission was built on `sample_submission.csv`, every non-Kade page is blank and scores
exactly 1.0. The Kade score is then **3 × leaderboard score − 2**. If it was built on a full team
submission, the leaderboard score mixes in the other two categories, so note which one was used.

| Run | Built on | Leaderboard score | Kade CER |
|---|---|---|---|
| | | | |

## Turning this into an ML+X update

Once the four claims have numbers, the update almost writes itself. A shape that works well:

1. **The problem, in one picture.** One Kade line image next to its label, and the old CER of
   about 0.5. Most people in the room have never seen Kurrent, and seeing it makes it obvious why
   general models struggle with it.
2. **What we did.** The three layers (Qwen3-VL, Bern's Kurrent adapter, our adapter on top), why
   we work line by line, and that only about 0.7% of the model is trained.
3. **The evidence.** The CER-by-epoch chart from the notebook for `kade_007`, the before and after
   numbers with their interval, and the `repeat` gap next to them, so the audience can see the
   improvement is real.
4. **One page, before and after.** The notebook prints the page that improved most. Show its
   reference, the baseline reading, and our reading side by side, and point at two or three words
   the model now gets right.
5. **What didn't work, and what's next.** Include the `no_kurrent` result whichever way it went,
   the pages kraken couldn't split cleanly, and any setting that changed nothing. These are often
   the most useful part for other teams.

The same table also covers sections 4 ("What we tried") and 5 ("Results") of the Kaggle write-up
template, so nothing here is wasted at the end of the competition.

## Things to say out loud

- `kade_007` is one document by the same writer as everything else. It is a much better check
  than `kade_112`, but it is still one document, and the leaderboard is the final word.
- The CER includes the `Postmark` line we can't read, which sets a floor of a few points on
  roughly a quarter of the pages. That's why the reading CER is reported next to it.
- Pages where kraken found a different number of lines than the label has are left out of
  training. The notebook prints how many. If that number is large, better line segmentation is
  probably the next thing worth working on.
