# Dev set error audit: what makes models get the dev pages wrong

What is on the 55 dev pages that makes models get them wrong, ranked by how much each problem
costs us on the leaderboard metric. Built from the images themselves, the dev answer key, and
CHURRO's existing outputs on the survey pages. The goal is to know what LoRA fine-tuning has to fix
before we start it.

Prepared by Lokesh, 6 Oct 2026. Dev pages only, from `src/data/dev_ids.csv`. No test images were
opened. Per-page numbers are in `docs/dev-set-error-audit-stats.csv` (columns: label source, file
size, image size, brightness, line and character counts, `#` marks, dittos, and CHURRO's CER where
we have an output). A version of this report with example images is at
https://claude.ai/artifact/AzSVD42HAbmK6VE214Pgm3. The images aren't in this repo, because the
Kaggle data stays out of git.

## The short version

On the survey dev pages, CHURRO's mean CER is **0.24**. **77% of that error comes from 12 of the 35
pages**, and on those 12 the model read the handwriting fine. It lost its place on a two-page
spread: it stopped early, read a page twice, or read the two pages in the wrong order. On the other
23 pages it scores **0.083**, which is already about the noise floor of the machine-drafted labels.

So for English, the first thing LoRA has to teach is *where to read and when to stop*, not what
the letters are. For German, the biggest problems are the image size, a line of text in the labels
that is not on the page, and the dialect spelling.

## What is in the dev set

| Category | Pages | Documents | Label source | Image size (px) | What the pages are |
|---|---|---|---|---|---|
| dominy_accounts | 9 | 4 | all machine draft | ≈1700 × 1060–2760 | Letters, bills, a 1796 ledger. Stained, folded, bleed-through. |
| kade_letters | 11 | 1 | all human | 693 × 1151 | Pencil Kurrent postcards, 1932–33, all from `kade_112`. |
| survey_notes | 35 | 1 | 33 machine draft, 2 human | ≈3130 × 2590 | Two-page spreads from one 1853 field notebook (`survey_010`). |

Each category counts for one third of the score, however many pages it has. The CHURRO run on the
shared drive covers the survey dev pages only. It has no output for the Dominy or Kade dev pages,
so findings for those two come from the images and labels, not from model errors.

## Survey: where CHURRO's error comes from

I sorted the 35 survey pages by comparing each output with its label line by line: how much text
came out, and in what order.

| What happened | Pages |
|---|---|
| Stopped early | 4 |
| Read part of the spread twice | 6 |
| Read the two pages interleaved | 2 |
| Layout read correctly | 23 |

If the 12 broken pages scored like the 23 good ones, the survey CER would fall from 0.24 to 0.083.
The label noise floor is about 0.06–0.08.

## Likely causes of error, ranked

Ranked by roughly how many CER points each costs us, given how the metric works. Page CER is edit
distance over the whole page after whitespace is collapsed. That makes three things expensive:
reading order (a block in the wrong place is charged as a deletion plus an insertion), short pages
(a small denominator), and text we cannot see.

### 1. Survey: two-page spreads break reading order (≈16 CER points on survey)

Every survey image is a whole opened notebook, about 3100 px wide. The convention is to read the
full left page, then the full right page. The header runs across the gutter, though ("Township 29
North Range 1 West | 4th principal Meridian Wisconsin"), and the label splits it between the two
pages. Both pages also use the same Chains table, so each row on the left lines up with a row on
the right. A model reading across the image naturally weaves the two pages together. In practice
CHURRO either stopped after a few lines (p0030 output 121 of 1080 characters), went back and
re-read the left page (p0025), or alternated between the pages (p0012).

**For LoRA:** cut each spread at the gutter and transcribe the halves separately. About half the
survey train labels (146 of 288) contain the right-hand page number on its own line, so they can be
split automatically, and the rest need a quick look. Set `max_new_tokens` well above the longest
label, and flag outputs below 0.7× or above 1.1× the expected length.

### 2. Kade: low resolution, pencil Kurrent, and words in English script

The dev Kade images are 693 × 1151 px, half the pixels of the other Kade documents (1.6–2.0 MP).
The writing is light pencil packed to the edges of a postcard, with about 23 lines per card and
short ascenders. The writer also switches script within a sentence: English loanwords like
*factry*, *mail*, *Farmers*, *road*, *car*, *Sat. Noon* are in Latin cursive, inside Kurrent
German. A model tuned only on Kurrent will force those words into German letterforms.

**For LoRA:** don't let the vision encoder shrink these further. Upscale 2× before encoding, or set
the processor's minimum pixel count so small cards get more image tokens. Train on line crops if we
go the TrOCR + Kraken route, since line height matters more than page size.

### 3. Kade: labels contain text that isn't on the page (≈4% CER floor on 10/11 dev pages)

10 of the 11 Kade dev labels end with a line like `Postmark June 14, 1932`. The image is the
message side of the card. The postmark is on the address side, and those odd-numbered pages aren't
in the data. The same line is in 79 of the 295 Kade train labels (median 4.3% of the label, up to
24%). No model can read a date it can't see, so these pages have a CER floor of about 0.04.

**For LoRA (team decision):** if we train on these labels as they are, the model learns to make up
postmark dates. Choices: (a) remove the line from training targets and accept the floor, or (b)
keep the word `Postmark` plus a year guessed from the letter, which gets partial credit. Either
way, decide on purpose. Don't leave it to the model.

### 4. Kade: dialect German spelled the way she wrote it

Anna Seifert writes American-farm German, and the labels keep her spelling exactly: *bei die
factry*, *jestern*, *mit denn Schlitten*, *bischen*, *solte*, *vieleicht*, *Dinstag*. A model with a
strong standard-German language model will "fix" these, and every fix costs characters. CHURRO's
Kade output on `kade_128` (CER 0.46) shows the opposite failure too: plausible German words that
aren't on the page.

**For LoRA:** all 295 Kade labels are human and verbatim, so they are good training data as they
are. Don't spell-correct them in preprocessing, and don't add a German spellchecker or dictionary
prior after decoding.

### 5. Dominy: damaged paper (staining, bleed-through, loss), 4 of 9 dev pages

The four `dominy_018` ledger pages are water-stained, and the ink from the back shows through as
mirrored text. The tops and left edges have faded or flaked off. The label for p0005 has 145 `#`
characters in the first four lines alone. A `#` in the label matches anything for free, but a `#`
in our output is an ordinary character and counts as an error.

**For LoRA:** mask `#` out of the training loss, or replace each run with nothing. Otherwise the
model learns to write `####` whenever it's unsure, which always costs us. Try contrast
normalisation (CLAHE) on the Dominy scans before encoding, and judge it on the dev set rather than
assuming it helps.

### 6. Dominy: ledger columns, money notation, and marks we must skip

The ledgers have a date column, a narrative column, a column of ✗ check marks, and £ / s / d
columns. The label keeps the numbers in row order and leaves out the ✗ marks and the dash leaders.
Account notation is dense: `7½ $`, `$0,,93¾`, `3/ -`, `2/0`, `""` for ditto, `Do.` The back of a
bill (`dominy_016_p003`) has the address upside down, sums written sideways, and a modern
archivist's pencil note that isn't in the label.

**For LoRA:** make the output alphabet match the conventions: ASCII plus `½ ¼ ¾ °` and the German
letters only. Map curly quotes, en and em dashes and `×` to plain characters after decoding. Try
rotating text that the layout model finds upside down (a 180° flip) before transcribing it.

### 7. All categories: short pages magnify every mistake

Page CER divides by the length of the label. `dominy_001_p003` is 36 characters long, so one wrong
word is about 10 CER points. `survey_010_p0036` is 161 characters, and CHURRO lost most of its 0.06
there by reading the page numbers 70 / 71 as `20.` / `21.` Page numbers, headers and short
addresses count for much more than their size.

**For LoRA:** include the near-empty pages in training (envelopes, covers, end pages) so the model
learns to write less on them, not to pad them. Weight evaluation by category, the way Kaggle does,
and not by character.

### 8. Survey: surveyor vocabulary and digits

On the pages where the layout went right, the errors left are small and repeat: 79→19, 40→50,
54→34, S→9/5, random→and (the writer runs *Eastrandom* together), corrected→connected,
Birch→Beech, Ironwood→Snowwood. Superscripts (2d rate, 4th), ½ written as 7/2, crossed-out text
that stays in, and insertions above the line all appear on nearly every page.

**For LoRA:** this is the part fine-tuning fixes best. The vocabulary is small and repetitive:
section numbers, bearings, tree species, soil rates. A few hundred spread-halves should teach it.

### 9. English labels: machine drafts limit what we can measure

42 of the 44 English dev labels are unchecked machine drafts, with an expected CER of 0.06–0.08
against the truth. The survey labels also disagree with the written conventions: the conventions
show chains as `40.00`, but all 279 chain entries in the survey dev labels are written `40 00`.
Below about 0.08 on English, the dev score mostly shows how well we copy the drafting model.

**For LoRA:** report the two human survey pages (p0007, p0008) as a separate line in our results.
Copy the labels' habits (`40 00`), not the conventions document, since the hidden test labels come
from the same pipeline.

## Caveats about the dev set itself

- **Kade dev is one document** (`kade_112`). `docs/dev-set.md` says to stop and tell the team if
  this happens. It is also the lowest-resolution Kade document, and the only one made of
  postcards, so 10 of 11 pages have the postmark line against 27% across train. Kade dev scores
  will look **worse** than test, and won't respond to fixes the way the other 284 pages would.
- **Survey dev is one surveyor's hand** in one 1853 township. Other survey documents have other
  hands. One of them (in `survey_003`) writes N so that it looks like π, and the first human pass
  misread it.
- **Only 55 pages.** On the Dominy pages one stained ledger page can move the category by 0.05.
  Before we trust a gain, check it against the run-to-run noise (plan step 8).

## Before we start LoRA: checklist

- [ ] Split survey spreads into left and right halves, both images and labels. Check the 142 labels
      that don't split automatically.
- [ ] Decide how to handle the `Postmark` line in Kade training targets.
- [ ] Mask `#` from the loss. Never train the model to write it.
- [ ] Upscale the small Kade images, or raise the processor's minimum pixel count.
- [ ] Add an output length guard (below 0.7× or above 1.1× expected) and set `max_new_tokens` high
      enough.
- [ ] Normalise output to the allowed characters after decoding.
- [ ] Get Dominy and Kade baseline outputs on the dev set, so each fix has a before and after number.
- [ ] Ask the team whether to add a second Kade document to dev.
