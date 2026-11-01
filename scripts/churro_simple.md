# How to run `churro_simple.py`

`churro_simple.py` sends page images to CHURRO (the handwriting model on BadgerBrain) one at a
time and saves what it reads. `score_churro.py` then tells you how accurate it was. You don't
need to know Python to run either one. Setup takes about 20 minutes the first time; after that
it's two commands.

**What goes in:** page images in the team's Google Shared drive, `Badger Scribe → images`.
**What comes out:** one text file per page, plus one spreadsheet with everything, in the same
Shared drive under `Badger Scribe → output data`, so the whole team sees the results.

## One-time setup

### 1. Get a BadgerBrain key

Request one with the [BadgerBrain access form](https://forms.gle/vkcLzApNrX7KbkTP9). Give your
NetID and say you're on Badger Scribe (MLM26). Two things happen behind the scenes, both with a
lead time of a day or so: your key arrives as a 1Password share link, and your NetID is added to
the campus firewall. Full details in the
[gateway quickstart](https://github.com/qualiaMachine/RunAI_apps/blob/main/docs/gateway-quickstart.md).

Never paste the key into a file, a notebook, or a chat. Everything you run is logged against it.

### 2. Install Google Drive for desktop and turn on streaming

The images are ~300 MB and live in a Shared drive. "Streaming" means they appear on your
computer as ordinary files but only download when something opens them — nothing fills your
disk.

1. Ask whoever owns the **Badger Scribe** Shared drive to add your `@wisc.edu` account.
2. Download and install [Google Drive for desktop](https://www.google.com/drive/download/).
3. Sign in with your **@wisc.edu** account (not a personal Gmail).
4. Click the Drive icon in the system tray / menu bar → gear → **Preferences** →
   **Google Drive** → make sure **Stream files** is selected (it's the default).
5. Check that the folder is visible:
   - **Windows:** open File Explorer → `G:\Shared drives\Badger Scribe\images`. If your drive
     letter isn't `G:`, see "Mac or a different drive letter" below.
   - **Mac:** Finder → `Library/CloudStorage/GoogleDrive-<you>@wisc.edu/Shared drives/Badger Scribe/images`
     (older versions: `/Volumes/GoogleDrive/Shared drives/...`).

   You should see files named like `kade_001_p001.jpg`. If the folder is empty or missing,
   you haven't been added to the Shared drive yet.

Optional but recommended: right-click the `images` folder → **Offline access → Available
offline**. Then the script doesn't wait for a download on every page.

### 3. Install Python and the packages the scripts need

- Install Python 3 from [python.org](https://www.python.org/downloads/) if you don't have it
  (on Windows, tick **"Add python.exe to PATH"** during install).
- Open a terminal (Windows: **PowerShell**; Mac: **Terminal**) and run:

  ```
  pip install openai pillow pandas
  ```

### 4. Get the scripts

Either clone this repo, or download `scripts/churro_simple.py` and `score_churro.py` from GitHub
into the same folder, somewhere you can find again (e.g. your Desktop).

## Every time you run it

1. **Connect to the UW VPN (GlobalProtect)** — required even on campus wifi.

2. **Open a terminal** and go to the folder that has the script:
   ```
   cd Desktop
   ```

3. **Give the script your key** for this session (it won't be saved anywhere):
   ```powershell
   # Windows PowerShell
   $env:CHURRO_API_KEY = "sk-..."
   ```
   ```bash
   # Mac
   export CHURRO_API_KEY=sk-...
   ```
   If you skip this, the script asks for the key when it starts (typing is hidden).

4. **Run it:**
   ```
   python churro_simple.py
   ```

5. It first shows how many images of each kind are on the drive, and warns if some are
   missing. Then it asks:
   ```
   Type 'eval' to test on labeled train pages from every category,
   or how many images to transcribe (a number, or 'all'):
   ```
   - `3` the first time, to make sure everything works.
   - `eval` to measure how good CHURRO is (see the next section).
   - `all` to transcribe every image, for a Kaggle submission.

   Each page prints `done in 45s` when it finishes. **The very first page can take 2–3
   minutes**: CHURRO goes to sleep when nobody is using it and needs to wake up. That's
   normal; the second page is fast.

6. When it finishes, open `Badger Scribe → output data` on the Shared drive:
   - `kade_001_p001.txt` etc.: one file per page, what CHURRO read. The file name is the
     page's Kaggle `page_id`.
   - `transcriptions.csv`: every finished page in one spreadsheet
     (columns: `image_id`, `collection`, `transcription`)

## Checking how good it is: `eval` and `score_churro.py`

Type `eval` at the question. The script then transcribes the same 36 training pages every time
(12 German letters, 12 account-book pages, 12 survey pages), whose correct text we already have.
Because the pages never change, a score from today can be compared with a score from next week.
If some of those pages aren't on the drive yet, it skips them and says which.

When it's done, run:

```
python score_churro.py
```

It prints an error rate for each category and one overall number. It scores only the pages
from your last `eval` run, so leftovers in the folder from an `all` run or an older prompt
don't get mixed in. A page whose request failed during that run isn't scored either. The
results are also saved in `output data → scores`: `summary.txt` (the table) and
`page_scores.csv` (every page, CHURRO's text next to the correct text).

Lower is better: 0 means perfect, 1 means every character was wrong. The account-book pages
are mostly scored against unchecked machine drafts, not human transcriptions, so trust the
German and survey numbers more. The summary says how many pages that applies to.

To score everything in the folder instead (for example, after an `all` run), use
`python score_churro.py all`. Test pages can't be scored, since we don't have their answers,
so they're listed as "no label".

## Starting over vs. picking up where you left off

Near the top of the script there's a line:

```python
START_OVER = True
```

- `True`: transcribes every chosen page again and **overwrites** earlier results.
- `False`: skips pages that already have a `.txt` file, so you can run it in chunks over
  several days, or fill in pages that failed last time.

If CHURRO returns an empty answer, the script tries again with a different prompt, then by
reading the top and bottom halves of the page separately. A page is only left without a file
when the requests themselves fail (network or gateway trouble); the script lists those at the
end, and a run with `START_OVER = False` picks them up.

## Mac or a different drive letter

Both scripts look for the Shared drive at `G:\Shared drives\Badger Scribe`. If yours is
somewhere else, you don't need to edit the scripts. Tell them where it is with `BADGER_DRIVE`,
in the same terminal, before running them:

```powershell
# Windows PowerShell, e.g. if Drive is mounted as H:
$env:BADGER_DRIVE = "H:\Shared drives\Badger Scribe"
```

```bash
# Mac (replace YOU with your NetID)
export BADGER_DRIVE="$HOME/Library/CloudStorage/GoogleDrive-YOU@wisc.edu/Shared drives/Badger Scribe"
```

Point it at the `Badger Scribe` folder itself, not at `images` inside it. To find the exact
path on Mac: open the folder in Finder, right-click it, hold **Option**, and choose
**Copy "…" as Pathname**.

## When something goes wrong

| what you see | what it means | fix |
|---|---|---|
| `Can't find G:\Shared drives\Badger Scribe` | The script can't see the Shared drive | Check Google Drive for desktop is running and signed in with `@wisc.edu`; if the drive is elsewhere, set `BADGER_DRIVE` (see above) |
| `WARNING: missing pages for ...` | Some images haven't been uploaded to the drive yet | Fine for testing. A Kaggle submission would score the missing pages as blank, so upload them first |
| `There's no last_eval_pages.csv` (from `score_churro.py`) | No `eval` run has finished yet | Run `churro_simple.py` and type `eval` first, or use `python score_churro.py all` |
| Hangs, or `Unable to connect` / `Connection error` | Not on VPN, or your NetID isn't in the firewall yet | Connect GlobalProtect. If it still hangs, email Chris your NetID |
| `Invalid proxy key` or `401` | Wrong or expired key | Check for typos; request a new share link if it's expired |
| `Malformed API Key` | The key didn't reach the script | In PowerShell you must use `$env:CHURRO_API_KEY`, not `$CHURRO_API_KEY` |
| `No module named openai`, `PIL` or `pandas` | Package not installed | `pip install openai pillow pandas` (try `pip3` on Mac) |
| `404 ... Model Group` | Model name typo | It must be `churro-3b` |
| First page takes minutes | Cold start | Wait; it's normal once per session |
| `429` | Too many requests | Wait a minute and run again with `START_OVER = False` |
| Spreadsheet looks garbled in Excel | Umlauts (ä, ö, ü, ß) in the German pages | Open it via **Data → From Text/CSV** and choose UTF-8, or use Google Sheets |

## Rules that still apply

- CHURRO is an open-weight model, so sending it **any** page — train or test — is allowed.
  That is *not* true of ChatGPT, Claude, or Gemini: never send a test page to those.
- The output is a first draft, not ground truth. Don't upload `transcriptions.csv` to Kaggle
  as-is: Kaggle needs columns `page_id` and `text`, one row for every test page, in the order
  of `sample_submission.csv`. The submission notebook turns the `.txt` files into that.
