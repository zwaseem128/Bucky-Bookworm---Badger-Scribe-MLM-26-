# How to run `churro_simple.py`

`churro_simple.py` sends page images to CHURRO (the handwriting model on BadgerBrain) one at a
time and saves what it reads. You don't need to know Python to run it. Setup takes about
20 minutes the first time; after that it's two commands.

**What goes in:** page images in the team's Google Shared drive, `Badger Scribe → images`.
**What comes out:** one text file per page, plus one spreadsheet with everything, in
`Downloads → Bucky Bookworm Images → churro_output`.

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

### 3. Install Python and the one package the script needs

- Install Python 3 from [python.org](https://www.python.org/downloads/) if you don't have it
  (on Windows, tick **"Add python.exe to PATH"** during install).
- Open a terminal (Windows: **PowerShell**; Mac: **Terminal**) and run:

  ```
  pip install openai
  ```

### 4. Get the script

Either clone this repo, or download `scripts/churro_simple.py` from GitHub into a folder you
can find again (e.g. your Desktop).

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

5. It tells you how many images are in the folder and asks:
   ```
   How many images should I send to CHURRO? (a number, or 'all')
   ```
   Type `3` the first time to make sure everything works. Each page prints `done in 45s` when
   it finishes. **The very first page can take 2–3 minutes** — CHURRO goes to sleep when nobody
   is using it and needs to wake up. That's normal; the second page is fast.

6. When it finishes, open `Downloads → Bucky Bookworm Images → churro_output`:
   - `kade_001_p001.txt` etc. — one file per page, what CHURRO read
   - `transcriptions.csv` — every finished page in one spreadsheet
     (columns: `image_id`, `collection`, `transcription`)

## Starting over vs. picking up where you left off

Near the top of the script there's a line:

```python
START_OVER = True
```

- `True` — begins from the first image and **overwrites** any earlier results.
- `False` — skips pages that already have a `.txt` file, so you can run it in chunks over
  several days. Change it to `False` once your first test run works.

If CHURRO returns an empty answer for a page, the script doesn't save it, so that page is
retried automatically next run.

## Mac or a different drive letter

The script has this line near the top:

```python
IMAGES = Path(r"G:\Shared drives\Badger Scribe\images")
```

If your images are somewhere else, change it to your path. On Mac it's usually:

```python
IMAGES = Path.home() / "Library/CloudStorage/GoogleDrive-YOU@wisc.edu/Shared drives/Badger Scribe/images"
```

(replace `YOU` with your NetID). To find the exact path on Mac: open the folder in Finder,
right-click any file, hold **Option**, and choose **Copy "…" as Pathname**.

## When something goes wrong

| what you see | what it means | fix |
|---|---|---|
| `0 images in the folder` or `IndexError` | The script can't see the images | Check the Drive folder exists in Explorer/Finder; fix the `IMAGES` line |
| Hangs, or `Unable to connect` / `Connection error` | Not on VPN, or your NetID isn't in the firewall yet | Connect GlobalProtect. If it still hangs, email Chris your NetID |
| `Invalid proxy key` or `401` | Wrong or expired key | Check for typos; request a new share link if it's expired |
| `Malformed API Key` | The key didn't reach the script | In PowerShell you must use `$env:CHURRO_API_KEY`, not `$CHURRO_API_KEY` |
| `No module named openai` | Package not installed | `pip install openai` (try `pip3` on Mac) |
| `404 ... Model Group` | Model name typo | It must be `churro-3b` |
| First page takes minutes | Cold start | Wait; it's normal once per session |
| `429` | Too many requests | Wait a minute and run again with `START_OVER = False` |
| Spreadsheet looks garbled in Excel | Umlauts (ä, ö, ü, ß) in the German pages | Open it via **Data → From Text/CSV** and choose UTF-8, or use Google Sheets |

## Rules that still apply

- CHURRO is an open-weight model, so sending it **any** page — train or test — is allowed.
  That is *not* true of ChatGPT, Claude, or Gemini: never send a test page to those.
- The output is a first draft, not ground truth. Don't upload `transcriptions.csv` to Kaggle
  as-is — the pipeline in `plan.md` step 4 turns results into a valid submission.
