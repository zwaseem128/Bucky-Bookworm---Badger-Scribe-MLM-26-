"""
Send images from the Badger Scribe shared drive to CHURRO-3b, one at a time.

Run it and answer the question:
  eval    transcribe the same 36 labeled train pages every time (12 per
          category), then run score_churro.py to see the macro CER
  5, all  transcribe that many images (or all of them) for a submission

Each transcription is saved to "output data"/<image>.txt in the Badger
Scribe shared drive (G:/Shared drives/Badger Scribe), so the team sees it.
Every image gets a transcription: if the model answers with nothing (or only notes like [illegible]), the script
retries with other prompts, then reads the top and bottom halves separately.
Only a network or gateway failure leaves a page without a file, and the next
run with START_OVER = False picks those pages up.

Needs campus wifi or the UW VPN to reach the gateway.
"""

import base64
import csv
import getpass
import io
import os
import re
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

# Everything comes from and goes to the team's shared drive, so anyone with
# Google Drive for Desktop can run this without a local copy of the data.
# score_churro.py reads OUT too, so change both if this moves.
# Google Drive for Desktop usually mounts as G:, but not always; a teammate
# with a different letter can set BADGER_DRIVE to their folder instead.
DRIVE = Path(os.environ.get("BADGER_DRIVE", r"G:\Shared drives\Badger Scribe"))
IMAGES = DRIVE / "images"
OUT = DRIVE / "output data"
# Eval mode writes the pages it just transcribed here, so score_churro.py
# scores those and nothing else. A .csv, not a .txt: every .txt in OUT is
# read as a page transcription.
EVAL_LIST = OUT / "last_eval_pages.csv"

# The Kaggle download has 319 survey pages (288 train + 31 test). As of
# 2026-09-30 the drive's images folder holds only 35 of them and none of the
# test ones, so check_images() warns until the rest are uploaded.
EXPECTED = {"dominy": 72, "kade": 436, "survey": 319}

CATEGORY = {"dominy": "dominy_accounts", "kade": "kade_letters", "survey": "survey_notes"}


def category_of(path):
    return CATEGORY.get(Path(path).stem.split("_")[0], "")


# ---------------------------------------------------------------- prompts

SIMPLE_PROMPT = (
    "Transcribe all of the text in this historical document image exactly as "
    "written. Preserve original spelling, punctuation, capitalization, and line "
    "breaks. Do not correct, modernize, translate, or summarize. Output only "
    "the transcription."
)

# Short on purpose: CHURRO is a 3B model, and the long rule lists in the old
# DETAILED_PROMPT made it describe the page ("[Stylized eagle emblem]")
# instead of reading it. Anything a rule used to handle (dashes, quotes,
# fractions, notes, shelf numbers) is now fixed in clean() after the answer
# comes back, where it can't go wrong.
ATTEMPT_ALL_PROMPT = """Transcribe all of the handwritten and printed text in this historical document image, top to bottom.

Copy spelling, capitalization, punctuation and abbreviations exactly as written; do not correct or modernize anything. Every image contains text, so always answer. If a word is hard to read, write your best guess at its letters. Write only the words on the page: no descriptions, no notes, no brackets.

{hint}
Output only the transcription."""

# One line of context per collection. Names the model kept misreading are
# spelled out, and the ledger hint fixes the row order it scrambled.
HINTS = {
    "dominy": "This is a receipt, bill, account or letter from the papers of the "
              "Dominy family of East Hampton, New York (1790-1860); names like "
              "Dominy, Nathaniel Dominy and Sag Harbor appear often. Old spellings "
              "like shews, ballance and do (ditto) are correct as written. For "
              "account rows, write each row left to right: date, item, amount.",
    "kade": "This is a German family letter in Kurrent handwriting. Write it in Latin "
            "letters, keeping ä ö ü ß as written. English words in the letter stay "
            "in English.",
    "survey": "These are land-survey field notes. Each line starts with the chain "
              "distance, then the entry, for example: 40.00 Set quarter Section post",
}

# Which prompt to try first: "attempt_all" or "simple".
PROMPT_STYLE = "attempt_all"


def prompt_for(path, style):
    if style == "simple":
        return SIMPLE_PROMPT
    return ATTEMPT_ALL_PROMPT.format(hint=HINTS.get(Path(path).stem.split("_")[0], ""))


# ---------------------------------------------------------------- cleanup
# Every rule below was checked against train.csv (all 619 references). Only
# the fraction rule ever changes a reference: 2 pages write 3/4 or 1/4 with a
# slash, against 206 dominy and survey pages that write ½ ¼ ¾.

PLAIN = str.maketrans({"—": "-", "–": "-", "‘": "'", "’": "'",
                       "“": '"', "”": '"', "„": '"', "…": "...", "|": " "})
FRACTIONS = {"1/2": "½", "1/4": "¼", "3/4": "¾"}


def clean(text, category=""):
    text = text.translate(PLAIN)
    text = re.sub(r"```[a-z]*", "", text)                  # markdown code fences
    text = text.replace("**", "").replace("__", "")        # markdown bold
    text = re.sub(r"^\s*(transcription|text)\s*:\s*", "", text, flags=re.I)
    # Notes and descriptions ("[illegible]", "[Stylized eagle emblem]"). No
    # reference contains a square bracket, so every bracketed span is wrong.
    text = re.sub(r"\[[^\]\n]*\]?", "", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        # Archive shelf numbers pencilled in a corner, like 63x96.3.
        if re.fullmatch(r"\d{1,3}\s?x\s?\d{1,3}(\.\d+)?", line):
            continue
        # A line the model got stuck repeating. No reference repeats a line
        # back to back.
        if lines and line and line == lines[-1]:
            continue
        lines.append(line)
    text = "\n".join(lines)
    # A phrase repeated over and over inside one line.
    text = re.sub(r"(.{8,}?)(?:\s*\1){2,}", r"\1", text)
    # Survey and Dominy references write fractions as ½ ¼ ¾ joined to the
    # number (40.12½, 7½). Kade references write 1 1/2, so leave those alone.
    if category in ("survey_notes", "dominy_accounts"):
        for slash, glyph in FRACTIONS.items():
            text = re.sub(rf"(\d)\s+{slash}(?![\d/])", rf"\1{glyph}", text)
            text = re.sub(rf"(?<![\d/]){slash}(?![\d/])", glyph, text)
    # The longest reference is about 2,000 characters. Anything far past that
    # is the model looping, and past the reference length every extra
    # character is a guaranteed error.
    return text[:3000].strip()


# ---------------------------------------------------------------- the model

# Long survey scans reach 11 MB. Shrinking the longest side to this many
# pixels keeps requests fast without making the writing too small to read.
# Worth testing 1600 and 3000 with eval mode.
MAX_SIDE = 2400


def encode(img):
    img = img.convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def ask(client, image_b64, prompt, temperature):
    resp = client.chat.completions.create(
        model="churro-3b",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        temperature=temperature,
        # The longest page in train.csv is about 2,000 characters (~700
        # tokens). This cap stops a page stuck repeating itself.
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def transcribe(client, path):
    """Return (text, how) for one image. text is "" only if every request failed."""
    category = category_of(path)
    image = Image.open(path)
    whole = encode(image)
    other = "simple" if PROMPT_STYLE == "attempt_all" else "attempt_all"
    # At temperature 0 the same prompt gives the same empty answer, so each
    # retry changes the prompt, the image, or adds a little randomness.
    tries = [(PROMPT_STYLE, 0), (other, 0), ("halves", 0), (PROMPT_STYLE, 0.5)]
    fallback, errors = "", 0
    for style, temp in tries:
        try:
            if style == "halves":
                # Small models sometimes give up on a crowded or faint page but
                # read each half fine. Overlap a little so no line is cut.
                w, h = image.size
                top = image.crop((0, 0, w, int(h * 0.55)))
                bottom = image.crop((0, int(h * 0.45), w, h))
                raw = "\n".join(ask(client, encode(part), prompt_for(path, "simple"), 0)
                                for part in (top, bottom))
            else:
                raw = ask(client, whole, prompt_for(path, style), temp)
        except Exception as e:
            errors += 1
            print(f"request failed ({e.__class__.__name__}: {str(e)[:80]})", end=" ", flush=True)
            continue
        text = clean(raw, category)
        if text:
            return text, f"{style}, temp {temp}"
        # Keep any raw answer as a last resort: a wrong guess can never score
        # worse than a blank page (both cap at CER 1.0).
        fallback = fallback or raw.strip()[:3000]
        print(f"empty with {style}, retrying ...", end=" ", flush=True)
    if fallback:
        return fallback, "raw answer, nothing survived cleanup"
    return "", f"{errors} failed requests"


# ---------------------------------------------------------------- choosing images

# Eval mode transcribes these pages: 12 labeled train pages per category,
# human-labeled pages first. They were picked once (seed 0, from the 619-page
# train.csv of 2026-09-30) and written down here, instead of re-picked from
# train.csv on every run. train.csv grows during the competition, and
# re-picking would quietly swap in different pages, so scores from different
# weeks couldn't be compared. Change this list only if the team agrees, and
# re-score the baseline when you do.
EVAL_PAGES = (
    # dominy_accounts: only 2 human-labeled pages exist, the rest are silver
    "dominy_002_p002", "dominy_020_p002", "dominy_001_p002", "dominy_017_p003",
    "dominy_015_p002", "dominy_018_p003", "dominy_018_p004", "dominy_016_p003",
    "dominy_034_p003", "dominy_029_p004", "dominy_008_p004", "dominy_008_p002",
    # kade_letters: all human-labeled
    "kade_007_p064", "kade_103_p178", "kade_128_p039", "kade_103_p100",
    "kade_128_p020", "kade_128_p131", "kade_128_p136", "kade_128_p079",
    "kade_103_p053", "kade_128_p034", "kade_128_p137", "kade_128_p056",
    # survey_notes: all human-labeled
    "survey_002_p0006", "survey_002_p0005", "survey_007_p0011", "survey_006_p0007",
    "survey_011_p0007", "survey_007_p0012", "survey_003_p0005", "survey_001_p0002",
    "survey_009_p0012", "survey_011_p0008", "survey_010_p0008", "survey_004_p0006",
)


def eval_pages():
    picked = [IMAGES / f"{page_id}.jpg" for page_id in EVAL_PAGES]
    # Keep the same fixed picks, but skip any the drive doesn't have yet.
    missing = [p for p in picked if not p.exists()]
    if missing:
        print(f"Skipping {len(missing)} eval pages not on the drive yet: "
              f"{', '.join(p.stem for p in missing)}")
    return [p for p in picked if p.exists()]


def check_images(images):
    counts = {c: sum(p.stem.startswith(c + "_") for p in images) for c in EXPECTED}
    short = {c: n for c, n in counts.items() if n < EXPECTED[c]}
    print("Images on the drive: " + ", ".join(f"{c} {n}/{EXPECTED[c]}" for c, n in counts.items()))
    if short:
        # Each category is a third of the score, and a missing page counts
        # as a blank answer (CER 1.0).
        print(f"WARNING: missing pages for {', '.join(short)}. A submission from "
              "this run will score those pages as blank until they're uploaded.")


# True: overwrite old results. False: skip images that already have a .txt
# file (use this to fill in pages whose requests failed last time).
START_OVER = True


def main():
    if not DRIVE.exists():
        raise SystemExit(f"Can't find {DRIVE}. Is Google Drive for Desktop running "
                         "and signed in to the account that has the shared drive?")
    OUT.mkdir(parents=True, exist_ok=True)
    all_images = sorted(IMAGES.glob("*.jpg"))
    check_images(all_images)
    answer = input("Type 'eval' to test on labeled train pages from every category,\n"
                   "or how many images to transcribe (a number, or 'all'): ").strip().lower()
    while not (answer.isdigit() or answer in ("all", "eval")):
        answer = input("Please type eval, a number like 5, or all: ").strip().lower()

    todo = eval_pages() if answer == "eval" else all_images
    eval_set = list(todo) if answer == "eval" else []
    if not START_OVER:
        todo = [p for p in todo if not (OUT / f"{p.stem}.txt").exists()]
    if answer.isdigit():
        todo = todo[:int(answer)]
    print(f"{len(todo)} images to transcribe.")

    key = os.environ.get("CHURRO_API_KEY") or getpass.getpass("UW gateway API key: ")
    client = OpenAI(base_url="https://llm-gw01.doit.wisc.edu/v1", api_key=key, timeout=900)

    failed = []
    for i, path in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {path.name} ...", end=" ", flush=True)
        start = time.time()
        text, how = transcribe(client, path)
        if not text:
            failed.append(path.name)
            print(f"no answer ({how}), will retry next run")
            continue
        (OUT / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        print(f"done in {time.time() - start:.0f}s ({how})")

    # Gather every finished page into one CSV.
    with open(OUT / "transcriptions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "collection", "transcription"])
        for txt in sorted(OUT.glob("*.txt")):
            writer.writerow([txt.stem, txt.stem.split("_")[0],
                             txt.read_text(encoding="utf-8")])

    if answer == "eval":
        # Tell score_churro.py which pages this run produced. A page whose
        # requests failed is left out, even if an older .txt for it is still
        # in the folder, so a stale answer from an earlier prompt can't sneak
        # into the score. Pages skipped because START_OVER is False are kept.
        done = [p.stem for p in eval_set
                if p.name not in failed and (OUT / f"{p.stem}.txt").exists()]
        with open(EVAL_LIST, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["page_id"])
            writer.writerows([page_id] for page_id in done)

    print(f"Done. All results are in {OUT / 'transcriptions.csv'}")
    if failed:
        print(f"{len(failed)} pages got no answer because the requests failed: "
              f"{', '.join(failed)}\nSet START_OVER = False and run again to fill them in.")
    if answer == "eval":
        print(f"Now run score_churro.py to see the CER for each category "
              f"({len(done)} eval pages to score).")


if __name__ == "__main__":
    main()
