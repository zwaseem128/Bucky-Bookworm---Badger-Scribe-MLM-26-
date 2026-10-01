"""
Send images from the Badger Scribe shared drive to CHURRO-3b, one at a time.

Run it and answer the question:
  eval    transcribe a fixed mix of labeled train pages from all three
          categories, then run score_churro.py to see the macro CER
  Enter   pick a first and last image, each by number (0 is the first in
          the sorted folder) or by name (kade_012_p001), both included,
          to split the work across teammates or pick up
          where an earlier run stopped. Results for those pages replace
          any older .txt files with the same name.

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
import random
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
LABELS = DRIVE / "train.csv"
OUT = DRIVE / "output data"

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

# Eval mode: this many labeled train pages per category, human-labeled pages
# first. The same pages every time (fixed seed), so runs are comparable.
EVAL_PER_CATEGORY = 12
EVAL_SEED = 0


def eval_pages():
    rows = list(csv.DictReader(open(LABELS, encoding="utf-8-sig", newline="")))
    rng = random.Random(EVAL_SEED)
    picked = []
    for cat in sorted({r["category"] for r in rows}):
        pool = [r for r in rows if r["category"] == cat]
        rng.shuffle(pool)
        pool.sort(key=lambda r: r["label_source"] != "human")  # stable: human first
        picked += [IMAGES / f"{r['page_id']}.jpg" for r in pool[:EVAL_PER_CATEGORY]]
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


def ask_image(question, images, default):
    """Ask for one image by its number or its name; return its position.

    A number is the position in the sorted list (0 is the first image). A name
    is the file name with or without .jpg, e.g. kade_012_p001. Enter gives
    default. Any image in the folder is allowed.
    """
    names = {p.stem.lower(): i for i, p in enumerate(images)}
    high = len(images) - 1
    while True:
        answer = input(f"{question} (number 0-{high} or an image name, "
                       f"Enter for {images[default].stem}): ").strip()
        if not answer:
            return default
        key = answer.lower().removesuffix(".jpg")
        if key.isdigit() and int(key) <= high:
            return int(key)
        if key in names:
            return names[key]
        print(f"Please type a number from 0 to {high}, or the name of an image in {IMAGES}.")


def ask_range(images):
    """Return (first, last) so that images[first:last + 1] is the chosen range.

    Both ends are included, so picking the same image twice runs just that one.
    The two ends can be typed in either order.
    """
    print(f"There are {len(images)} images, numbered 0 to {len(images) - 1} "
          f"({images[0].stem} to {images[-1].stem}).")
    first = ask_image("First image", images, 0)
    last = ask_image("Last image", images, len(images) - 1)
    if last < first:
        print( f"Running {last} to {first}, since {images[last].stem} comes first in the folder.")
        first, last = last, first
    return first, last


def main():
    if not DRIVE.exists():
        raise SystemExit(f"Can't find {DRIVE}. Is Google Drive for Desktop running "
                         "and signed in to the account that has the shared drive?")
    OUT.mkdir(parents=True, exist_ok=True)
    all_images = sorted(IMAGES.glob("*.jpg"))
    check_images(all_images)
    answer = input("Type 'eval' to test on labeled train pages from every category,\n"
                   "or press Enter to pick a range of images: ").strip().lower()
    while answer not in ("", "eval"):
        answer = input("Please type eval, or press Enter for a range: ").strip().lower()

    if answer == "eval":
        todo = eval_pages()
    else:
        first, last = ask_range(all_images)
        # Slice before the START_OVER filter so a number always means the same
        # image in the sorted list, however many pages are already done.
        todo = all_images[first:last + 1]
        print(f"Images {first} to {last}: {todo[0].name} through {todo[-1].name}. "
              f"Results go to {OUT}, replacing any with the same name.")
    if not START_OVER:
        todo = [p for p in todo if not (OUT / f"{p.stem}.txt").exists()]
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

    print(f"Done. All results are in {OUT / 'transcriptions.csv'}")
    if failed:
        print(f"{len(failed)} pages got no answer because the requests failed: "
              f"{', '.join(failed)}\nSet START_OVER = False and run again to fill them in.")
    if answer == "eval":
        print("Now run score_churro.py to see the CER for each category.")


if __name__ == "__main__":
    main()
