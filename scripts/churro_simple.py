"""
Send images from the Badger Scribe shared drive to CHURRO-3b, one at a time.

Run it and answer the question:
  eval    transcribe a fixed mix of labeled train pages from all three
          categories, then run score_churro.py to see the macro CER
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
import random
import re
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

# The Kaggle download in the team repo. The shared drive's images folder is
# missing most survey pages (35 of 319), so don't point this back at it.
DATA = Path.home() / "Desktop" / "BuckyBookworm" / "data"
IMAGES = DATA / "images"
LABELS = DATA / "train.csv"
# Results go to the team's shared drive. score_churro.py reads from the same
# folder, so change both if this moves.
DRIVE = Path(r"G:\Shared drives\Badger Scribe")
OUT = DRIVE / "output data"

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
    return picked


# True: overwrite old results. False: skip images that already have a .txt
# file (use this to fill in pages whose requests failed last time).
START_OVER = True


def main():
    if not DRIVE.exists():
        raise SystemExit(f"Can't find {DRIVE}. Is Google Drive for Desktop running "
                         "and signed in to the account that has the shared drive?")
    OUT.mkdir(parents=True, exist_ok=True)
    all_images = sorted(IMAGES.glob("*.jpg"))
    answer = input("Type 'eval' to test on labeled train pages from every category,\n"
                   "or how many images to transcribe (a number, or 'all'): ").strip().lower()
    while not (answer.isdigit() or answer in ("all", "eval")):
        answer = input("Please type eval, a number like 5, or all: ").strip().lower()

    todo = eval_pages() if answer == "eval" else all_images
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

    print(f"Done. All results are in {OUT / 'transcriptions.csv'}")
    if failed:
        print(f"{len(failed)} pages got no answer because the requests failed: "
              f"{', '.join(failed)}\nSet START_OVER = False and run again to fill them in.")
    if answer == "eval":
        print("Now run score_churro.py to see the CER for each category.")


if __name__ == "__main__":
    main()
