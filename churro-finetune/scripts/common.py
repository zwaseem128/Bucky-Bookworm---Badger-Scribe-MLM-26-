"""Shared pieces for fine-tuning CHURRO-3B on whole Badger Scribe pages.

Used by prepare_data.py, finetune_churro.py and transcribe.py, so the prompt,
the image handling and the text clean-up are identical in training and when
reading new pages.

Adapted from badger_common.py in the "ML with AWS SageMaker" lesson
(carpentries-incubator/ML_with_AWS_SageMaker, MIT licence), with the team's
additions: a hint per collection, survey spreads cut at the fold, and label
and output clean-up from Lokesh's churro_simple.py and dev-set audit.
"""

import re
import statistics

import pandas as pd
from PIL import Image, ImageFilter

MODEL_ID = "stanford-oval/churro-3B"

# Qwen2.5-VL (which CHURRO is built on) turns every 28 x 28 pixel block into one
# image token. Each image is resized to at most MAX_PIXELS, about a megapixel or
# 1,280 tokens: the lesson's setting. More pixels cost time and rarely help.
PATCH = 28 * 28
MAX_PIXELS = 1280 * PATCH
MIN_PIXELS = 256 * PATCH

CATEGORY = {"kade": "kade_letters", "survey": "survey_notes", "dominy": "dominy_accounts"}

INSTRUCTION = (
    "Transcribe all handwritten and printed text on this page exactly as "
    "written, in natural reading order. Keep the original spelling, "
    "capitalization, punctuation and abbreviations; do not modernize, "
    "translate, summarize or add commentary. Write one line of text per "
    "line on the page. Do not transcribe drawings, ruled lines or filler "
    "dashes."
)

# One line of context per collection (from churro_simple.py, checked against
# train.csv). The survey hint doesn't give a chain example: the conventions
# document writes 40.00, but the labels write 40 00, and the model should copy
# the labels.
HINTS = {
    "kade_letters": "This is a German family letter in Kurrent handwriting. Write it in Latin letters, "
                    "keeping ä ö ü ß as written. English words in the letter stay in English.",
    "survey_notes": "These are land-survey field notes. Each line starts with the chain distance, then the entry.",
    "dominy_accounts": "This is a receipt, bill, account or letter from the papers of the Dominy family of "
                       "East Hampton, New York (1790-1860). Old spellings like shews, ballance and do (ditto) "
                       "are correct as written. For account rows, write each row left to right: date, item, amount.",
}


def category_of(page_id: str) -> str:
    return CATEGORY.get(page_id.split("_")[0], "")


def build_prompt(category: str, doc_context: str = "") -> str:
    lang = "German (Kurrent script)" if category == "kade_letters" else "English"
    parts = [INSTRUCTION, f"Language: {lang}.", HINTS.get(category, "")]
    if doc_context:
        parts.append(f"Catalog record: {doc_context}")
    return "\n".join(p for p in parts if p)


def load_metadata(path) -> dict:
    """metadata.csv -> {doc_id: one-line catalog context}, if the file has the columns."""
    try:
        md = pd.read_csv(path, keep_default_na=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    if "doc_id" not in md.columns:
        return {}
    ctx = {}
    for r in md.to_dict("records"):
        bits = [str(r.get(k, "")) for k in ("title", "date", "language") if r.get(k, "")]
        prefix = f"{r.get('collection', '')}: " if r.get("collection", "") else ""
        if bits:
            ctx[r["doc_id"]] = prefix + "; ".join(bits)
    return ctx


def user_messages(prompt: str):
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def full_messages(prompt: str, answer: str):
    return user_messages(prompt) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]


# ---------------------------------------------------------------------------
# Survey spreads: every survey image is an open notebook, two pages side by
# side. Reading across the fold mixes the pages up, so each spread is cut in
# two and the halves are read separately, left then right.
# ---------------------------------------------------------------------------
def is_spread(width: int, height: int) -> bool:
    return width > 1.15 * height


def find_fold(img: Image.Image) -> int:
    """x position of the fold: the smoothest column (least writing) in the middle fifth."""
    small = img.convert("L")
    scale = 800 / small.width
    small = small.resize((800, max(1, int(small.height * scale))))
    edges = small.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size
    px = edges.load()
    ink = [sum(px[x, y] for y in range(0, h, 2)) for x in range(w)]
    lo, hi = int(0.4 * w), int(0.6 * w)
    smooth = [statistics.mean(ink[max(lo, x - 4):min(hi, x + 5)]) for x in range(lo, hi)]
    best = lo + min(range(len(smooth)), key=smooth.__getitem__)
    return int(best / scale)


def spread_halves(width: int, height: int, fold: int, overlap: float = 0.02):
    """(left box, right box) for a spread, each reaching a little past the fold."""
    pad = int(overlap * width)
    return (0, 0, min(width, fold + pad), height), (max(0, fold - pad), 0, width, height)


def split_survey_label(text: str):
    """Split a spread's label into (left page, right page) at the right page's number.

    The left page's text starts with its page number on its own line (13); the
    right page starts where the next number (14) is on a line by itself. That's
    true for 146 of the 288 survey training labels; for the rest this returns
    None and the spread is trained on whole.
    """
    lines = text.split("\n")
    first = lines[0].strip() if lines else ""
    if not re.fullmatch(r"\d{1,3}", first):
        return None
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == str(int(first) + 1):
            return "\n".join(lines[:i]).strip(), "\n".join(lines[i:]).strip()
    return None


# ---------------------------------------------------------------------------
# Label clean-up for training targets
# ---------------------------------------------------------------------------
def training_text(text: str, category: str) -> str:
    """The label as the model should learn to write it.

    Kade postcards: the 'Postmark ...' line is on the back of the card, so it
    isn't in the image; training on it would teach the model to invent dates.
    """
    if category == "kade_letters":
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("Postmark"))
    return text.strip()


# ---------------------------------------------------------------------------
# Output clean-up (from churro_simple.py; every rule was checked against the
# 619 references in train.csv)
# ---------------------------------------------------------------------------
_PLAIN = str.maketrans({"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "„": '"', "…": "...", "|": " "})
_FRACTIONS = {"1/2": "½", "1/4": "¼", "3/4": "¾"}


def clean_output(text: str, category: str = "", max_chars: int = 3000) -> str:
    text = text.translate(_PLAIN)
    text = re.sub(r"```[a-z]*", "", text)                  # markdown code fences
    text = text.replace("**", "").replace("__", "")        # markdown bold
    text = re.sub(r"^\s*(transcription|text)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\[[^\]\n]*\]?", "", text)               # notes like [illegible]: no label has brackets
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if re.fullmatch(r"\d{1,3}\s?x\s?\d{1,3}(\.\d+)?", line):   # archive shelf numbers like 63x96.3
            continue
        if lines and line and line == lines[-1]:                   # a line repeated back to back
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"(.{8,}?)(?:\s*\1){2,}", r"\1", text)         # a phrase looping inside a line
    if category in ("survey_notes", "dominy_accounts"):          # these labels write ½ ¼ ¾
        for slash, glyph in _FRACTIONS.items():
            text = re.sub(rf"(\d)\s+{slash}(?![\d/])", rf"\1{glyph}", text)
            text = re.sub(rf"(?<![\d/]){slash}(?![\d/])", glyph, text)
    return text[:max_chars].strip()


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def load_image(path, box=None, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Open a page (or one half of a spread), decoded at about the size the model sees."""
    img = Image.open(path)
    if box is None:
        side = int((max_pixels * 2) ** 0.5)
        img.draft("RGB", (side, side))   # lets the JPEG decoder skip detail we'd throw away
    img = img.convert("RGB")
    return img.crop(tuple(box)) if box is not None else img


def augment(img: Image.Image, rng) -> Image.Image:
    """A light random change to a training image, so 27 Dominy pages go further:
    a small tilt, a little brightness and contrast change, sometimes a slight blur."""
    from PIL import ImageEnhance
    img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=False, fillcolor=(235, 228, 210))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.25))
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return img
