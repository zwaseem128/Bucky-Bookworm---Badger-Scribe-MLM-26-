"""Shared helpers for fine-tuning the Qwen3-VL Kurrent adapter on Kade lines.

Used by finetune_qwen_kurrent_lora.py (the SageMaker training job) and by
prepare_kade_lines.py (run in the notebook), so the prompt, image handling and
label clean-up are the same everywhere. Keep this file in the same `code/`
folder as the training script: SageMaker uploads the whole folder.
"""

import re
import unicodedata

import pandas as pd
from PIL import Image

# The adapter we start from, and the model it was trained on.
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
START_ADAPTER = "dh-unibe/qwen3vl-german-xix-v2"

# The adapter was trained with exactly this prompt and no other text. Any
# change in wording is something it has never seen, so leave it alone.
PROMPT = "Transcribe the handwritten text in this image exactly as written."

# Qwen3-VL turns every 32x32 block of pixels into one image token. The
# processor resizes each line crop so its pixel count lands between these two
# numbers (the adapter's own settings). A Kade line crop is about 600 x 45 px
# = 27,000 pixels, so MIN_PIXELS upscales it to ~64 tokens. Raising MIN_PIXELS
# gives small, faint writing more detail at the cost of memory and time.
MIN_PIXELS = 65_536
MAX_PIXELS = 262_144


def label_lines(text: str) -> list:
    """The lines of a Kade label as training targets: blank lines dropped,
    indentation stripped (the metric ignores whitespace), and the
    'Postmark ...' line removed, because the postmark is on the other side of
    the card and isn't in the image. Training on it would teach the model to
    invent dates."""
    return [l.strip() for l in str(text).splitlines()
            if l.strip() and not l.strip().startswith("Postmark")]


def strip_postmark(text: str) -> str:
    """The label without its 'Postmark ...' line. Scoring against this gives
    the 'reading' CER: how well the model reads what is actually on the card,
    without the few points every page with a postmark loses for a line the
    model can't see."""
    return "\n".join(l for l in str(text).splitlines() if not l.strip().startswith("Postmark"))


def user_messages(prompt: str = PROMPT):
    """The user turn in the chat format Qwen-VL expects: an image placeholder
    (the processor swaps in the image tokens) followed by the prompt."""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


def full_messages(answer: str, prompt: str = PROMPT):
    """The user turn plus the answer we want: one complete training example."""
    return user_messages(prompt) + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]


def load_line_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# A copy of the competition's normalization and page CER, so a job can print
# a score without shipping metric.py. Official numbers come from metric.py.
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"</?(?:u|del|ins)>")


def normalize_text(text) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).replace("\\n", "\n")
    text = _TAG_RE.sub("", text)
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def page_cer(prediction: str, reference: str) -> float:
    """Edit distance / length of the reference, capped at 1.0. A `#` in the
    reference matches any character, as in the leaderboard metric."""
    pred, ref = normalize_text(prediction), normalize_text(reference)
    if not ref:
        return 0.0 if not pred else 1.0
    if not pred:
        return 1.0
    prev = list(range(len(ref) + 1))
    for i, cp in enumerate(pred, start=1):
        curr = [i]
        for j, cr in enumerate(ref, start=1):
            sub = 0 if (cr == "#" or cr == cp) else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + sub))
        prev = curr
    return min(1.0, prev[-1] / len(ref))
