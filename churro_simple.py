"""
Send images from the Badger Scribe shared drive to CHURRO-3b, one at a time.

Run it, type how many images you want, and each transcription is saved to
churro_output/<image>.txt in Downloads/Bucky Bookworm Images. Images that already have a
.txt file are skipped, so running it again picks up where you left off
(unless START_OVER is True, which starts again from the first image).

Needs campus wifi or the UW VPN to reach the gateway.
"""

import base64
import csv
import getpass
import os
import time
from pathlib import Path

from openai import OpenAI

IMAGES = Path(r"G:\Shared drives\Badger Scribe\images")
OUT = Path.home() / "Downloads" / "Bucky Bookworm Images" / "churro_output"
PROMPT = (
    "Transcribe all of the text in this historical document image exactly as "
    "written. Preserve original spelling, punctuation, capitalization, and line "
    "breaks. Do not correct, modernize, translate, or summarize. Output only "
    "the transcription."
)

# True: start from the first image in the folder and overwrite old results.
# False: skip images that already have a .txt file (pick up where you left off).
START_OVER = True

OUT.mkdir(parents=True, exist_ok=True)
all_images = sorted(IMAGES.glob("*.jpg"))
if START_OVER:
    todo = all_images
    print(f"{len(all_images)} images in the folder, starting from {all_images[0].name}.")
else:
    todo = [p for p in all_images if not (OUT / f"{p.stem}.txt").exists()]
    print(f"{len(all_images)} images in the folder, {len(todo)} not done yet.")

answer = input("How many images should I send to CHURRO? (a number, or 'all') ").strip()
while not (answer.isdigit() or answer.lower() == "all"):
    answer = input("Please type a number like 5, or 'all': ").strip()
if answer.isdigit():
    todo = todo[:int(answer)]

key = os.environ.get("CHURRO_API_KEY") or getpass.getpass("UW gateway API key: ")
client = OpenAI(base_url="https://llm-gw01.doit.wisc.edu/v1", api_key=key, timeout=900)

for i, path in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {path.name} ...", end=" ", flush=True)
    start = time.time()
    image_b64 = base64.b64encode(path.read_bytes()).decode()
    try:
        resp = client.chat.completions.create(
            model="churro-3b",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]}],
            temperature=0,
        )
    except Exception as e:
        print(f"failed: {e}")
        continue
    text = resp.choices[0].message.content or ""
    if not text.strip():
        # Don't save it, so the next run tries this page again.
        print("got an empty transcription, will retry next run")
        continue
    (OUT / f"{path.stem}.txt").write_text(text, encoding="utf-8")
    print(f"done in {time.time() - start:.0f}s")

# Gather every finished page into one CSV.
with open(OUT / "transcriptions.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "collection", "transcription"])
    for txt in sorted(OUT.glob("*.txt")):
        writer.writerow([txt.stem, txt.stem.split("_")[0],
                         txt.read_text(encoding="utf-8")])

print(f"Done. All results are in {OUT / 'transcriptions.csv'}")
