## Zain Waseem - Badger Scribe (Bucky Bookworm) 

import os
from pathlib import Path

# Shared drive location (override by setting the BADGER_DRIVE environment variable)
DRIVE = Path(os.environ.get("BADGER_DRIVE", r"G:\Shared drives\Badger Scribe"))

# Check the drive exists before switching to it
if not DRIVE.exists():
    raise FileNotFoundError(f"Could not find {DRIVE}. Is the shared drive mounted?")

print("Before:", os.getcwd())
os.chdir(DRIVE)
print("After: ", os.getcwd())


# Step 1: Find and load in the Kaggle dataset files
INPUT_ROOT = Path(".")  # now points at the shared drive

for i, path in enumerate(sorted(INPUT_ROOT.rglob("*"))):
    if path.is_file():
        print(path)
        if i >= 4:
            break


# Step 2: Load the metadata and inspect the submission format
import pandas as pd

DATASET_DIR = Path(".")  # or DRIVE / "some_subfolder" if the CSVs live in a subfolder
train_df = pd.read_csv(DATASET_DIR / "train.csv")
test_df = pd.read_csv(DATASET_DIR / "test.csv")
sample_submission = pd.read_csv(DATASET_DIR / "sample_submission.csv")

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

display(train_df.head())
display(test_df.head())
display(sample_submission.head())

print("Train columns:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())
print("Submission columns:", sample_submission.columns.tolist())

# Step 3: Understand where image IDs map to image files 

from pathlib import Path 

image_files = [ 
    p for p in DATASET_DIR.rglob("*")
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
]

print(f"Found {len(image_files)} image files")
print(*image_files[:10], sep="\n")


# Dummy Submission 

#Define a dummy model and cache 

from hashlib import sha256
from pathlib import Path
import json
from PIL import Image


CACHE_DIR = Path()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def transcribe(image: Image.Image, meta: dict) -> str:
    """Dummy adapter. Replace this later with a real model."""
    return ""


def cache_key(model_name: str, prompt: str, image_id: str) -> str:
    raw = f"{model_name}|{prompt}|{image_id}".encode("utf-8")
    return sha256(raw).hexdigest()


def cached_transcribe(model_name: str, image: Image.Image, meta: dict) -> str:
    image_id = str(meta["image_id"])
    prompt = meta.get("prompt", "Transcribe this document faithfully.")
    cache_path = CACHE_DIR / f"{cache_key(model_name, prompt, image_id)}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())["text"]

    text = transcribe(image, meta)
    cache_path.write_text(json.dumps({"text": text}))
    return text


# Creating a submission pipeline 

import pandas as pd

submission = sample_submission.copy()

# Prefer an ID column that exists in both test and sample submission.
candidate_id_columns = ["page_id", "doc_id", "category", "label_source", "text"]

id_col = next(
    (
        col for col in candidate_id_columns
        if col in test_df.columns and col in submission.columns
    ),
    None,
)

if id_col is None:
    raise ValueError(
        "Could not infer the ID column. "
        f"test columns={test_df.columns.tolist()}, "
        f"submission columns={submission.columns.tolist()}"
    )

prediction_columns = [col for col in submission.columns if col != id_col]

if len(prediction_columns) != 1:
    raise ValueError(
        "Expected exactly one prediction column. "
        f"Found: {prediction_columns}"
    )

prediction_col = prediction_columns[0]

# Critical: use the sample-submission row order.
submission[prediction_col] = ""

print("ID column:", id_col)
print("Prediction column:", prediction_col)
print("Submission shape:", submission.shape)

display(submission.head())


# Kaggle needs page_id, text for every test page in order of sample_submission.csv -> Code collect those files into a single file. 
import os
DRIVE = Path(os.environ.get("BADGER_DRIVE", r"G:\Shared drives\Badger Scribe"))
CHURRO_OUT = DRIVE / "output data"

def read_prediction(page_id):
    f = CHURRO_OUT / f"{page_id}.txt"
    return f.read_text(encoding="utf-8") if f.exists() else ""

submission = sample_submission[["page_id"]].copy()
submission["text"] = submission["page_id"].map(read_prediction)
missing = (submission["text"] == "").sum()
print(f"{len(submission) - missing} of {len(submission)} test pages have a transcription; {missing} will score 1.0")


#Validate the submission dataset before submitting 

from pathlib import Path

OUTPUT_PATH = Path("submission.csv")

submission.to_csv(OUTPUT_PATH, index=False)

print(f"Wrote: {OUTPUT_PATH.resolve()}")
print("Rows:", len(submission))
print("Columns:", submission.columns.tolist())

assert len(submission) == len(sample_submission), "Wrong number of rows"
assert submission.columns.tolist() == sample_submission.columns.tolist(), "Wrong columns"
assert submission[id_col].equals(sample_submission[id_col]), "IDs/order do not match sample submission"

print("Success: Submission has the expected schema and ID ordering.")