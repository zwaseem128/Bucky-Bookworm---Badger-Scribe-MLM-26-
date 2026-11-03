"""Find where every line of writing is on each Kade page, with kraken.

Run by 1_prepare_lines.ipynb, with the Python of the separate kraken
environment (kraken needs a different PyTorch from the training code):

    python segment_lines.py --images <folder of kade_*.jpg> --out <folder>

This only *finds* the lines; it doesn't cut anything out. For each page it
writes <out>/<page_id>.json with every line's baseline (the line the writing
sits on) and kraken's outline around it, in reading order. make_crops.py then
cuts the images from these, in whichever way Notebook 1 asks for. Finding the
lines is the slow part, so it's done once, and the cutting can be changed
later in seconds.

A page that is already done is skipped, so stopping halfway and running again
picks up where it stopped.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

_model = None


def _init_worker(device):
    # One CPU thread per worker: several pages side by side is faster than
    # one page using every core.
    import torch
    torch.set_num_threads(1)
    global _model, _device
    from importlib import resources
    from kraken.lib import vgsl
    _model = vgsl.TorchVGSLModel.load_model(resources.files("kraken").joinpath("blla.mlmodel"))
    _device = device


def _segment(job):
    """Find the lines on one page; return (page_id, number of lines, seconds)."""
    image_path, out_dir = job
    from kraken import blla
    from PIL import Image

    start = time.time()
    page = Path(image_path).stem
    im = Image.open(image_path).convert("RGB")
    seg = blla.segment(im, model=_model, device=_device)
    lines = [{"baseline": [[int(x), int(y)] for x, y in l.baseline],
              "boundary": [[int(x), int(y)] for x, y in l.boundary]}
             for l in seg.lines if l.baseline and l.boundary]
    tmp = Path(out_dir) / f"{page}.json.partial"
    tmp.write_text(json.dumps({"page_id": page, "width": im.width, "height": im.height, "lines": lines}))
    tmp.rename(Path(out_dir) / f"{page}.json")   # only a finished page gets its real name
    return page, len(lines), time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with the page images")
    p.add_argument("--out", required=True, help="folder to write the line positions to")
    p.add_argument("--pattern", default="kade_*.jpg")
    p.add_argument("--limit", type=int, default=0, help="only the first N pages (0 = all)")
    p.add_argument("--workers", type=int, default=os.cpu_count(), help="pages processed side by side")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = sorted(Path(args.images).glob(args.pattern))
    if args.limit:
        pages = pages[:args.limit]
    todo = [(str(p), str(out)) for p in pages if not (out / f"{p.stem}.json").exists()]
    print(f"{len(pages)} pages, {len(pages) - len(todo)} already done, {len(todo)} to do "
          f"with {args.workers} workers", flush=True)
    if not todo:
        return

    start = time.time()
    with mp.get_context("spawn").Pool(args.workers, _init_worker, (args.device,)) as pool:
        for i, (page, n, secs) in enumerate(pool.imap_unordered(_segment, todo), 1):
            left = (time.time() - start) / i * (len(todo) - i)
            print(f"[{i}/{len(todo)}] {page}: {n} lines in {secs:.0f}s  (about {left / 60:.0f} min left)", flush=True)
    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
