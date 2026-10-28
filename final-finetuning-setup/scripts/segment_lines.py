"""Cut every Kade page into one image per line of writing, with kraken.

Run by 1_prepare_lines.ipynb, with the Python of the separate kraken
environment (kraken needs a different PyTorch from the training code):

    python segment_lines.py --images <folder of kade_*.jpg> --out <folder>

For each page it writes <out>/<page_id>/000.png, 001.png, ... in reading order
(top to bottom). A page is first written to <page_id>.partial and renamed
when it is finished, so stopping halfway never leaves a half-done page, and
running again skips the pages that are already done.
"""

import argparse
import multiprocessing as mp
import os
import shutil
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
    """Segment one page; return (page_id, number of lines, seconds)."""
    image_path, out_dir = job
    from kraken import blla
    from kraken.lib.segmentation import extract_polygons
    from PIL import Image

    start = time.time()
    page = Path(image_path).stem
    im = Image.open(image_path).convert("RGB")
    seg = blla.segment(im, model=_model, device=_device)
    tmp = Path(out_dir) / f"{page}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    n = 0
    for n, (crop, _) in enumerate(extract_polygons(im, seg), 1):
        crop.save(tmp / f"{n - 1:03d}.png")
    tmp.rename(Path(out_dir) / page)
    return page, n, time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with the page images")
    p.add_argument("--out", required=True, help="folder to write the line crops to")
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
    todo = [(str(p), str(out)) for p in pages if not (out / p.stem).is_dir()]
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
