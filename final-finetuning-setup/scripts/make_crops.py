"""Cut the Kade pages into the images the model reads, in one of three ways.

Run by 1_prepare_lines.ipynb, after segment_lines.py has found the lines:

    lines    one image per line of writing
    groups   one image per few lines (GROUP lines each), so fewer edges and more context
    pages    the whole page, as it is (doesn't need segment_lines.py at all)

For lines and groups, each image is a strip that follows the slope of the
writing. It reaches up toward the line above (ABOVE x the space between the
two lines) and down toward the line below (BELOW x that space), and always
includes kraken's own outline around the line's writing, so dots, umlauts,
the tall loops of l, h and ß and the tails of g, f and h stay in. Outside the
strip, where the neighbouring lines' writing is, the image is filled with the
colour of the paper, so each image shows only its own line(s).

Writes a fresh <out> folder:

    <page_id>/000.png, 001.png, ...   top to bottom
    crops.csv            page_id, crop, first_line, n_lines, left, top, right, bottom
                         (which lines of the page each image holds, and where on the page it is)
    crop_settings.json   how the images were cut, plus matching settings for training and reading
"""

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Settings that go with each way of cutting, read by the training and
# transcription scripts. A page is about 25 lines, so it needs far more words
# out (max_new_tokens) and more image detail (max_pixels) than one line.
MODE_SETTINGS = {
    "lines":  {"max_new_tokens": 128,  "max_pixels": 262_144,   "eval_batch": 16},
    "groups": {"max_new_tokens": 384,  "max_pixels": 524_288,   "eval_batch": 8},
    "pages":  {"max_new_tokens": 1536, "max_pixels": 1_310_720, "eval_batch": 2},
}


def _y_at(line, x):
    """Height of a line's baseline at position x (straight-line in between its points)."""
    pts = sorted(line["baseline"])
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1)
    return pts[-1][1]


def join_pieces(lines, width, height, tolerance=0.65, tiny=0.08):
    """Put back together the pieces of one written line that kraken found separately.

    Where a line has a gap in it (a detached capital, a wide space), kraken can
    return two or three short lines side by side, and sometimes a tiny stray
    piece (a loose curl or dot). Then the number of lines no longer matches the
    transcription. Here:
      - a piece joins the line to its left when it starts after that line ends,
        at about the same height (within `tolerance` x the usual space between
        lines; well under one full space, so separate lines never join);
      - a tiny piece (narrower than `tiny` x the page width) that sits within a
        line's width and close to its height is folded into that line.
    The result is sorted top to bottom.
    """
    if not lines:
        return []
    mids = sorted(statistics.mean(y for _, y in l["baseline"]) for l in lines)
    gaps = [b - a for a, b in zip(mids, mids[1:]) if b - a > 0.01 * height]
    typical = statistics.median(gaps) if gaps else 0.05 * height
    span = lambda l: (min(x for x, _ in l["baseline"]), max(x for x, _ in l["baseline"]))

    small = [l for l in lines if span(l)[1] - span(l)[0] < tiny * width]
    joined = []
    for piece in sorted((l for l in lines if l not in small), key=lambda l: span(l)[0]):
        x0 = span(piece)[0]
        y0 = _y_at(piece, x0)
        best, best_dy = None, None
        for line in joined:
            end = span(line)[1]
            dy = abs(_y_at(line, end) - y0)
            if end <= x0 + 15 and dy < tolerance * typical and (best is None or dy < best_dy):
                best, best_dy = line, dy
        if best is None:
            joined.append({"baseline": list(piece["baseline"]), "boundary": list(piece["boundary"])})
        else:
            best["baseline"] += piece["baseline"]
            best["boundary"] += piece["boundary"]
    for piece in small:
        x = statistics.mean(x for x, _ in piece["baseline"])
        y = statistics.mean(y for _, y in piece["baseline"])
        near = [(abs(_y_at(l, x) - y), l) for l in joined if span(l)[0] - 15 <= x <= span(l)[1] + 15]
        near = [(d, l) for d, l in near if d < tolerance * typical]
        if near:
            min(near, key=lambda t: t[0])[1]["boundary"] += piece["boundary"]   # widen that line's strip only
        else:
            joined.append({"baseline": list(piece["baseline"]), "boundary": list(piece["boundary"])})
    joined.sort(key=lambda l: statistics.mean(y for _, y in l["baseline"]))
    return joined


# Joining strengths to try, in order: (tolerance, tiny) for join_pieces, or
# None for kraken's pieces as they are. The first is the default, used for
# every page whose transcription we don't use (dev and test pages). For a
# training page, the first whose line count matches its transcription wins.
# On 26 real Kade pages: 13 match with the default alone, 20 with all of these.
JOINING = [(0.65, 0.08), None, (0.45, 0.08), (0.3, 0.04), (0.8, 0.12), (0.3, 0.08)]


def best_joining(pieces, width, height, wanted):
    """The pieces joined in the first way whose line count is `wanted` (or the default way)."""
    tries = [pieces if j is None else join_pieces(pieces, width, height, *j) for j in JOINING]
    if wanted:
        for lines in tries:
            if len(lines) == wanted:
                return lines
    return tries[0]


def line_gaps(lines):
    """For each line, the space to the line above and to the line below (in pixels)."""
    mid = [statistics.mean(y for _, y in l["baseline"]) for l in lines]
    gaps = [b - a for a, b in zip(mid, mid[1:]) if b > a]
    typical = statistics.median(gaps) if gaps else 50
    out = []
    for i in range(len(lines)):
        above = mid[i] - mid[i - 1] if i > 0 and mid[i] > mid[i - 1] else typical
        below = mid[i + 1] - mid[i] if i + 1 < len(lines) and mid[i + 1] > mid[i] else typical
        # A big gap (a blank line, a paragraph break) shouldn't make the strip huge.
        out.append((min(above, 1.5 * typical), min(below, 1.5 * typical)))
    return out


def strip(first, last, gap_above, gap_below, left, right, above, below, step=8):
    """The outline of a strip that follows the slope of the writing: from `above` x the
    space to the line above, over `first` line's baseline, down to `below` x the space
    to the next line, under `last` line's baseline."""
    xs = list(range(left, right, step)) + [right]
    top = [(x, _y_at(first, x) - above * gap_above) for x in xs]
    bottom = [(x, _y_at(last, x) + below * gap_below) for x in reversed(xs)]
    return top + bottom


def paper_colour(im):
    """The colour of the paper: the most common brightness band of the page."""
    small = im.resize((64, 64))
    data = small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata()
    pixels = sorted(data, key=sum)
    return pixels[int(len(pixels) * 0.7)]   # lighter than most ink, darker than the brightest specks


def line_columns(line, width, margin=10):
    xs = [x for x, _ in line["boundary"]] + [x for x, _ in line["baseline"]]
    return max(0, min(xs) - margin), min(width, max(xs) + margin + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="folder with the page images")
    p.add_argument("--segments", help="folder of <page_id>.json from segment_lines.py (not needed for pages)")
    p.add_argument("--out", required=True, help="folder to write the images to; replaced completely")
    p.add_argument("--mode", choices=sorted(MODE_SETTINGS), default="lines")
    p.add_argument("--group", type=int, default=3, help="lines per image, for groups")
    p.add_argument("--above", type=float, default=0.7, help="how far up toward the line above, 0 to 1")
    p.add_argument("--below", type=float, default=0.5, help="how far down toward the line below, 0 to 1")
    p.add_argument("--pattern", default="kade_*.jpg")
    p.add_argument("--labels", help="train.csv: lets training pages pick the joining that matches their transcription")
    p.add_argument("--dev_docs", default="kade_112", help="documents never cut using their transcription (comma-separated)")
    args = p.parse_args()

    wanted = {}
    if args.labels:
        import pandas as pd
        from kurrent_common import label_lines
        dev = set(args.dev_docs.split(","))
        for r in pd.read_csv(args.labels, keep_default_na=False, encoding="utf-8-sig").to_dict("records"):
            if r["doc_id"] not in dev:
                wanted[r["page_id"]] = len(label_lines(r["text"]))

    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)   # a fresh folder: never mix two ways of cutting
    out.mkdir(parents=True)
    rows, skipped = [], []
    for image_path in sorted(Path(args.images).glob(args.pattern)):
        page = image_path.stem
        im = Image.open(image_path).convert("RGB")
        if args.mode == "pages":
            (out / page).mkdir()
            im.save(out / page / "000.png")
            rows.append({"page_id": page, "crop": f"{page}/000.png", "first_line": 0, "n_lines": "",
                         "left": 0, "top": 0, "right": im.width, "bottom": im.height})
            continue
        seg_path = Path(args.segments) / f"{page}.json"
        if not seg_path.exists():
            skipped.append(page)
            continue
        paper = paper_colour(im)
        lines = best_joining(json.loads(seg_path.read_text())["lines"], im.width, im.height, wanted.get(page))
        (out / page).mkdir()
        gaps = line_gaps(lines)
        step = args.group if args.mode == "groups" else 1
        for n, first in enumerate(range(0, len(lines), step)):
            members = range(first, min(first + step, len(lines)))
            cols = [line_columns(lines[i], im.width) for i in members]
            left, right = min(c[0] for c in cols), max(c[1] for c in cols)
            # Keep: the strip that follows the slope of these lines, plus kraken's own
            # outline around their writing. Everything else becomes plain paper.
            keep = Image.new("L", im.size, 0)
            draw = ImageDraw.Draw(keep)
            draw.polygon(strip(lines[members[0]], lines[members[-1]], gaps[members[0]][0], gaps[members[-1]][1],
                               left, right, args.above, args.below), fill=255)
            for i in members:
                draw.polygon([tuple(pt) for pt in lines[i]["boundary"]], fill=255)
            box = keep.getbbox()
            if box is None:
                continue
            left, top, right, bottom = box
            crop = Image.composite(im.crop(box), Image.new("RGB", (right - left, bottom - top), paper),
                                   keep.crop(box))
            crop.save(out / page / f"{n:03d}.png")
            rows.append({"page_id": page, "crop": f"{page}/{n:03d}.png", "first_line": first, "n_lines": len(members),
                         "left": left, "top": top, "right": right, "bottom": bottom})

    with open(out / "crops.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["page_id", "crop", "first_line", "n_lines", "left", "top", "right", "bottom"])
        writer.writeheader()
        writer.writerows(rows)
    settings = {"mode": args.mode, "group": args.group if args.mode == "groups" else None,
                "above": args.above if args.mode != "pages" else None,
                "below": args.below if args.mode != "pages" else None,
                **MODE_SETTINGS[args.mode]}
    (out / "crop_settings.json").write_text(json.dumps(settings, indent=2))
    pages = len({r["page_id"] for r in rows})
    print(f"{len(rows)} images from {pages} pages, cut as '{args.mode}', in {out}")
    if skipped:
        print(f"WARNING: {len(skipped)} pages have no line positions yet (run Step 4 for all pages), "
              f"e.g. {', '.join(skipped[:3])}")


if __name__ == "__main__":
    main()
