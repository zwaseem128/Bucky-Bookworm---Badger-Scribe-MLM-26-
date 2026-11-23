"""Cross-check: two (or more) models judge each other's readings of the same pages.

Run by 3_predict_test.ipynb (Step 5), usually as a job:

    python cross_check.py --data_dir <data> --out_dir <folder> --names churro,olmocr \
        --run1 <churro run folder> --run2 <olmOCR run folder>

<folder>/candidates_<name>.csv holds each model's readings (page_id, text): its test pages and,
to measure the cross-check, its held-out dev pages. Every model then scores every candidate
reading of a page: it is shown the page image and the candidate text, and gives the mean
log-probability of the candidate's tokens, the closing end-of-answer mark included (so a
reading that stops early is marked down too). No text is generated, so this is quick.

A model's own sureness was a poor judge on 18 Nov: a degenerate reading of dominy_034_p002 turned
180 degrees was surer (-0.017) than the right one (-0.204). A second model doesn't share the first
one's habits, so a reading that both models find likely is more often the right one. For each page
the candidate with the best mean score over all the models is kept; the first name in --names
(the current best model) wins a tie or a near-tie within --margin. A candidate stuck repeating
itself is never kept while another candidate isn't.

Writes, in --out_dir:
    cross_scores.csv  every (page, candidate, judge): the score, its tokens, the best rotation
    picked.csv        page_id, text, picked: the kept reading of every page
    cross_report.json for pages with a label in train.csv (the held-out dev pages): the CER of
                      each candidate alone, of the cross-check, and of the best possible pick
Short pages of collections in a run's "rotations" setting are scored turned 0/90/180/270 degrees
and the best of the four counts: the candidate may have come from a turned reading.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DENSE_CHARS, build_prompt, catalog_context, category_of, full_messages,  # noqa: E402
                    ink_chars, load_image, looks_wrong, training_text, user_messages)
from finetune_churro import listed, load_model, page_cer_function, parse_box  # noqa: E402

MAX_ANSWER_TOKENS = 3000   # a candidate longer than this is scored on its first 3000 tokens


def read_candidates(out_dir, names):
    cands = {}
    for name in names:
        path = os.path.join(out_dir, f"candidates_{name}.csv")
        df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
        col = "text" if "text" in df.columns else "prediction"
        cands[name] = dict(zip(df.page_id, df[col].astype(str)))
    return cands


@torch.no_grad()
def answer_logprob(model, processor, image, prompt, answer):
    """(mean log-probability of answer's tokens and the end mark after it, how many tokens)."""
    proc = processor
    proc.tokenizer.padding_side = "right"
    full = proc.apply_chat_template(full_messages(prompt, answer), tokenize=False)
    head = proc.apply_chat_template(user_messages(prompt), tokenize=False, add_generation_prompt=True)
    enc = proc(text=[full], images=[image], return_tensors="pt").to(model.device)
    n_head = proc(text=[head], images=[image], return_tensors="pt")["input_ids"].shape[1]
    ids = enc["input_ids"][0]
    # The answer's tokens and the end-of-answer mark right after it; not the template's last newline.
    end_mark = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    stop = next((i for i in range(n_head, len(ids)) if int(ids[i]) == end_mark), len(ids) - 1) + 1
    stop = min(stop, n_head + MAX_ANSWER_TOKENS)
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = lm.model(**enc, use_cache=False).last_hidden_state[0]
    # position t predicts token t + 1: the answer's tokens n_head..stop-1 come from n_head-1..stop-2
    logits = lm.lm_head(hidden[n_head - 1:stop - 1]).float()
    lp = torch.log_softmax(logits, dim=-1).gather(1, ids[n_head:stop].unsqueeze(1)).squeeze(1)
    return float(lp.mean()), int(stop - n_head)


def judge(run_dir, data_dir, pages, cands, parts, model_id=None):
    """[(page_id, candidate name, judge, score, tokens, rotation)] for one run's model."""
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    opts = summary["reading"]
    adapter = run_dir if os.path.exists(os.path.join(run_dir, "adapter_config.json")) \
        else summary.get("adapter_dir") or None
    who = os.path.basename(os.path.normpath(run_dir))
    model, processor, _ = load_model(model_id or summary["model_id"], opts["max_pixels"], adapter)
    doc_ctx = catalog_context(data_dir, opts.get("rich_context", ""))
    rows, start = [], time.time()
    for n, pid in enumerate(pages, 1):
        cat = category_of(pid)
        own = parts[parts.page_id == pid].to_dict("records")
        if len(own) != 1:   # a survey spread, read as two halves: its text can't be scored against one image
            continue
        path, box = os.path.join(data_dir, own[0]["image"]), parse_box(own[0]["box"])
        turned = listed(opts.get("rotations", ""), cat)
        prompt = build_prompt(cat, doc_ctx.get(pid.rsplit("_", 1)[0], ""), sideways=turned)
        degrees = (0, 90, 180, 270) if turned and ink_chars(path, box) < DENSE_CHARS else (0,)
        images = {d: load_image(path, box, opts["max_pixels"], bool(opts["upscale"]), listed(opts["enhance"], cat),
                                rotate=d) for d in degrees}
        seen = {}
        for name, by_page in cands.items():
            text = training_text(by_page.get(pid, ""), cat)
            if not text:
                continue
            if text not in seen:   # two models that wrote the same thing: score it once
                tries = [(answer_logprob(model, processor, images[d], prompt, text), d) for d in degrees]
                (score, tokens), d = max(tries, key=lambda t: t[0][0])
                seen[text] = (score, tokens, d)
            score, tokens, d = seen[text]
            rows.append({"page_id": pid, "candidate": name, "judge": who, "score": round(score, 4),
                         "tokens": tokens, "rotation": d})
        if n % 10 == 0 or n == len(pages):
            print(f"[{who[:40]}] {n}/{len(pages)} pages judged ({(time.time() - start) / 60:.1f} min)", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def pick(scores, cands, names, margin):
    """{page_id: (kept text, name of the kept candidate)}."""
    kept = {}
    mean = scores.groupby(["page_id", "candidate"])["score"].mean() if len(scores) else pd.Series(dtype=float)
    for pid in sorted(set().union(*[set(c) for c in cands.values()])):
        options = [n for n in names if cands[n].get(pid, "").strip()]
        if not options:
            kept[pid] = ("", names[0])
            continue
        fine = [n for n in options if not looks_wrong(cands[n][pid], 0)] or options
        scored = [n for n in fine if (pid, n) in mean.index]
        if not scored:
            kept[pid] = (cands[fine[0]][pid], fine[0])
            continue
        best = max(scored, key=lambda n: mean[(pid, n)])
        first = scored[0]   # the first model named, unless it was ruled out
        if best != first and mean[(pid, best)] - mean[(pid, first)] <= margin:
            best = first
        kept[pid] = (cands[best][pid], best)
    return kept


def report(kept, cands, names, data_dir):
    """CER of each way of choosing, on the pages that have a label (the held-out dev pages)."""
    labels_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(labels_path):
        return {}
    labels = pd.read_csv(labels_path, keep_default_na=False, encoding="utf-8-sig")
    ref = dict(zip(labels.page_id, labels.text.astype(str)))
    pages = sorted(p for p in kept if p in ref)
    if not pages:
        return {}
    page_cer, _ = page_cer_function()
    cer = {n: {p: page_cer(cands[n].get(p, ""), ref[p]) for p in pages} for n in names}
    out = {"pages": len(pages)}
    for n in names:
        out[f"only {n}"] = round(sum(cer[n].values()) / len(pages), 4)
    out["cross-check"] = round(sum(cer[kept[p][1]][p] for p in pages) / len(pages), 4)
    out["best possible pick"] = round(sum(min(cer[n][p] for n in names) for p in pages) / len(pages), 4)
    out["by page"] = {p: {**{n: round(cer[n][p], 3) for n in names}, "picked": kept[p][1]} for p in pages}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True, help="holds candidates_<name>.csv; the results go here too")
    p.add_argument("--names", required=True, help="the candidates, best model first, e.g. churro,olmocr")
    p.add_argument("--run1", required=True, help="a run folder whose model judges every candidate")
    p.add_argument("--run2", default="")
    p.add_argument("--run3", default="")
    p.add_argument("--margin", type=float, default=0.0,
                   help="another candidate must beat the first one's mean score by more than this")
    args = p.parse_args()

    names = [n.strip() for n in args.names.split(",") if n.strip()]
    cands = read_candidates(args.out_dir, names)
    pages = sorted(set().union(*[set(c) for c in cands.values()]))
    parts = pd.read_csv(os.path.join(args.data_dir, "parts.csv"), keep_default_na=False)
    scores_path = os.path.join(args.out_dir, "cross_scores.csv")
    done = pd.read_csv(scores_path, keep_default_na=False) if os.path.exists(scores_path) else pd.DataFrame()
    print(f"{len(pages)} pages, candidates from {', '.join(names)}", flush=True)
    for run in [r for r in (args.run1, args.run2, args.run3) if r]:
        who = os.path.basename(os.path.normpath(run))
        if len(done) and who in set(done.judge):   # a restarted job: this judge already finished
            print(f"{who}: already judged", flush=True)
            continue
        rows = judge(run, args.data_dir, pages, cands, parts)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(scores_path, index=False)

    kept = pick(done, cands, names, args.margin)
    pd.DataFrame([{"page_id": pid, "text": t, "picked": n} for pid, (t, n) in kept.items()]) \
        .to_csv(os.path.join(args.out_dir, "picked.csv"), index=False)
    counts = pd.Series([n for _, n in kept.values()]).value_counts().to_dict()
    print("kept: " + ", ".join(f"{n} {c}" for n, c in counts.items()), flush=True)
    rep = report(kept, cands, names, args.data_dir)
    if rep:
        json.dump(rep, open(os.path.join(args.out_dir, "cross_report.json"), "w"), indent=2)
        print("on the " + str(rep["pages"]) + " labelled pages: " + ", ".join(
            f"{k} {v}" for k, v in rep.items() if k not in ("pages", "by page")), flush=True)


if __name__ == "__main__":
    main()
