"""What a run will cost, worked out before it starts.

A run's bill is its GPU hours times the instance's hourly price. The hours are four kinds of
work, and each is estimated from what the finished runs in churro-work/runs actually took:

- starting the machine and downloading the model (a fixed few minutes),
- training: images per epoch x epochs x seconds per image,
- reading the held-out pages before training and after every epoch (the curve),
- reading them once more at the end, plus a few training pages (the overfitting check).

Seconds per image and minutes per page come from each finished run's summary.json and the
"[dev ...] N pages in M min" lines of its train.log, so the estimate gets better as runs finish.
Where no finished run matches (a new model, or train_vision the other way), the nearest one is
scaled, and the printout says so: treat those figures as rough.

Notebook 2, Step 4b calls estimate(); bills() asks SageMaker what the jobs so far were billed.
"""

import glob
import json
import math
import os
import re
import statistics
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, split_survey_label, training_text  # noqa: E402

# Dollars per hour, on demand, US East (Ohio), from the AWS price list dated 1 Nov 2026
# (pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-2/index.csv).
# Training jobs and notebook instances are priced separately.
PRICES_FROM = "AWS price list for US East (Ohio), 1 Nov 2026"
JOB_PRICES = {"ml.g4dn.xlarge": 0.736, "ml.g5.xlarge": 1.2575, "ml.g6.xlarge": 1.127,
              "ml.g6e.xlarge": 2.61, "ml.g6e.2xlarge": 2.80, "ml.g6e.12xlarge": 13.12}
NOTEBOOK_PRICES = {"ml.t3.medium": 0.05, "ml.g4dn.xlarge": 0.736, "ml.g5.xlarge": 1.408, "ml.g6.xlarge": 1.127,
                   "ml.g6e.xlarge": 2.6054, "ml.g6e.2xlarge": 2.8026}

# How fast each GPU is next to the A10G of ml.g5.xlarge, which the team's runs used, and how much
# memory it has. The speeds are rough (T4, A10G, L4, L40S); the finished runs don't record the
# instance they ran on, so they are all taken as ml.g5.xlarge.
# ml.g6e.12xlarge has four L40S: a model spread over them fits 192 GB but runs no faster than on one.
GPU_SPEED = {"ml.g4dn.xlarge": 0.4, "ml.g5.xlarge": 1.0, "ml.g6.xlarge": 0.8, "ml.g6e.xlarge": 1.8, "ml.g6e.2xlarge": 1.8,
             "ml.g6e.12xlarge": 1.8}
GPU_GB = {"ml.g4dn.xlarge": 16, "ml.g5.xlarge": 24, "ml.g6.xlarge": 24, "ml.g6e.xlarge": 48, "ml.g6e.2xlarge": 48,
          "ml.g6e.12xlarge": 192}

# Model sizes in billions of parameters, to scale a model with no finished run from one that has.
PARAMS = {"churro": 3.8, "qwen3-vl-4b": 4.4, "qwen3-vl-8b": 8.8, "olmocr-2-7b": 8.3, "qwen3.8-27b": 27.8}

# Used only when churro-work/runs has no finished run to measure: CHURRO on ml.g5.xlarge, from the
# team's runs up to 24 Nov 2026. Training the image side too (train_vision 1) took 6.5 s (Kade)
# and 11.8 s (survey) per image, against 1.6 to 1.7 s without it.
FALLBACK = {"sec_per_image": {0: 1.7, 1: 9.0}, "trained_min_per_page": 0.3, "untrained_min_per_page": 1.5}
VISION_SLOWDOWN = 5.0     # train_vision 1 against 0, when only the other kind has been measured
FOUR_BIT_SLOWDOWN = 1.4   # a model too big for the GPU in bf16 would have to load in 4-bit

PASS = re.compile(r"\[dev ([^\]]+)\] (\d+) pages in ([\d.]+) min")


def _short(model_id):
    """'stanford-oval/churro-3B' and 'churro' are the same model in summary.json files of different dates."""
    for short, full in MODELS.items():
        if model_id in (short, full):
            return short
    return model_id or "churro"


def _listed(setting, category):
    names = {s.strip() for s in str(setting).split(",") if s.strip()}
    return "all" in names or category in names


def past_runs(runs_dir):
    """One row per finished run: its training seconds per image and reading minutes per page."""
    rows = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        folder = os.path.dirname(path)
        try:
            s = json.load(open(path))
            log = open(os.path.join(folder, "train.log"), errors="replace").read()
        except (OSError, ValueError):
            continue
        passes = [(label, int(n), float(m)) for label, n, m in PASS.findall(log)]
        epochs, images = float(s.get("epochs") or 0), int(s.get("train_images_per_epoch") or 0)
        row = {"run": os.path.basename(folder), "model": _short(s.get("model_id", "")),
               "vision": int(s.get("train_vision") or 0), "categories": s.get("categories", "all"),
               "epochs": epochs, "images": images, "train_minutes": float(s.get("train_minutes") or 0)}
        # train_minutes includes the reading after every epoch; take those passes out. A run that
        # was started again repeats its log lines, so only the last ones belong to the training.
        after_epoch = [p for p in passes if p[0].startswith("after epoch")]
        after_epoch = after_epoch[-math.ceil(epochs):] if epochs > 0 else []
        pure = row["train_minutes"] - sum(m for _, _, m in after_epoch)
        if epochs > 0 and images and pure > 0:
            row["sec_per_image"] = pure * 60 / (images * epochs)
        # "Untrained" readings are the plain model's; they take longer because it runs on and
        # gets read a second time (CHURRO untrained on survey notes: 2.4 minutes a page, 0.3 trained).
        plain = not (s.get("init_adapter") or s.get("adapter_dir"))
        untrained = [p for p in passes if p[0] in ("before training", "baseline (no training)") and plain][-1:]
        trained = after_epoch + [p for p in passes if p[0].startswith(("after training", "training pages"))][-2:]
        if not plain:
            trained += [p for p in passes if p[0] in ("before training", "baseline (no training)")][-1:]
        for name, group in (("untrained_min_per_page", untrained), ("trained_min_per_page", trained)):
            pages = sum(n for _, n, _ in group)
            if pages:
                row[name] = sum(m for _, _, m in group) / pages
        row["script_minutes"] = round(pure + sum(m for _, _, m in passes), 1) if pure > 0 else round(sum(m for _, _, m in passes), 1)
        rows.append(row)
    return rows


def _spread(values, wide=1.0):
    """(low, likely, high) of some measurements; one measurement alone gets -25% / +35%."""
    values = sorted(values)
    likely = statistics.median(values)
    low, high = (likely * 0.75, likely * 1.35) if len(values) == 1 else (values[0], values[-1])
    return low / wide, likely, high * wide


def _measured(past, key, model, categories, vision=None):
    """Measurements of `key` on finished runs of this model, the same collections first."""
    rows = [r for r in past if r["model"] == model and key in r and (vision is None or r["vision"] == vision)]
    same = [r for r in rows if r["categories"] == categories]
    return same or rows


def _speed(past, key, model, categories, vision=None):
    """(low, likely, high) and a line saying where the figure comes from."""
    rows = _measured(past, key, model, categories, vision)
    if rows:
        names = ", ".join(r["run"][:28] for r in rows[:3]) + (" ..." if len(rows) > 3 else "")
        return _spread([r[key] for r in rows]), f"measured on {len(rows)} finished run{'s' if len(rows) > 1 else ''} ({names})"
    if vision is not None and _measured(past, key, model, categories, 1 - vision):
        other = _spread([r[key] for r in _measured(past, key, model, categories, 1 - vision)], wide=1.3)
        factor = VISION_SLOWDOWN if vision else 1 / VISION_SLOWDOWN
        return tuple(v * factor for v in other), (f"no finished run of {model} with train_vision {vision}: scaled x{factor:g} "
                                                 f"from its runs with train_vision {1 - vision} (rough)")
    known = [m for m in PARAMS if m != model and any(r["model"] == m and key in r for r in past)]
    if known and model in PARAMS:
        # Scale from a model measured on these collections if there is one, and of those from the
        # one nearest in size. (Not simply from CHURRO: untrained it runs on, and reads a Dominy
        # page two to four times more slowly than Qwen3-VL-8B, a model over twice its size.)
        same = [m for m in known if any(r["model"] == m and key in r and r["categories"] == categories for r in past)]
        near = min(same or known, key=lambda m: abs(math.log(PARAMS[model] / PARAMS[m])))
        base, _ = _speed(past, key, near, categories, vision)
        factor = PARAMS[model] / PARAMS[near]
        return (base[0] * factor / 1.3, base[1] * factor, base[2] * factor * 1.3), (
            f"no finished run of {model} here: scaled by model size (x{factor:.1f}) from {near}'s runs (a rough guess)")
    fallback = FALLBACK[key][vision] if vision is not None else FALLBACK[key]
    factor = PARAMS.get(model, PARAMS["churro"]) / PARAMS["churro"]
    return _spread([fallback * factor], wide=1.3), "no finished runs to measure: CHURRO's figures on ml.g5.xlarge up to 24 Nov (rough)"


def plan(settings, data_dir):
    """How many images a run trains on per epoch and how many pages each reading pass covers:
    the same choices finetune_churro.py makes, without loading the model."""
    need = [n for n in ("train_items.csv", "dev_pages.csv", "parts.csv") if not os.path.exists(os.path.join(data_dir, n))]
    if need:
        raise SystemExit(f"Missing {', '.join(need)} in {data_dir}. Run Notebook 1 first.")
    get = lambda k, default: settings.get(k, default)   # noqa: E731  (the script's own defaults)
    items = pd.read_csv(os.path.join(data_dir, "train_items.csv"), keep_default_na=False)
    dev = pd.read_csv(os.path.join(data_dir, "dev_pages.csv"), keep_default_na=False)
    parts = pd.read_csv(os.path.join(data_dir, "parts.csv"), keep_default_na=False)
    n_parts = parts.page_id.value_counts()

    def images_of(pages):   # a survey spread whose label splits at the fold is two images
        return sum(2 if n_parts.get(r["page_id"], 0) == 2 and split_survey_label(training_text(r["text"], r["category"]))
                   else 1 for r in pages.to_dict("records") if n_parts.get(r["page_id"], 0))

    held, train_pages = get("holdout_docs", ""), None
    if held or get("train_score_pages", 0):
        train_pages = pd.read_csv(os.path.join(data_dir, "train.csv"), keep_default_na=False, encoding="utf-8-sig")
        if "category" not in train_pages.columns:
            from common import category_of
            train_pages["category"] = train_pages["page_id"].map(category_of)
        train_pages = train_pages[train_pages.page_id.isin(set(parts.page_id))]
    extra_by_cat = {}
    if held:
        held = {d.strip() for d in held.split(",") if d.strip()}
        items = items[~items.doc_id.isin(held)]
        held_pages = train_pages[train_pages.doc_id.isin(held)]
        freed = dev[dev.category.isin(set(held_pages.category)) & ~dev.doc_id.isin(held)]
        for c, g in freed.groupby("category"):
            extra_by_cat[c] = extra_by_cat.get(c, 0) + images_of(g)
        dev = pd.concat([dev[~dev.category.isin(set(held_pages.category))][["page_id", "doc_id", "category"]],
                         held_pages[["page_id", "doc_id", "category"]]], ignore_index=True)
    categories = get("categories", "all")
    if categories != "all":
        items = items[items.category.map(lambda c: _listed(categories, c)).astype(bool)]
        dev = dev[dev.category.map(lambda c: _listed(categories, c)).astype(bool)]
        extra_by_cat = {c: n for c, n in extra_by_cat.items() if _listed(categories, c)}
    if get("train_on_dev", 0):
        for c, g in dev.groupby("category"):
            extra_by_cat[c] = extra_by_cat.get(c, 0) + images_of(g)
    by_cat = items.category.value_counts().to_dict()
    for c, n in extra_by_cat.items():
        by_cat[c] = by_cat.get(c, 0) + n
    trained_pages = set(items.page_id)
    if get("max_train_items", 0):
        by_cat = {c: min(n, max(1, int(get("max_train_items", 0)) // 3)) for c, n in by_cat.items()}
    for part in [s for s in str(get("oversample", "dominy_accounts:4")).split(",") if ":" in s]:
        cat, times = part.split(":")
        if cat.strip() in by_cat:
            by_cat[cat.strip()] *= int(times)
    dev_by_cat = dev.category.value_counts()
    if get("dev_pages", 0):
        dev_by_cat = dev_by_cat.clip(upper=max(1, int(get("dev_pages", 0)) // 3))
    if get("dev_sample", 0):
        dev_by_cat = dev_by_cat.clip(upper=int(get("dev_sample", 0)))
    curve = int(dev_by_cat.clip(upper=int(get("curve_pages", 3))).sum()) if get("curve_pages", 3) else 0
    check = 0
    if get("train_score_pages", 0) and train_pages is not None:
        seen = train_pages[train_pages.page_id.isin(trained_pages)]
        check = int(seen.category.value_counts().clip(upper=int(get("train_score_pages", 0))).sum())
    return {"images_per_epoch": int(sum(by_cat.values())), "images_by_collection": by_cat,
            "dev_pages": int(dev_by_cat.sum()), "curve_pages": curve, "train_check_pages": check}


def estimate(settings, data_dir, runs_dir, instance="ml.g5.xlarge", max_hours=None, spot=False, where="job",
             run_name="", model=None, show=True):
    """Print and return what this run should take and cost.

    settings: Notebook 2's `settings`. model: another model's short name, to ask what the same
    run would cost on it (e.g. "qwen3.8-27b"); instance: the GPU it would run on."""
    prices = JOB_PRICES if where == "job" else NOTEBOOK_PRICES
    if instance not in prices:
        raise SystemExit(f"No price saved for {instance}. Add it to {'JOB' if where == 'job' else 'NOTEBOOK'}_PRICES in scripts/cost.py.")
    price, model = prices[instance], model or _short(settings.get("model_id", "churro"))
    todo, past = plan(settings, data_dir), past_runs(runs_dir)
    epochs, vision = float(settings.get("epochs", 2.0)), int(settings.get("train_vision", 0) or 0)
    categories = settings.get("categories", "all")
    warm = bool(settings.get("init_adapter") or settings.get("adapter_dir"))   # reads like a trained model from the start

    notes = []
    slow = 1 / GPU_SPEED.get(instance, 1.0)
    weights_gb = PARAMS.get(model, PARAMS["churro"]) * 2          # bf16: two bytes per parameter
    if weights_gb + 4 > GPU_GB.get(instance, 24):
        slow *= FOUR_BIT_SLOWDOWN
        notes.append(f"{model} is about {weights_gb:.0f} GB in bf16, more than this GPU's {GPU_GB.get(instance, 24)} GB: it would have "
                     f"to load in 4-bit, which these scripts don't do yet (figures include x{FOUR_BIT_SLOWDOWN} for that).")
    train, train_from = _speed(past, "sec_per_image", model, categories, vision)
    read, read_from = _speed(past, "trained_min_per_page", model, categories)
    first, first_from = (read, read_from) if warm else _speed(past, "untrained_min_per_page", model, categories)

    start_up = 8 + weights_gb / 3   # the machine, the container, then the model at about 3 GB a minute
    steps = [("Starting up, downloading the model", "", (start_up * 0.7, start_up, start_up * 1.5), "a fixed allowance")]
    if epochs > 0:
        if todo["curve_pages"]:
            steps.append(("Reading before training", f"{todo['curve_pages']} pages",
                          tuple(v * todo["curve_pages"] * slow for v in first), first_from))
        n = todo["images_per_epoch"] * epochs
        steps.append(("Training", f"{todo['images_per_epoch']} images x {epochs:g} epochs",
                      tuple(v * n / 60 * slow for v in train), train_from))
        if todo["curve_pages"]:
            steps.append(("Reading after every epoch", f"{todo['curve_pages']} pages x {math.ceil(epochs)}",
                          tuple(v * todo["curve_pages"] * math.ceil(epochs) * slow for v in read), read_from))
    last = todo["dev_pages"] + todo["train_check_pages"]
    final = read if (epochs > 0 or warm) else first
    steps.append(("Final reading" + (" and training-page check" if todo["train_check_pages"] else ""), f"{last} pages",
                  tuple(v * last * slow for v in final), read_from if (epochs > 0 or warm) else first_from))

    low, likely, high = (sum(s[2][i] for s in steps) for i in range(3))
    result = {"run": run_name, "model": model, "instance": instance, "price_per_hour": price, "plan": todo,
              "hours": round(likely / 60, 2), "hours_low": round(low / 60, 2), "hours_high": round(high / 60, 2),
              "cost": round(likely / 60 * price, 2), "cost_low": round(low / 60 * price, 2),
              "cost_high": round(high / 60 * price, 2),
              "steps": [{"step": s[0], "work": s[1], "minutes": round(s[2][1], 1), "cost": round(s[2][1] / 60 * price, 2),
                         "from": s[3]} for s in steps]}
    if max_hours:
        result["cap_cost"] = round(max_hours * price, 2)
        if high / 60 > max_hours:
            notes.append(f"JOB_HOURS = {max_hours:g} could stop this run before it ends (up to {high / 60:.1f} h): raise it. "
                         "A stopped run carries on from its last epoch when started again.")
    if epochs > 0 and todo["curve_pages"] and not warm and steps[1][2][1] > 0.15 * likely:
        notes.append(f"Reading before training is {steps[1][2][1] / likely:.0%} of the bill, and it only re-measures the plain "
                     "model, which a finished run or a \"-baseline\" run has already scored on these pages.")
    if epochs > 0 and run_name and os.path.exists(os.path.join(runs_dir, run_name, "summary.json")):
        notes.append("This run has already finished, so Step 5 will refuse to start it again (its results are kept). "
                     "The estimate is for the same settings run afresh.")
    if spot:
        notes.append("SPOT = True: the bill is usually well under the on-demand figures here; the discount changes by the hour.")
    if where != "job":
        notes.append("WHERE = \"here\": the notebook instance is billed for every hour it is switched on, not only for this run.")
    result["notes"] = notes
    if show:
        print(f"Cost estimate{' for ' + run_name if run_name else ''}")
        print(f"  {model} on {instance}, ${price:.2f} an hour ({'training job' if where == 'job' else 'notebook instance'}, "
              f"on demand; {PRICES_FROM})\n")
        table = pd.DataFrame([{"step": s["step"], "work": s["work"], "minutes": f"{s['minutes']:.0f}", "cost": f"${s['cost']:.2f}"}
                              for s in result["steps"]])
        print(table.to_string(index=False))
        print(f"\n  Likely: {result['hours']:.1f} hours, ${result['cost']:.2f}    "
              f"(between {result['hours_low']:.1f} and {result['hours_high']:.1f} hours, "
              f"${result['cost_low']:.2f} to ${result['cost_high']:.2f})")
        if max_hours:
            print(f"  At most: ${result['cap_cost']:.2f}, since JOB_HOURS stops the job after {max_hours:g} hours.")
        print("\n  Where the figures come from:")
        for label, line in (("training", train_from), ("reading, trained model", read_from),
                            ("reading, plain model", first_from)):
            if label != "training" or epochs > 0:
                print(f"    {label}: {line}")
        for note in notes:
            print(f"\n  Note: {note}")
    return result


def history(runs_dir, instance="ml.g5.xlarge"):
    """The finished runs in churro-work/runs: what each took, and what that costs at this instance's
    job price. Script time only (no start-up), and a run started more than once counts every time."""
    rows = [{"run": r["run"][:60], "model": r["model"], "vision": r["vision"], "images": r["images"], "epochs": r["epochs"],
             "sec/image": round(r["sec_per_image"], 1) if "sec_per_image" in r else "",
             "minutes": r["script_minutes"], "cost": round(r["script_minutes"] / 60 * JOB_PRICES[instance], 2)}
            for r in past_runs(runs_dir)]
    table = pd.DataFrame(rows)
    if len(table):
        print(table.to_string(index=False))
        print(f"\n{len(table)} runs, {table.minutes.sum() / 60:.1f} GPU hours, about ${table.cost.sum():.2f} at {instance}'s price.")
    return table


def bills(prefix="churro-", days=45):
    """What SageMaker billed for the training jobs whose names start with `prefix`: the real figure,
    start-up included. Needs permission to list training jobs, which the workshop role may not have."""
    import datetime

    import boto3
    client = boto3.client("sagemaker")
    since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    rows = []
    try:
        for page in client.get_paginator("list_training_jobs").paginate(NameContains=prefix, CreationTimeAfter=since):
            for job in page["TrainingJobSummaries"]:
                d = client.describe_training_job(TrainingJobName=job["TrainingJobName"])
                kind, seconds = d["ResourceConfig"]["InstanceType"], d.get("BillableTimeInSeconds") or 0
                rows.append({"job": job["TrainingJobName"], "started": str(job["CreationTime"])[:16],
                             "status": d["TrainingJobStatus"], "instance": kind, "billed minutes": round(seconds / 60),
                             "cost": round(seconds / 3600 * JOB_PRICES.get(kind, 0), 2)})
    except Exception as e:   # no permission, or no network
        print(f"SageMaker wouldn't list the jobs ({type(e).__name__}: {str(e)[:160]}). history() shows the runs saved here instead.")
        return pd.DataFrame(rows)
    table = pd.DataFrame(rows)
    if len(table):
        print(table.to_string(index=False))
        print(f"\n{len(table)} jobs in the last {days} days, {table['billed minutes'].sum() / 60:.1f} billed hours, "
              f"about ${table.cost.sum():.2f} ({PRICES_FROM}).")
    else:
        print(f"No training jobs named {prefix}* in the last {days} days.")
    return table
