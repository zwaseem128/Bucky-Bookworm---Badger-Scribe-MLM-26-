"""Average several LoRA adapters of the same run (e.g. its epoch1 ... epoch4 folders) into one.

Epoch adapters of one run differ mostly in noise: Dominy's scored within 0.02 of each other.
Their average is usually steadier than any one of them, at no training cost.

The average is exact. A LoRA adapter changes each layer's weights by B @ A (times alpha / r).
Averaging the A's and the B's separately would not average those changes, so the adapters are
stacked instead: A = [A1; A2; ...] and B = [B1 B2 ...] / k give B @ A = mean of Bi @ Ai.
The result has rank k * r (four epochs of r 16 -> r 64, about 480 MB), with lora_alpha scaled
the same way so alpha / r, and with it every layer's change, is unchanged.

usage: python average_adapters.py <out folder> <adapter folder> <adapter folder> ...
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file


def average(folders, out):
    if len(folders) < 2:
        raise SystemExit("Give at least two adapter folders to average.")
    configs = [json.load(open(os.path.join(f, "adapter_config.json"))) for f in folders]
    first = configs[0]
    for f, c in zip(folders, configs):
        for k in ("r", "lora_alpha", "base_model_name_or_path", "target_modules", "use_rslora"):
            if c.get(k) != first.get(k):
                raise SystemExit(f"{f} differs from {folders[0]} in {k}: only adapters of the same run can be averaged.")
        if c.get("rank_pattern") or c.get("alpha_pattern"):
            raise SystemExit(f"{f} has per-layer ranks (rank_pattern / alpha_pattern), which this doesn't handle.")
    if first.get("use_rslora"):
        raise SystemExit("rsLoRA adapters scale by alpha / sqrt(r), so stacking them would change their strength.")
    weights = [load_file(os.path.join(f, "adapter_model.safetensors")) for f in folders]
    keys = set(weights[0])
    if any(set(w) != keys for w in weights):
        raise SystemExit("The adapters don't hold the same layers.")
    k = len(folders)
    merged = {}
    for name in sorted(keys):
        parts = [w[name] for w in weights]
        if ".lora_A." in name:
            merged[name] = torch.cat(parts, dim=0).contiguous()            # (k*r, in)
        elif ".lora_B." in name:
            merged[name] = (torch.cat(parts, dim=1) / k).contiguous()      # (out, k*r)
        else:                                                              # anything else: a plain mean
            merged[name] = (sum(p.float() for p in parts) / k).to(parts[0].dtype)
    os.makedirs(out, exist_ok=True)
    save_file(merged, os.path.join(out, "adapter_model.safetensors"), metadata={"format": "pt"})
    config = dict(first, r=first["r"] * k, lora_alpha=first["lora_alpha"] * k)
    json.dump(config, open(os.path.join(out, "adapter_config.json"), "w"), indent=2)
    json.dump({"averaged": [os.path.abspath(f) for f in folders]}, open(os.path.join(out, "averaged_from.json"), "w"),
              indent=2)
    for extra in ("README.md",):
        if os.path.exists(os.path.join(folders[0], extra)):
            shutil.copy(os.path.join(folders[0], extra), out)
    print(f"averaged {k} adapters (r {first['r']} -> {config['r']}) into {out}", flush=True)
    return out


if __name__ == "__main__":
    average(sys.argv[2:], sys.argv[1])
