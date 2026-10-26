"""``notagen prefs``: build a CLaMP-DPO preference set (the in-process version of ``RL/data.py``).

Generated pieces are ranked by CLaMP 2 similarity to a target, then the top / bottom
``portion`` become chosen / rejected. Chosen pieces are filtered as in ``RL/data.py``:
misaligned bars, ungrouped staves of one instrument, and (with ground truth) CLaMP 2 similarity
> 0.95 to any ground-truth piece. The target is either

* ``--gt-features DIR``: mean of ground-truth CLaMP 2 features (the paper's setup), or
* ``--prompt "period|composer|instrumentation"``: a text description of the prompt (no data needed).

Writes the same ``{"chosen": [...], "rejected": [...]}`` JSON that ``notagen train dpo`` and
``RL/train.py`` read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .clamp2 import CLaMP2Encoder, cosine_matrix
from .metrics import alignment_ok, staves_grouped
from .reward import Prompt


def build(
    interleaved_dir,
    original_dir,
    out_path,
    enc: CLaMP2Encoder,
    portion=0.1,
    gt_features=None,
    prompt: Prompt | None = None,
) -> dict:
    files = sorted(Path(interleaved_dir).glob("*.abc"))
    texts = [f.read_text(encoding="utf-8") for f in files]
    feats = enc.music_features(texts)
    gt = None
    if gt_features:
        gt = np.stack([np.load(f).reshape(-1) for f in sorted(Path(gt_features).rglob("*.npy"))])
        target = gt.mean(0)
    elif prompt is not None:
        target = enc.text_features([prompt.text])[0]
    else:
        raise ValueError("need gt_features or prompt")
    sim = cosine_matrix(feats, target[None])[:, 0]
    order = np.argsort(-sim)
    k = int(len(files) * portion)
    copy_sim = cosine_matrix(feats, gt).max(1) if gt is not None else np.zeros(len(files))
    chosen, skipped = [], 0
    for i in order:
        if len(chosen) >= k:
            break
        lines = texts[i].splitlines(keepends=True)
        if alignment_ok(lines) and staves_grouped(lines) and copy_sim[i] <= 0.95:
            chosen.append(str(Path(original_dir) / files[i].name))
        else:
            skipped += 1
    rejected = [str(Path(original_dir) / files[i].name) for i in order[len(order) - k :]] if k else []
    prefs = {"chosen": chosen, "rejected": rejected}
    Path(out_path).write_text(json.dumps(prefs, indent=4))
    return {
        "n": len(files),
        "chosen": len(chosen),
        "rejected": len(rejected),
        "skipped": skipped,
        "sim_top": float(sim[order[0]]) if len(files) else None,
        "sim_bottom": float(sim[order[-1]]) if len(files) else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser("notagen prefs")
    ap.add_argument("--interleaved", required=True, help="output/interleaved/<run> (CLaMP 2 input)")
    ap.add_argument("--original", required=True, help="output/original/<run> (what DPO trains on)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--portion", type=float, default=0.1)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gt-features")
    g.add_argument("--prompt")
    ap.add_argument("--weights", default="weights/clamp2")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    enc = CLaMP2Encoder.load(a.weights, device=a.device)
    info = build(
        a.interleaved, a.original, a.out, enc, a.portion, a.gt_features, Prompt.parse(a.prompt) if a.prompt else None
    )
    print(json.dumps(info, indent=2))
