"""``notagen eval``: generate for a prompt set and report structure + CLaMP 2 metrics as JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .clamp2 import CLaMP2Encoder
from .generation import Generator
from .infer import InferenceOptions, load
from .reward import Prompt, RewardWeights, Scorer, retrieval_accuracy

KEYBOARD_8 = [
    "Baroque|Bach, Johann Sebastian|Keyboard",
    "Baroque|Scarlatti, Domenico|Keyboard",
    "Classical|Mozart, Wolfgang Amadeus|Keyboard",
    "Classical|Beethoven, Ludwig van|Keyboard",
    "Romantic|Chopin, Frederic|Keyboard",
    "Romantic|Liszt, Franz|Keyboard",
    "Romantic|Debussy, Claude|Keyboard",
    "Romantic|Satie, Erik|Keyboard",
]


def load_reference(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """A folder of ``.npy`` CLaMP 2 features (``notagen clamp2 extract`` / legacy extractor)."""
    if not path:
        return None, None
    feats = np.stack([np.load(f).reshape(-1) for f in sorted(Path(path).rglob("*.npy"))])
    return feats.mean(0), feats


def summarize(scores_by_prompt: dict, feats_by_prompt: dict, prompts: list[Prompt], enc, ref_feats=None) -> dict:
    out = {"per_prompt": {}, "overall": {}}
    all_feats, all_labels, all_scores = [], [], []
    for li, p in enumerate(prompts):
        sc = scores_by_prompt[str(p)]
        f = feats_by_prompt[str(p)]
        all_scores += sc
        all_feats.append(f)
        all_labels += [li] * len(f)
        out["per_prompt"][str(p)] = _agg(sc)
    feats = np.concatenate(all_feats) if all_feats else np.zeros((0, 768))
    o = _agg(all_scores)
    o["retrieval_acc"] = retrieval_accuracy(feats, np.array(all_labels), [p.text for p in prompts], enc)
    o["retrieval_chance"] = 1 / len(prompts)
    if ref_feats is not None and len(feats):
        from .metrics import near_copy_rate

        o["near_copy_rate"] = near_copy_rate(ref_feats, feats)
    out["overall"] = o
    return out


def _agg(scores) -> dict:
    n = len(scores)

    def mean(key):
        v = [getattr(s, key) for s in scores if not np.isnan(getattr(s, key))]
        return float(np.mean(v)) if v else float("nan")

    return {
        "n": n,
        "finished": sum(s.ok for s in scores) / n,
        "restored": sum(s.restored for s in scores) / n,
        "aligned": sum(s.aligned for s in scores) / n,
        "grouped": sum(s.grouped for s in scores) / n,
        "text_sim": mean("text_sim"),
        "ref_sim": mean("ref_sim"),
        "diversity": 1 - mean("group_sim"),
        "reward": mean("reward"),
    }


def evaluate(
    opts: InferenceOptions,
    prompts: list[Prompt],
    n: int,
    clamp_dir: str,
    out_dir: str | None = None,
    reference: str | None = None,
    stream: bool = True,
    backend_and_tok=None,
) -> dict:
    backend, tok = backend_and_tok or load(opts)
    enc = CLaMP2Encoder.load(clamp_dir, device="auto", batch_size=16)
    centroid, ref_feats = load_reference(reference)
    w = RewardWeights(reference=4.0 if centroid is not None else 0.0)
    scorer = Scorer(enc, w, centroid)
    gen = Generator(backend, tok, stream=stream)
    scores, feats = {}, {}
    t0 = time.time()
    for p in prompts:
        pieces = list(gen.generate(p.lines, n, opts.batch_size))
        sc, f = scorer.score_group([x.text for x in pieces], [x.ok for x in pieces], p)
        scores[str(p)], feats[str(p)] = sc, f
        if out_dir:
            d = Path(out_dir) / str(p).replace("|", "_").replace(" ", "")
            d.mkdir(parents=True, exist_ok=True)
            for i, (x, s) in enumerate(zip(pieces, sc)):
                (d / f"{i}.abc").write_text(x.text)
                if s.interleaved:
                    (d / f"{i}.interleaved.abc").write_text(s.interleaved)
    report = summarize(scores, feats, prompts, enc, ref_feats)
    report["seconds"] = time.time() - t0
    report["weights"] = opts.weights
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser("notagen eval")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--clamp2", default="weights/clamp2", help="dir with the CLaMP 2 checkpoint + xlm-roberta-base/")
    ap.add_argument("--prompts", nargs="+", default=KEYBOARD_8, help='"period|composer|instrumentation" ...')
    ap.add_argument("--n", type=int, default=8, help="pieces per prompt")
    ap.add_argument("--reference", help="folder of ground-truth CLaMP 2 .npy features (paper's CLaMP 2 score)")
    ap.add_argument("--out", help="write pieces + report.json here")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.2)
    a = ap.parse_args(argv)
    opts = InferenceOptions(
        a.weights,
        a.backend,
        dtype=a.dtype,
        seed=a.seed,
        top_k=a.top_k,
        top_p=a.top_p,
        temperature=a.temperature,
        batch_size=a.batch_size,
    )
    report = evaluate(opts, [Prompt.parse(p) for p in a.prompts], a.n, a.clamp2, a.out, a.reference)
    print(json.dumps(report["overall"], indent=2))
