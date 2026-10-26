"""Evaluation metrics for generated pieces (roadmap Phase 0).

The structural checks are the filters ``RL/data.py`` applies to chosen DPO samples, turned into
reusable functions so they can be reported as rates and, later, used as RL rewards.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def alignment_ok(interleaved_lines: list[str]) -> bool:
    """Barlines, bar numbers and bar durations agree across voices (``RL/data.py`` check)."""
    from abctoolkit.check import check_alignment_unrotated
    from abctoolkit.rotate import unrotate_abc

    try:
        barline_equal, bar_no_equal, bar_dur_equal = check_alignment_unrotated(unrotate_abc(interleaved_lines))
    except Exception:
        return False
    return bool(barline_equal and bar_no_equal and bar_dur_equal)


def staves_grouped(abc_lines: list[str]) -> bool:
    """False when staves of one instrument are split by another instrument's staves."""
    appeared, last = set(), ""
    for line in abc_lines:
        if line.startswith("V:") and "nm=" in line:
            m = re.search(r'nm="([^"]+)"', line)
            if m:
                inst = m.group(1)
                if inst != last and inst in appeared:
                    return False
                last = inst
                appeared.add(inst)
    return True


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def mean_pairwise_distance(features: np.ndarray) -> float:
    """Diversity: mean ``1 - cosine`` over all pairs of CLaMP 2 features (``[n, d]``)."""
    f = np.asarray(features, dtype=np.float64).reshape(len(features), -1)
    if len(f) < 2:
        return float("nan")
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    sim = f @ f.T
    iu = np.triu_indices(len(f), 1)
    return float(1.0 - sim[iu].mean())


def clamp2_score(gt_features: np.ndarray, gen_features: np.ndarray) -> float:
    """Mean cosine similarity of generated features to the mean ground-truth feature."""
    centre = np.asarray(gt_features).reshape(len(gt_features), -1).mean(0)
    return float(np.mean([cosine(centre, g) for g in np.asarray(gen_features).reshape(len(gen_features), -1)]))


def near_copy_rate(gt_features: np.ndarray, gen_features: np.ndarray, threshold: float = 0.95) -> float:
    gt = np.asarray(gt_features, dtype=np.float64).reshape(len(gt_features), -1)
    gen = np.asarray(gen_features, dtype=np.float64).reshape(len(gen_features), -1)
    gt /= np.linalg.norm(gt, axis=1, keepdims=True)
    gen /= np.linalg.norm(gen, axis=1, keepdims=True)
    return float(((gen @ gt.T).max(1) > threshold).mean())


@dataclass
class StructureReport:
    n: int
    alignment_rate: float
    grouped_rate: float
    mean_chars: float

    def to_dict(self):
        return asdict(self)


def structure_report(interleaved_folder: str | Path) -> StructureReport:
    files = sorted(Path(interleaved_folder).glob("*.abc"))
    if not files:
        return StructureReport(0, float("nan"), float("nan"), float("nan"))
    aligned = grouped = chars = 0
    for f in files:
        text = f.read_text()
        lines = text.splitlines(keepends=True)
        aligned += alignment_ok(lines)
        grouped += staves_grouped(lines)
        chars += len(text)
    n = len(files)
    return StructureReport(n, aligned / n, grouped / n, chars / n)
