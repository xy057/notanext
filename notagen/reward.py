"""Scoring generated pieces: structure checks + CLaMP 2 fidelity / diversity (eval and RL reward).

Two CLaMP 2 fidelity signals are supported:

* **text fidelity**: cosine between a piece and a text description of its prompt
  (:func:`notagen.clamp2.prompt_text`). Needs no data, so it works with the released weights alone.
* **reference fidelity**: cosine to the mean feature of a reference set. This is the paper's
  CLaMP 2 score and needs ground-truth features (e.g. from ``notagen clamp2 extract``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .abc import rest_unreduce, to_lines
from .clamp2 import CLaMP2Encoder, cosine_matrix, prompt_text
from .metrics import alignment_ok, staves_grouped


@dataclass
class Prompt:
    period: str
    composer: str
    instrumentation: str

    @property
    def lines(self) -> list[str]:
        return ["%" + self.period + "\n", "%" + self.composer + "\n", "%" + self.instrumentation + "\n"]

    @property
    def text(self) -> str:
        return prompt_text(self.period, self.composer, self.instrumentation)

    @classmethod
    def parse(cls, s: str) -> Prompt:
        parts = [p.strip() for p in s.replace("_", "|").split("|")]
        if len(parts) != 3:
            raise ValueError(f"prompt must be 'period|composer|instrumentation', got {s!r}")
        return cls(*parts)

    def __str__(self):
        return f"{self.period}|{self.composer}|{self.instrumentation}"


@dataclass
class RewardWeights:
    text: float = 4.0  # CLaMP 2 piece <-> prompt-description cosine (typical range -0.1 .. 0.3)
    reference: float = 0.0  # CLaMP 2 piece <-> reference centroid cosine (needs reference features)
    aligned: float = 0.5  # bars agree in duration/count across voices
    grouped: float = 0.25  # staves of one instrument are adjacent
    diversity: float = 1.0  # penalty: mean cosine to the other pieces of the same group
    invalid: float = -1.0  # reward for pieces that did not finish or cannot be restored


@dataclass
class PieceScore:
    ok: bool
    restored: bool = False
    aligned: bool = False
    grouped: bool = False
    text_sim: float = float("nan")
    ref_sim: float = float("nan")
    group_sim: float = float("nan")
    reward: float = float("nan")
    interleaved: str = field(default="", repr=False)

    def to_dict(self):
        d = asdict(self)
        d.pop("interleaved")
        return d


class Scorer:
    def __init__(
        self,
        encoder: CLaMP2Encoder | None,
        weights: RewardWeights | None = None,
        reference_centroid: np.ndarray | None = None,
    ):
        self.enc = encoder
        self.w = weights or RewardWeights()
        self.ref = reference_centroid
        self._text_cache: dict[str, np.ndarray] = {}

    def text_feature(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            self._text_cache[text] = self.enc.text_features([text])[0]
        return self._text_cache[text]

    def score_group(self, texts: list[str], oks: list[bool], prompt: Prompt) -> tuple[list[PieceScore], np.ndarray]:
        """Score one group (same prompt). Returns scores and the CLaMP 2 features of valid pieces."""
        scores = []
        for text, ok in zip(texts, oks):
            s = PieceScore(ok=ok)
            if ok:
                try:
                    lines = rest_unreduce(to_lines(text))
                    s.restored = True
                    s.aligned = alignment_ok(lines)
                    s.grouped = staves_grouped(lines)
                    s.interleaved = "".join(lines)
                except Exception:
                    pass
            scores.append(s)
        valid = [i for i, s in enumerate(scores) if s.restored]
        feats = np.zeros((0, 768), dtype=np.float32)
        if self.enc is not None and valid:
            feats = self.enc.music_features([scores[i].interleaved for i in valid])
            tsim = cosine_matrix(feats, self.text_feature(prompt.text)[None])[:, 0]
            gsim = cosine_matrix(feats, feats)
            for j, i in enumerate(valid):
                scores[i].text_sim = float(tsim[j])
                if len(valid) > 1:
                    scores[i].group_sim = float((gsim[j].sum() - gsim[j, j]) / (len(valid) - 1))
                if self.ref is not None:
                    scores[i].ref_sim = float(cosine_matrix(feats[j : j + 1], self.ref[None])[0, 0])
        w = self.w
        for s in scores:
            if not s.restored:
                s.reward = w.invalid
                continue
            r = w.aligned * s.aligned + w.grouped * s.grouped
            if not np.isnan(s.text_sim):
                r += w.text * s.text_sim
            if not np.isnan(s.ref_sim):
                r += w.reference * s.ref_sim
            if not np.isnan(s.group_sim):
                r -= w.diversity * s.group_sim
            s.reward = float(r)
        return scores, feats


def retrieval_accuracy(
    features: np.ndarray, labels: np.ndarray, candidate_texts: list[str], enc: CLaMP2Encoder
) -> float:
    """Share of pieces whose most similar candidate prompt description is their own prompt."""
    if len(features) == 0:
        return float("nan")
    t = enc.text_features(candidate_texts)
    return float((cosine_matrix(features, t).argmax(1) == labels).mean())
