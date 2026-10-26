"""Next-char samplers.

``TorchSampler`` runs entirely on the model's device (no host round-trip per char).
``LegacySampler`` reproduces the original NumPy chain bit-for-bit, including its use of the
global ``np.random`` state, so seeded legacy runs can be replayed for regression tests.

The two differ in one documented way: the legacy chain adds 1e-12 to every probability
before each step, so tokens removed by top-k/top-p keep a ~1e-10 chance after temperature.
``TorchSampler`` removes them exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SamplingParams:
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2


def filter_probs(probs: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    """top-k -> top-p -> temperature on ``[b, vocab]`` float32 probabilities (rows sum to 1)."""
    if p.top_k > 0:
        kth = probs.topk(min(p.top_k, probs.shape[-1]), dim=-1).values[..., -1:]
        probs = torch.where(probs >= kth, probs, 0.0)
        probs = probs / probs.sum(-1, keepdim=True)
    if 0 < p.top_p < 1:
        sorted_p, order = probs.sort(-1, descending=True)
        remove = sorted_p.cumsum(-1) > p.top_p
        remove[..., 1:] = remove[..., :-1].clone()  # keep the token that crosses top_p
        remove[..., 0] = False
        probs = probs.scatter(-1, order, sorted_p.masked_fill(remove, 0.0))
        probs = probs / probs.sum(-1, keepdim=True)
    if p.temperature != 1:
        probs = probs.pow(1.0 / p.temperature)
        probs = probs / probs.sum(-1, keepdim=True)
    return probs


class TorchSampler:
    def __init__(self, params: SamplingParams, generator: torch.Generator | None = None):
        self.params = params
        self.generator = generator

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        probs = filter_probs(logits.float().softmax(-1), self.params)
        # Exponential-race sampling: argmax(p / E), E ~ Exp(1). Equivalent to multinomial and
        # works with a seeded generator on every backend.
        e = torch.empty_like(probs).exponential_(generator=self.generator)
        return (probs / e).argmax(-1)


def _safe_normalize_probs(probs):
    epsilon = 1e-12
    probs = np.array(probs, dtype=np.float64)
    probs = np.where(np.isnan(probs) | (probs < 0), 0, probs)
    probs = probs + epsilon
    s = probs.sum()
    if s > 0:
        probs = probs / s
    else:
        probs = np.zeros_like(probs)
        probs[0] = 1.0
    return probs


class LegacySampler:
    def __init__(self, params: SamplingParams):
        from samplings import temperature_sampling, top_k_sampling, top_p_sampling

        self._fns = top_k_sampling, top_p_sampling, temperature_sampling
        self.params = params

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        top_k_sampling, top_p_sampling, temperature_sampling = self._fns
        p = self.params
        out = []
        for prob in logits.float().softmax(-1).cpu().numpy():
            prob = _safe_normalize_probs(prob)
            prob = top_k_sampling(prob, top_k=p.top_k, return_probs=True)
            prob = _safe_normalize_probs(prob)
            prob = top_p_sampling(prob, top_p=p.top_p, return_probs=True)
            prob = _safe_normalize_probs(prob)
            out.append(temperature_sampling(prob, temperature=p.temperature))
        return torch.tensor(out, device=logits.device)
