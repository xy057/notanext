"""Native MLX backend: model, sampler, generation backend, and weight conversion."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import ModelConfig
from ..sampling import LegacySampler, SamplingParams
from .model import CacheView, NotaGenMLX, from_torch_state_dict

_DTYPES = {
    "fp32": mx.float32,
    "float32": mx.float32,
    "bf16": mx.bfloat16,
    "bfloat16": mx.bfloat16,
    "fp16": mx.float16,
    "float16": mx.float16,
    "auto": mx.bfloat16,
}

__all__ = ["MLXBackend", "NotaGenMLX", "convert_to_mlx", "load_mlx_model", "model_from_torch"]


def _quantize(model: NotaGenMLX, bits: int, group_size: int = 64):
    # Only the transformer Linear layers: embeddings/LM head stay full precision (they are tiny
    # and the char LM head sets the sampling distribution directly).
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda path, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % group_size == 0,
    )


def model_from_torch(torch_model, dtype="auto", quantize: int | None = None) -> NotaGenMLX:
    sd = {k: v.detach().float().cpu().numpy() for k, v in torch_model.state_dict().items()}
    model = NotaGenMLX(torch_model.cfg)
    weights = {k: mx.array(v).astype(_DTYPES[dtype]) for k, v in from_torch_state_dict(sd, torch_model.cfg).items()}
    model.load_weights(list(weights.items()), strict=True)
    if quantize:
        _quantize(model, quantize)
    mx.eval(model.parameters())
    return model


def convert_to_mlx(torch_model, out_dir, dtype="bf16", quantize: int | None = None) -> Path:
    """Write ``weights.safetensors`` + ``config.json`` (with an ``mlx`` section) to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model_from_torch(torch_model, dtype, quantize)
    model.save_weights(str(out / "weights.safetensors"))
    meta = {"model_type": "notagen", **torch_model.cfg.to_dict(), "mlx": {"dtype": dtype, "quantize": quantize}}
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    return out


def load_mlx_model(path, cfg: ModelConfig | None = None, dtype="auto", trust_pickle=False) -> NotaGenMLX:
    """Load a converted MLX dir directly, or convert a torch checkpoint (.pth/.safetensors) on the fly."""
    path = Path(path)
    if path.is_dir() and (path / "weights.safetensors").exists():
        meta = json.loads((path / "config.json").read_text())
        model = NotaGenMLX(ModelConfig.from_dict(meta))
        if meta.get("mlx", {}).get("quantize"):
            _quantize(model, meta["mlx"]["quantize"])
        model.load_weights(str(path / "weights.safetensors"), strict=True)
        mx.eval(model.parameters())
        return model
    from ..checkpoint import load_model

    return model_from_torch(load_model(path, cfg=cfg, trust_pickle=trust_pickle), dtype)


def filter_probs(probs: mx.array, p: SamplingParams) -> mx.array:
    """Same chain as ``notagen.sampling.filter_probs``: top-k -> top-p -> temperature."""
    if p.top_k > 0:
        kth = mx.sort(probs, axis=-1)[..., -min(p.top_k, probs.shape[-1])][..., None]
        probs = mx.where(probs >= kth, probs, 0.0)
        probs = probs / probs.sum(-1, keepdims=True)
    if 0 < p.top_p < 1:
        order = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, order, axis=-1)
        cum = mx.cumsum(sorted_p, axis=-1)
        # keep tokens whose preceding cumulative mass is <= top_p (keeps the crossing token)
        keep_sorted = (cum - sorted_p) <= p.top_p
        keep_sorted[..., 0] = True
        inv = mx.argsort(order, axis=-1)
        keep = mx.take_along_axis(keep_sorted, inv, axis=-1)
        probs = mx.where(keep, probs, 0.0)
        probs = probs / probs.sum(-1, keepdims=True)
    if p.temperature != 1:
        probs = probs ** (1.0 / p.temperature)
        probs = probs / probs.sum(-1, keepdims=True)
    return probs


class MLXBackend:
    """Implements the ``notagen.generation.Backend`` protocol on MLX."""

    def __init__(self, model: NotaGenMLX, params: SamplingParams, seed: int | None = None, legacy_sampler=False):
        self.model = model
        self.params = params
        self.patch_size = model.cfg.patch_size
        self.patch_length = model.cfg.patch_length
        # Never draw the default key from np.random: the legacy sampler replays that global stream.
        self.key = mx.random.key(seed if seed is not None else int.from_bytes(os.urandom(4), "little") >> 1)
        self.legacy = LegacySampler(params) if legacy_sampler else None

    def _sample(self, logits: mx.array) -> mx.array:
        if self.legacy is not None:
            import torch

            ids = self.legacy(torch.from_numpy(np.array(logits.astype(mx.float32))))
            return mx.array(ids.numpy())
        probs = filter_probs(mx.softmax(logits.astype(mx.float32), axis=-1), self.params)
        self.key, sub = mx.random.split(self.key)
        return mx.random.categorical(mx.log(probs), key=sub)

    def start(self, batch_size: int) -> None:
        self.cache = self.model.new_patch_cache(batch_size)
        self.lens = [0] * batch_size
        self.hidden = mx.zeros((batch_size, self.model.cfg.hidden_size), dtype=self.model.dtype)

    def prefill(self, row: int, patches: list[list[int]]) -> None:
        x = mx.array([patches])
        out = self.model.encode_patches(x, CacheView(self.cache, mx.array([0]), len(patches), rows=mx.array([row])))
        self.hidden[row] = out[0, -1]
        self.lens[row] = len(patches)
        mx.eval(self.hidden, *self.cache.k, *self.cache.v)

    def feed(self, patches: dict[int, list[int]]) -> None:
        bsz = self.cache.batch
        x = np.zeros((bsz, 1, self.patch_size), dtype=np.int32)
        for r, p in patches.items():
            assert self.lens[r] < self.patch_length, "patch context overflow"
            x[r, 0] = p
        start = [min(l, self.patch_length - 1) for l in self.lens]
        out = self.model.encode_patches(mx.array(x), CacheView(self.cache, mx.array(start), max(start) + 1))
        idx = mx.array(list(patches))
        self.hidden[idx] = out[idx, 0]
        for r in patches:
            self.lens[r] += 1
        # No eval here: the patch step stays lazy and is fused into the following char sampling graph.

    def sample(self, rows: list[int], prefixes: list[list[int]]) -> list[list[int]]:
        model, ps = self.model, self.patch_size
        b = len(rows)
        cache = model.new_char_cache(b)
        need = [ps - len(p) for p in prefixes]
        enc = self.hidden[mx.array(rows)]
        logits = mx.zeros((b, model.cfg.vocab_size), dtype=model.dtype)

        by_len: dict[int, list[int]] = {}
        for i, p in enumerate(prefixes):
            by_len.setdefault(len(p), []).append(i)
        for plen, members in by_len.items():
            idx = mx.array(members)
            x = enc[idx][:, None]
            if plen:
                x = mx.concatenate([x, model.embed_chars(mx.array([prefixes[i] for i in members]))], axis=1)
            view = CacheView(cache, mx.zeros((len(members),), dtype=mx.int32), plen + 1, rows=idx)
            logits[idx] = model.char_step(x, view)[:, -1]

        pos = [len(p) + 1 for p in prefixes]
        steps = []
        for step in range(max(need)):
            tokens = self._sample(logits)
            steps.append(tokens)
            if step == max(need) - 1:
                break
            start = [min(p, ps) for p in pos]
            view = CacheView(cache, mx.array(start), max(start) + 1)
            logits = model.char_step(model.embed_chars(tokens)[:, None], view)[:, -1]
            pos = [p + 1 for p in pos]
        host = np.array(mx.stack(steps, axis=1)).tolist()  # single evaluation for the whole patch
        return [host[i][: need[i]] for i in range(b)]
