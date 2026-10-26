"""Generation throughput benchmark (random weights are fine: speed doesn't depend on values)."""

from __future__ import annotations

import time

import torch

from .config import PRESETS, ModelConfig
from .generation import TorchBackend
from .sampling import SamplingParams, TorchSampler
from .tokenizer import Patchilizer

PROMPT = ["%Romantic\n", "%Chopin, Frederic\n", "%Keyboard\n"]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_torch(cfg: ModelConfig, device, dtype, n_patches=64, batch=1, warmup=4, context=0) -> dict:
    """Patches/s for ``batch`` rows. ``context`` pre-fills that many patches (long-piece regime)."""
    from .model import NotaGenLMHeadModel

    torch.manual_seed(0)
    model = NotaGenLMHeadModel(cfg).to(device=device, dtype=dtype).eval()
    backend = TorchBackend(model, TorchSampler(SamplingParams()))
    prompt = Patchilizer.from_config(cfg).encode_prompt(PROMPT)
    prompt = prompt + [prompt[-1]] * context
    backend.start(batch)
    for r in range(batch):
        backend.prefill(r, prompt)
    rows = list(range(batch))
    for i in range(warmup + n_patches):
        if i == warmup:
            _sync(device)
            t0 = time.perf_counter()
        out = backend.sample(rows, [[] for _ in rows])
        backend.feed({r: o for r, o in zip(rows, out)})
    _sync(device)
    dt = time.perf_counter() - t0
    return {
        "patches_per_s": batch * n_patches / dt,
        "ms_per_step": 1000 * dt / n_patches,
        "batch": batch,
        "context": len(prompt) + n_patches,
    }


def bench_mlx(cfg: ModelConfig, dtype="bf16", n_patches=64, batch=1, warmup=4, context=0, quantize=None) -> dict:
    import mlx.core as mx

    from .mlx import _DTYPES, MLXBackend, _quantize
    from .mlx.model import NotaGenMLX

    mx.random.seed(0)
    model = NotaGenMLX(cfg)
    model.set_dtype(_DTYPES[dtype])
    for leaf in ("table", "bias"):  # random init for the hand-made table params
        setattr(
            model.patch_embedding,
            leaf,
            (mx.random.normal(getattr(model.patch_embedding, leaf).shape) * 0.02).astype(_DTYPES[dtype]),
        )
    if quantize:
        _quantize(model, quantize)
    mx.eval(model.parameters())
    backend = MLXBackend(model, SamplingParams(), seed=0)
    prompt = Patchilizer.from_config(cfg).encode_prompt(PROMPT)
    prompt = prompt + [prompt[-1]] * context
    backend.start(batch)
    for r in range(batch):
        backend.prefill(r, prompt)
    rows = list(range(batch))
    for i in range(warmup + n_patches):
        if i == warmup:
            t0 = time.perf_counter()
        out = backend.sample(rows, [[] for _ in rows])
        backend.feed({r: o for r, o in zip(rows, out)})
    mx.eval(backend.hidden)
    dt = time.perf_counter() - t0
    return {
        "patches_per_s": batch * n_patches / dt,
        "ms_per_step": 1000 * dt / n_patches,
        "batch": batch,
        "context": len(prompt) + n_patches,
    }


def main(argv=None):
    import argparse
    import json

    from .device import pick_device, pick_dtype

    ap = argparse.ArgumentParser("notagen bench")
    ap.add_argument("--preset", default="large", choices=sorted(PRESETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--patches", type=int, default=64)
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--context", type=int, default=0)
    ap.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    ap.add_argument("--quantize", type=int, choices=[4, 8])
    a = ap.parse_args(argv)
    if a.backend == "mlx":
        dt = "bf16" if a.dtype == "auto" else a.dtype
        for b in a.batch:
            r = bench_mlx(PRESETS[a.preset], dt, a.patches, b, context=a.context, quantize=a.quantize)
            print(json.dumps({"preset": a.preset, "backend": "mlx", "dtype": dt, "quantize": a.quantize, **r}))
        return
    dev = pick_device(a.device)
    dt = pick_dtype(a.dtype, dev)
    for b in a.batch:
        r = bench_torch(PRESETS[a.preset], dev, dt, a.patches, b, context=a.context)
        print(json.dumps({"preset": a.preset, "device": str(dev), "dtype": str(dt).split(".")[-1], **r}))


if __name__ == "__main__":
    main()
