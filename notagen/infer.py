"""High-level inference: load weights, pick a backend, write legacy-compatible output folders."""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

from .abc import rest_unreduce, strip_prompt_and_number, to_lines
from .generation import Generator
from .sampling import SamplingParams
from .tokenizer import Patchilizer


@dataclass
class InferenceOptions:
    weights: str
    backend: str = "auto"  # auto | torch | mlx
    device: str = "auto"
    dtype: str = "auto"
    sampler: str = "fast"  # fast (on-device) | legacy (NumPy, bit-compatible with the original scripts)
    seed: int | None = None
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2
    batch_size: int = 1
    trust_pickle: bool = False
    model_config: object | None = None  # ModelConfig; needed only for a .pth with a non-standard name


def _mlx_available() -> bool:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(name: str) -> str:
    if name != "auto":
        return name
    return "mlx" if _mlx_available() else "torch"


def load(opts: InferenceOptions):
    """Return ``(backend, patchilizer)`` ready for :class:`Generator`."""
    params = SamplingParams(opts.top_k, opts.top_p, opts.temperature)
    kind = resolve_backend(opts.backend)
    if kind == "mlx":
        from .mlx import MLXBackend, load_mlx_model

        model = load_mlx_model(opts.weights, cfg=opts.model_config, dtype=opts.dtype, trust_pickle=opts.trust_pickle)
        backend = MLXBackend(model, params, seed=opts.seed, legacy_sampler=opts.sampler == "legacy")
        cfg = model.cfg
    else:
        import numpy as np
        import torch

        from .checkpoint import load_model
        from .device import pick_device, pick_dtype
        from .generation import TorchBackend
        from .sampling import LegacySampler, TorchSampler

        device = pick_device(opts.device)
        model = load_model(
            opts.weights,
            cfg=opts.model_config,
            device=device,
            dtype=pick_dtype(opts.dtype, device),
            trust_pickle=opts.trust_pickle,
        )
        if opts.sampler == "legacy":
            if opts.seed is not None:
                np.random.seed(opts.seed)
            sampler = LegacySampler(params)
        else:
            gen = None
            if opts.seed is not None:
                gen = torch.Generator(device=device).manual_seed(opts.seed)
            sampler = TorchSampler(params, gen)
        backend = TorchBackend(model, sampler)
        cfg = model.cfg
    return backend, Patchilizer.from_config(cfg)


def output_folder_name(weights: str, opts: InferenceOptions) -> str:
    """Same naming as ``inference/config.py`` so downstream stages find the folders."""
    stem = os.path.splitext(os.path.split(weights.rstrip("/"))[-1])[0]
    return f"{stem}_k_{opts.top_k}_p_{opts.top_p}_temp_{opts.temperature}"


def batch_inference(
    opts: InferenceOptions,
    num_samples: int,
    out_root: str = "../output",
    prompt_lines: list[str] | None = None,
    verbose: bool = True,
    max_attempts: int | None = None,
) -> tuple[Path, Path]:
    """Port of ``inference/inference.py``: keep generating until ``num_samples`` pieces both finish
    and survive ``rest_unreduce``; write them to ``original/`` and ``interleaved/``."""
    prompt_lines = prompt_lines or []
    name = output_folder_name(opts.weights, opts)
    original = Path(out_root) / "original" / name
    interleaved = Path(out_root) / "interleaved" / name
    original.mkdir(parents=True, exist_ok=True)
    interleaved.mkdir(parents=True, exist_ok=True)

    backend, tok = load(opts)
    gen = Generator(backend, tok)
    file_no = attempts = 1
    while file_no <= num_samples:
        if max_attempts is not None and attempts > max_attempts:
            raise RuntimeError(f"only {file_no - 1}/{num_samples} pieces succeeded in {max_attempts} attempts")
        for piece in gen.generate(prompt_lines, num_samples - file_no + 1, opts.batch_size):
            attempts += 1
            if not piece.ok:
                if verbose:
                    print(f"failed ({piece.reason})")
                continue
            filename = time.strftime("%Y%m%d-%H%M%S") + "_" + format(piece.seconds, ".2f") + "_" + str(file_no) + ".abc"
            try:
                lines = rest_unreduce(to_lines(piece.text))
            except Exception:
                continue
            (interleaved / filename).write_text("".join(lines))
            (original / filename).write_text(piece.text)
            if verbose:
                print(f"[{file_no}/{num_samples}] {filename}")
            file_no += 1
            if file_no > num_samples:
                break
    return original, interleaved


def generate_for_demo(
    backend, tok, period: str, composer: str, instrumentation: str, on_text=None, time_limit: float = 10 * 60
) -> str:
    """Port of ``gradio/inference.py:inference_patch``: retry until a piece converts cleanly."""
    prompt = ["%" + period + "\n", "%" + composer + "\n", "%" + instrumentation + "\n"]
    gen = Generator(backend, tok, time_limit=time_limit, on_text=lambda _i, t: on_text and on_text(t))
    while True:
        if on_text:
            on_text("".join(prompt))
        piece = next(gen.generate(prompt, 1))
        if not piece.ok:
            continue
        try:
            lines = rest_unreduce(to_lines(piece.text))
        except Exception:
            continue
        return strip_prompt_and_number(lines)
