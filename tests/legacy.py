"""Import a stage's pre-refactor ``utils.py`` as a regression oracle, with its ``config`` globals overridden."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REF = Path(__file__).resolve().parent / "legacy_ref"


def load_legacy(stage: str, **overrides):
    """Load the frozen pre-refactor ``<stage>/utils.py`` (kept in tests/legacy_ref/)."""
    cfg_mod = types.ModuleType("config")
    cfg_mod.__dict__.update(PATCH_SAMPLING_BATCH_SIZE=0, PATCH_STREAM=True)
    cfg_mod.__dict__.update(overrides)
    saved = sys.modules.get("config")
    sys.modules["config"] = cfg_mod
    try:
        spec = importlib.util.spec_from_file_location(f"legacy_{stage}_utils", REF / f"{stage}_utils.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)
    mod.CONFIG = cfg_mod
    return mod


def legacy_model(mod, cfg):
    from transformers import GPT2Config

    patch_config = GPT2Config(
        num_hidden_layers=cfg.patch_num_layers,
        max_length=cfg.patch_length,
        max_position_embeddings=cfg.patch_length,
        n_embd=cfg.hidden_size,
        num_attention_heads=cfg.hidden_size // 64,
        vocab_size=1,
    )
    char_config = GPT2Config(
        num_hidden_layers=cfg.char_num_layers,
        max_length=cfg.patch_size + 1,
        max_position_embeddings=cfg.patch_size + 1,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.hidden_size // 64,
        vocab_size=128,
    )
    return mod.NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config)
