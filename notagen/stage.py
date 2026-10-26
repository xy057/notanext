"""Helpers for the per-stage scripts, which keep their ``config.py`` files as the user-facing settings."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_config_from(cfg_module):
    """Build a ModelConfig from a stage ``config.py``'s legacy globals."""
    from .config import ModelConfig

    return ModelConfig(
        patch_size=cfg_module.PATCH_SIZE,
        patch_length=cfg_module.PATCH_LENGTH,
        patch_num_layers=cfg_module.PATCH_NUM_LAYERS,
        char_num_layers=cfg_module.CHAR_NUM_LAYERS,
        hidden_size=cfg_module.HIDDEN_SIZE,
        stream=getattr(cfg_module, "PATCH_STREAM", True),
    )


def ensure_importable():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
