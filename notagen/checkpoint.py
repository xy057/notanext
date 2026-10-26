"""Load legacy ``.pth`` checkpoints and save/load the new ``safetensors`` + ``config.json`` format."""

from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig
from .model import NotaGenLMHeadModel

# Non-persistent buffers some transformers versions wrote into state dicts.
_IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias")
_TIED_KEY = "char_level_decoder.base.lm_head.weight"
_WTE_KEY = "char_level_decoder.base.transformer.wte.weight"


def clean_state_dict(sd: dict) -> dict:
    sd = {k.removeprefix("module."): v for k, v in sd.items() if not k.endswith(_IGNORED_SUFFIXES)}
    sd.setdefault(_TIED_KEY, sd.get(_WTE_KEY))
    return sd


def read_legacy_state_dict(path: str | Path, trust_pickle: bool = False) -> dict:
    """Read a legacy checkpoint. ``weights_only=True`` unless the caller explicitly trusts the file."""
    ckpt = torch.load(path, map_location="cpu", weights_only=not trust_pickle)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    return clean_state_dict(sd)


def load_model(
    path: str | Path,
    cfg: ModelConfig | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    trust_pickle: bool = False,
) -> NotaGenLMHeadModel:
    """Load either format. For a directory or ``.safetensors`` file the config comes from
    ``config.json``; for a legacy ``.pth`` it is parsed from the filename unless given."""
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        cfg = cfg or ModelConfig.load(path.with_name("config.json"))
        sd = clean_state_dict(load_file(str(path)))
    else:
        cfg = cfg or ModelConfig.from_legacy_name(path.name)
        sd = read_legacy_state_dict(path, trust_pickle)
    model = NotaGenLMHeadModel(cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def save_model(model: NotaGenLMHeadModel, out_dir: str | Path) -> Path:
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k != _TIED_KEY}
    save_file(sd, str(out / "model.safetensors"))
    model.cfg.save(out / "config.json")
    return out
