"""Model configuration.

The legacy code kept these as module-level globals in each stage's ``config.py`` and
encoded them only in checkpoint filenames. Here they are a dataclass that is saved
next to the weights (``config.json``) and can still be recovered from a legacy filename.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 16
    patch_length: int = 1024
    patch_num_layers: int = 20
    char_num_layers: int = 6
    hidden_size: int = 1280
    stream: bool = True
    # The char vocabulary is ASCII (< 128); id 0 = pad/special, 1 = BOS, 2 = EOS.
    vocab_size: int = 128
    head_dim: int = 64
    layer_norm_eps: float = 1e-5
    dropout: float = 0.1

    @property
    def num_heads(self) -> int:
        return self.hidden_size // self.head_dim

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"model_type": "notagen", **self.to_dict()}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_legacy_name(cls, name: str) -> ModelConfig:
        """Parse hyperparameters out of a legacy checkpoint filename.

        Handles all released names, including the fine-tune checkpoint whose name has
        the layer counts swapped (``p_layers_c_layers_6_20``) and NotaGen-X, which omits
        ``c_layers`` (it uses the large model's 6).
        """
        name = Path(name).name

        def grab(key: str) -> int | None:
            m = re.search(rf"(?:^|_){key}_(\d+)", name)
            return int(m.group(1)) if m else None

        kw: dict = {}
        if (v := grab("p_size")) is not None:
            kw["patch_size"] = v
        if (v := grab("p_length")) is not None:
            kw["patch_length"] = v
        if (v := grab("h_size")) is not None:
            kw["hidden_size"] = v
        swapped = re.search(r"p_layers_c_layers_(\d+)_(\d+)", name)
        if swapped:
            kw["char_num_layers"], kw["patch_num_layers"] = int(swapped.group(1)), int(swapped.group(2))
        else:
            if (v := grab("p_layers")) is not None:
                kw["patch_num_layers"] = v
            if (v := grab("c_layers")) is not None:
                kw["char_num_layers"] = v
        if not kw:
            raise ValueError(f"Cannot infer model config from checkpoint name {name!r}")
        return cls(**kw)


PRESETS = {
    "small": ModelConfig(patch_length=2048, patch_num_layers=12, char_num_layers=3, hidden_size=768),
    "medium": ModelConfig(patch_length=2048, patch_num_layers=16, char_num_layers=3, hidden_size=1024),
    "large": ModelConfig(patch_length=1024, patch_num_layers=20, char_num_layers=6, hidden_size=1280),
}
PRESETS["x"] = PRESETS["large"]
