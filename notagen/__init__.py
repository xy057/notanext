"""NotaGen: symbolic (ABC) classical music generation."""

from .config import PRESETS, ModelConfig
from .tokenizer import Patchilizer

__all__ = ["ModelConfig", "PRESETS", "Patchilizer"]
__version__ = "0.2.0"
