import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notagen.config import ModelConfig  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TINY = ModelConfig(patch_length=48, patch_num_layers=2, char_num_layers=2, hidden_size=128, dropout=0.0)


def legacy_overrides(cfg: ModelConfig) -> dict:
    return dict(
        PATCH_SIZE=cfg.patch_size,
        PATCH_LENGTH=cfg.patch_length,
        PATCH_NUM_LAYERS=cfg.patch_num_layers,
        CHAR_NUM_LAYERS=cfg.char_num_layers,
        HIDDEN_SIZE=cfg.hidden_size,
        PATCH_STREAM=True,
        PATCH_SAMPLING_BATCH_SIZE=0,
    )


@pytest.fixture(scope="session")
def abc_text():
    return (FIXTURES / "interleaved.abc").read_text()


@pytest.fixture(scope="session")
def tiny_pair():
    """(legacy HF model, new model) with identical random weights, fp32, eval mode."""
    from legacy import legacy_model, load_legacy

    from notagen.model import NotaGenLMHeadModel

    torch.manual_seed(0)
    mod = load_legacy("inference", **legacy_overrides(TINY))
    old = legacy_model(mod, TINY).eval()
    # Random-init weights are small; scale up so attention/LayerNorm paths are actually exercised.
    with torch.no_grad():
        for p in old.parameters():
            p.mul_(3.0)
    new = NotaGenLMHeadModel(TINY).eval()
    from notagen.checkpoint import clean_state_dict

    new.load_state_dict(clean_state_dict(old.state_dict()), strict=True)
    return mod, old, new
