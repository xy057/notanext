"""Golden-output checks on a released checkpoint. Skipped unless ``NOTAGEN_WEIGHTS`` points at one.

    NOTAGEN_WEIGHTS=weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth \
        .venv/bin/python -m pytest tests/test_real_weights.py -v

Everything runs in fp32 on CPU so the legacy and new code can be compared exactly. The first run
writes ``tests/fixtures/golden_<weights stem>.json``; later runs compare against it, which catches
regressions without needing the legacy code or transformers.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

WEIGHTS = os.environ.get("NOTAGEN_WEIGHTS")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="set NOTAGEN_WEIGHTS to a released checkpoint")

from conftest import FIXTURES, legacy_overrides  # noqa: E402

from notagen.generation import Generator, TorchBackend  # noqa: E402
from notagen.sampling import LegacySampler, SamplingParams  # noqa: E402
from notagen.tokenizer import Patchilizer  # noqa: E402

PROMPTS = {
    "chopin": ["%Romantic\n", "%Chopin, Frederic\n", "%Keyboard\n"],
    "mozart": ["%Classical\n", "%Mozart, Wolfgang Amadeus\n", "%Chamber\n"],
}
MAX_CHARS = 700  # ~60 patches: long enough to cover metadata, the [r:0/ hack and several bars
GREEDY = SamplingParams(1, 1.0, 1.0)
SAMPLED = SamplingParams(9, 0.9, 1.2)


def _golden_path():
    return FIXTURES / f"golden_{Path(WEIGHTS).stem[:60]}.json"


@pytest.fixture(scope="module")
def models():
    from legacy import legacy_model, load_legacy

    from notagen.checkpoint import load_model

    torch.set_num_threads(max(1, os.cpu_count() // 2))
    new = load_model(WEIGHTS)  # weights_only=True
    mod = load_legacy("inference", **legacy_overrides(new.cfg))
    old = legacy_model(mod, new.cfg).eval()
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    old.load_state_dict(ck["model"], strict=True)
    return mod, old, new


def _new_piece(model, params, prompt, seed):
    np.random.seed(seed)
    gen = Generator(TorchBackend(model, LegacySampler(params)), Patchilizer.from_config(model.cfg), max_chars=MAX_CHARS)
    return next(gen.generate(prompt, 1)).text


def _legacy_piece(mod, old, params, prompt, seed):
    from test_generation import legacy_loop

    def generate(x):
        with torch.no_grad():
            return old.generate(x, top_k=params.top_k, top_p=params.top_p, temperature=params.temperature)

    np.random.seed(seed)
    text, _ = legacy_loop(generate, mod.Patchilizer(), prompt, 16, mod.CONFIG.PATCH_LENGTH, MAX_CHARS)
    return text


def test_forward_matches_legacy(models):
    _, old, new = models
    tok = Patchilizer.from_config(new.cfg)
    ids = torch.tensor([tok.encode_train((FIXTURES / "interleaved.abc").read_text())])
    masks = torch.ones(ids.shape[:2], dtype=torch.long)
    with torch.no_grad():
        a = old(ids.reshape(1, -1).clone(), masks.clone()).loss
        b = new(ids.reshape(1, -1).clone(), masks.clone())
        ha = old.patch_level_decoder(ids.reshape(1, -1))["last_hidden_state"]
        hb = new.patch_level_decoder(ids)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(ha, hb, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("name", list(PROMPTS))
@pytest.mark.parametrize("kind", ["greedy", "sampled"])
def test_generation_matches_legacy(models, name, kind):
    mod, old, new = models
    params = GREEDY if kind == "greedy" else SAMPLED
    want = _legacy_piece(mod, old, params, PROMPTS[name], seed=0)
    got = _new_piece(new, params, PROMPTS[name], seed=0)
    assert got == want


def test_mlx_matches_torch(models):
    mx = pytest.importorskip("mlx.core")
    from notagen.mlx import MLXBackend, model_from_torch

    _, _, new = models
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mm = model_from_torch(new, dtype="fp32")
        tok = Patchilizer.from_config(new.cfg)
        for name, prompt in PROMPTS.items():
            for params in (GREEDY, SAMPLED):
                want = _new_piece(new, params, prompt, seed=0)
                np.random.seed(0)
                gen = Generator(MLXBackend(mm, params, legacy_sampler=True), tok, max_chars=MAX_CHARS)
                assert next(gen.generate(prompt, 1)).text == want, (name, params)
    finally:
        mx.set_default_device(prev)


def test_golden_outputs(models):
    """Record on first run; compare afterwards (no legacy code involved)."""
    _, _, new = models
    out = {}
    for name, prompt in PROMPTS.items():
        for kind, params in (("greedy", GREEDY), ("sampled", SAMPLED)):
            text = _new_piece(new, params, prompt, seed=0)
            out[f"{name}/{kind}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "head": text[:300]}
    path = _golden_path()
    if not path.exists():
        path.write_text(json.dumps({"weights": Path(WEIGHTS).name, "max_chars": MAX_CHARS, "outputs": out}, indent=2))
        pytest.skip(f"recorded {path.name}")
    golden = json.loads(path.read_text())["outputs"]
    assert {k: v["sha256"] for k, v in out.items()} == {k: v["sha256"] for k, v in golden.items()}
