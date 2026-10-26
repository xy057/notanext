"""MLX backend parity with the PyTorch model (skipped off Apple Silicon)."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from conftest import TINY  # noqa: E402

from notagen.generation import Generator, TorchBackend  # noqa: E402
from notagen.mlx import MLXBackend, convert_to_mlx, load_mlx_model, model_from_torch  # noqa: E402
from notagen.mlx.model import CacheView  # noqa: E402
from notagen.model import CacheView as TCacheView  # noqa: E402
from notagen.sampling import LegacySampler, SamplingParams  # noqa: E402
from notagen.tokenizer import Patchilizer  # noqa: E402

PROMPT = [
    "%Classical\n",
    "%Mozart, Wolfgang Amadeus\n",
    "%Keyboard\n",
    "L:1/8\n",
    "M:4/4\n",
    "K:G\n",
    "[r:0/3][V:1]abcd efga|[V:2]C,8|\n",
    "[r:1/2][V:1]bagf edcB|[V:2]G,,8|\n",
]


@pytest.fixture(scope="module", autouse=True)
def mlx_cpu():
    """Strict parity runs on MLX's CPU stream: GPU fp32 matmuls are ~TF32 precision (~1e-3 relative)."""
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture(scope="module")
def pair(tiny_pair):
    _, _, tm = tiny_pair
    return tm, model_from_torch(tm, dtype="fp32")


def test_parameter_names_strict(pair):
    _, mm = pair
    from mlx.utils import tree_flatten

    names = {k for k, _ in tree_flatten(mm.parameters())}
    assert "patch_embedding._offsets" not in names and "patch_embedding.table" in names


def test_patch_hidden_matches_torch(pair, abc_text):
    tm, mm = pair
    ids = Patchilizer(patch_length=TINY.patch_length).encode_train(abc_text)[:30]
    with torch.no_grad():
        want = tm.patch_level_decoder(torch.tensor([ids])).numpy()
    got = mm.encode_patches(mx.array([ids]), CacheView(mm.new_patch_cache(1), mx.array([0]), 30))
    np.testing.assert_allclose(np.array(got), want, rtol=1e-4, atol=1e-4)


def test_char_logits_match_torch(pair):
    tm, mm = pair
    torch.manual_seed(2)
    enc = torch.randn(1, 1, TINY.hidden_size)
    toks = torch.tensor([[91, 86, 58, 49, 93]])
    with torch.no_grad():
        x = torch.cat([enc, tm.char_level_decoder.base.transformer.wte(toks)], 1)
        want = tm.char_level_decoder.step(x, TCacheView(tm.new_char_cache(1), torch.tensor([0]), 6)).numpy()
    xm = mx.concatenate([mx.array(enc.numpy()), mm.embed_chars(mx.array(toks.numpy()))], axis=1)
    got = mm.char_step(xm, CacheView(mm.new_char_cache(1), mx.array([0]), 6))
    np.testing.assert_allclose(np.array(got), want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("params", [SamplingParams(1, 1.0, 1.0), SamplingParams(9, 0.9, 1.2)])
def test_generation_matches_torch(pair, params):
    """fp32 greedy is exact; seeded legacy sampling matches as long as no draw lands on a 1e-6 boundary."""
    tm, mm = pair
    tok = Patchilizer(patch_length=TINY.patch_length)
    for seed in range(2):
        np.random.seed(seed)
        want = next(Generator(TorchBackend(tm, LegacySampler(params)), tok, max_chars=1500).generate(PROMPT, 1))
        np.random.seed(seed)
        got = next(Generator(MLXBackend(mm, params, legacy_sampler=True), tok, max_chars=1500).generate(PROMPT, 1))
        assert got.text == want.text and got.ok == want.ok


def test_fast_sampler_filters_like_torch():
    from notagen.mlx import filter_probs as mfp
    from notagen.sampling import filter_probs as tfp

    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(128) * 0.3, size=16).astype(np.float32)
    for p in (SamplingParams(9, 0.9, 1.2), SamplingParams(0, 0.5, 1.0), SamplingParams(3, 1.0, 0.7)):
        np.testing.assert_allclose(
            np.array(mfp(mx.array(probs), p)), tfp(torch.from_numpy(probs), p).numpy(), rtol=1e-5, atol=1e-6
        )


def test_batched_fast_sampling_runs(pair):
    _, mm = pair
    tok = Patchilizer(patch_length=TINY.patch_length)
    pieces = list(Generator(MLXBackend(mm, SamplingParams(), seed=0), tok, max_chars=800).generate(PROMPT, 5, 3))
    assert len(pieces) == 5 and all(p.text.startswith("".join(PROMPT)) for p in pieces)


@pytest.mark.parametrize("quantize", [None, 8])
def test_convert_roundtrip(tmp_path, pair, quantize):
    tm, _ = pair
    out = convert_to_mlx(tm, tmp_path / f"q{quantize}", dtype="fp32", quantize=quantize)
    mm = load_mlx_model(out)
    ids = [[1] * 15 + [2], [ord(c) for c in "%Classical\n"] + [2] + [0] * 4]
    with torch.no_grad():
        want = tm.patch_level_decoder(torch.tensor([ids])).numpy()
    got = np.array(mm.encode_patches(mx.array([ids]), CacheView(mm.new_patch_cache(1), mx.array([0]), 2)))
    tol = 1e-4 if quantize is None else 0.15  # 8-bit is lossy; just a sanity bound on random weights
    np.testing.assert_allclose(got, want, rtol=tol, atol=tol)


def test_gpu_close_to_torch(pair, abc_text):
    tm, _ = pair
    ids = Patchilizer(patch_length=TINY.patch_length).encode_train(abc_text)[:30]
    with torch.no_grad():
        want = tm.patch_level_decoder(torch.tensor([ids])).numpy()
    with mx.stream(mx.gpu):
        mm = model_from_torch(tm, dtype="fp32")
        got = np.array(mm.encode_patches(mx.array([ids]), CacheView(mm.new_patch_cache(1), mx.array([0]), 30)))
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-2
