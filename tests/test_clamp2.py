"""CLaMP 2 integration checks. Need ``weights/clamp2`` (or ``NOTAGEN_CLAMP2``); skipped otherwise."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CLAMP_DIR = Path(os.environ.get("NOTAGEN_CLAMP2", ROOT / "weights" / "clamp2"))
pytestmark = pytest.mark.skipif(
    not (CLAMP_DIR / "xlm-roberta-base" / "config.json").exists() or not any(CLAMP_DIR.glob("weights_clamp2_*.pth")),
    reason="CLaMP 2 weights not present",
)

from conftest import FIXTURES  # noqa: E402

from notagen.clamp2 import CLaMP2Encoder, _segments, cosine_matrix, prompt_text  # noqa: E402


def test_segments_match_legacy_rule():
    for n, L in [(5, 8), (8, 8), (13, 8), (16, 8), (17, 8)]:
        legacy = [list(range(n))[i : i + L] for i in range(0, n, L)]
        legacy[-1] = list(range(n))[-L:]
        cnt, rem = divmod(n, L)
        lw = [L] * cnt + ([rem] if rem else [])
        ours = _segments(n, L)
        assert [list(range(s, e)) for s, e, _ in ours] == legacy
        assert [w for *_, w in ours] == lw


@pytest.fixture(scope="module")
def encoder():
    return CLaMP2Encoder.load(CLAMP_DIR, device="cpu")


def _legacy_extract(abc_or_text: str, is_text: bool):
    """``extract_feature`` from clamp2/extract_clamp2.py on the original CLaMP2Model."""
    cfg = types.ModuleType("config")
    exec((ROOT / "clamp2" / "config.py").read_text(), cfg.__dict__)
    cfg.M3_WEIGHTS_PATH = "/nonexistent"  # skip the M3 init load; the CLaMP 2 checkpoint overwrites it
    cfg.TEXT_MODEL_NAME = str(CLAMP_DIR / "xlm-roberta-base")
    saved = sys.modules.get("config")
    sys.modules["config"] = cfg
    import transformers

    orig = transformers.AutoModel.from_pretrained
    transformers.AutoModel.from_pretrained = lambda name, **kw: transformers.AutoModel.from_config(
        transformers.AutoConfig.from_pretrained(name)
    )
    try:
        spec = importlib.util.spec_from_file_location("legacy_clamp2_utils", ROOT / "clamp2" / "utils.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m3 = transformers.BertConfig(
            vocab_size=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
        )
        model = mod.CLaMP2Model(m3, text_model_name=cfg.TEXT_MODEL_NAME, load_m3=False).eval()
        ck = torch.load(next(CLAMP_DIR.glob("weights_clamp2_*.pth")), map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"], strict=False)
        tok = transformers.AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
        lines = [l for l in abc_or_text.splitlines(keepends=True) if not (l.startswith("%") and not l.startswith("%%"))]
        item = "".join(lines)
        if is_text:
            item = tok.sep_token.join([c for c in item.split("\n") if c])
            data, L = tok(item, return_tensors="pt")["input_ids"].squeeze(0), 128
        else:
            data, L = torch.tensor(mod.M3Patchilizer().encode(item, add_special_patches=True)), 512
        segs = [data[i : i + L] for i in range(0, len(data), L)]
        segs[-1] = data[-L:]
        outs = []
        for seg in segs:
            m = torch.cat([torch.ones(len(seg)), torch.zeros(L - len(seg))])
            pad = (
                torch.ones(L - len(seg)).long() * tok.pad_token_id if is_text else torch.zeros(L - len(seg), 64).long()
            )
            x = torch.cat([seg, pad])[None]
            with torch.no_grad():
                f = model.get_text_features(x, m[None], True) if is_text else model.get_music_features(x, m[None], True)
            outs.append(f)
        cnt, rem = divmod(len(data), L)
        w = torch.tensor([L] * cnt + ([rem] if rem else []), dtype=torch.float).view(-1, 1)
        return ((torch.cat(outs) * w).sum(0) / w.sum()).numpy()
    finally:
        transformers.AutoModel.from_pretrained = orig
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def test_music_and_text_features_match_legacy(encoder, abc_text):
    long_piece = abc_text + "".join(abc_text.splitlines(keepends=True)[10:]) * 12  # > 512 patches: 2+ segments
    texts = [abc_text, long_piece]
    ours = encoder.music_features(texts)
    for i, t in enumerate(texts):
        np.testing.assert_allclose(ours[i], _legacy_extract(t, False), rtol=1e-4, atol=1e-4)
    q = prompt_text("Classical", "Mozart, Wolfgang Amadeus", "Keyboard")
    np.testing.assert_allclose(encoder.text_features([q])[0], _legacy_extract(q, True), rtol=1e-4, atol=1e-4)


def test_text_retrieval_is_meaningful(encoder):
    """Sanity: a composer's text query is closer to its own name than to unrelated text."""
    t = encoder.text_features(["Mozart, Wolfgang Amadeus", "Mozart", "a recipe for tomato soup"])
    sim = cosine_matrix(t[:1], t[1:])[0]
    assert sim[0] > sim[1]


def test_batching_does_not_change_features(abc_text):
    a = CLaMP2Encoder.load(CLAMP_DIR, device="cpu", batch_size=1).music_features([abc_text, abc_text[:400]])
    b = CLaMP2Encoder.load(CLAMP_DIR, device="cpu", batch_size=8).music_features([abc_text, abc_text[:400]])
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_fixture_exists():
    assert (FIXTURES / "interleaved.abc").exists()


def test_extract_folder_and_prefs(encoder, abc_text, tmp_path):
    from notagen.clamp2 import extract_folder
    from notagen.prefs import build
    from notagen.reward import Prompt

    inter, orig = tmp_path / "interleaved", tmp_path / "original"
    inter.mkdir(), orig.mkdir()
    lines = abc_text.splitlines(keepends=True)
    for i in range(10):
        piece = "".join(lines[:10] + lines[10 + i : 30 + i])
        (inter / f"p{i}.abc").write_text(piece)
        (orig / f"p{i}.abc").write_text(piece)
    assert extract_folder(encoder, inter, tmp_path / "feats") == 10
    assert extract_folder(encoder, inter, tmp_path / "feats") == 0  # existing outputs are skipped
    f = np.load(tmp_path / "feats" / "p3.npy")
    assert f.shape == (1, 768)
    np.testing.assert_allclose(f[0], encoder.music_features([(inter / "p3.abc").read_text()])[0], rtol=1e-5, atol=1e-6)

    info = build(
        inter,
        orig,
        tmp_path / "prefs.json",
        encoder,
        portion=0.2,
        prompt=Prompt.parse("Classical|Mozart, Wolfgang Amadeus|Keyboard"),
    )
    prefs = json.loads((tmp_path / "prefs.json").read_text())
    assert info["chosen"] == 2 and len(prefs["rejected"]) == 2 and not set(prefs["chosen"]) & set(prefs["rejected"])
    info = build(inter, orig, tmp_path / "prefs_gt.json", encoder, portion=0.2, gt_features=tmp_path / "feats")
    assert info["chosen"] == 0 and info["skipped"] == 10  # every piece is a > 0.95 copy of the "ground truth"


def test_scorer_rewards(encoder, abc_text):
    from notagen.reward import Prompt, Scorer

    lines = abc_text.splitlines(keepends=True)
    tagged = [l if not l.startswith("[V:") else f"[r:{k}/0]" + l for k, l in enumerate(lines)]
    good = "".join(tagged)
    scores, feats = Scorer(encoder).score_group(
        [good, good, "garbage"], [True, True, False], Prompt.parse("Classical|Mozart, Wolfgang Amadeus|Keyboard")
    )
    assert scores[0].restored and scores[0].aligned and not np.isnan(scores[0].text_sim)
    assert scores[2].reward == -1.0 and feats.shape == (2, 768)
    assert abs(scores[0].group_sim - 1.0) < 1e-5  # identical pieces
