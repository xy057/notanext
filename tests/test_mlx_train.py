"""MLX LoRA fine-tuning / DPO on a tiny model (CPU stream for determinism)."""

import json
from argparse import Namespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_train import _cfg, dataset  # noqa: E402,F401

from notagen.mlx import load_mlx_model  # noqa: E402
from notagen.mlx.train import apply_lora, dpo, finetune, fuse_lora, load_with_adapters  # noqa: E402
from notagen.train import train_lm  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mlx_cpu():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(prev)


@pytest.fixture(scope="module")
def pretrained(dataset, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("pre")
    train_lm(_cfg(dataset, out, epochs=2))
    return out / "best"


def _ids():
    p = np.zeros((1, 4, 16), dtype=np.int32)
    p[0, 0] = [1] * 15 + [2]
    for j, s in enumerate(["%Classical\n", "[r:0/3][V:1]abc|", "[V:2]C,8|\n"], 1):
        p[0, j, : len(s)] = [ord(c) for c in s]
    return mx.array(p), mx.ones((1, 4), dtype=mx.int32)


def test_lora_starts_as_identity_and_fuses_exactly(pretrained):
    base = load_mlx_model(pretrained, dtype="fp32")
    x, m = _ids()
    want = base.sequence_logps(x, m).item()
    model = apply_lora(load_mlx_model(pretrained, dtype="fp32"), rank=4)
    assert model.sequence_logps(x, m).item() == pytest.approx(want, abs=1e-5)
    for blk in model.patch_decoder.h:  # perturb the adapters, then fuse
        blk.attn.c_attn.lora_b = mx.random.normal(blk.attn.c_attn.lora_b.shape) * 0.01
    lora_val = model.sequence_logps(x, m).item()
    assert lora_val != pytest.approx(want, abs=1e-3)
    assert fuse_lora(model).sequence_logps(x, m).item() == pytest.approx(lora_val, rel=1e-5)


def test_mlx_lora_finetune_lowers_eval_loss_and_reloads(dataset, pretrained, tmp_path):  # noqa: F811
    args = Namespace(
        init_from=str(pretrained),
        out_dir=str(tmp_path / "ft"),
        rank=8,
        alpha=16.0,
        lr=3e-3,
        batch_size=2,
        dtype="fp32",
        quantize=None,
        fuse=True,
        seed=0,
        train_index=str(dataset / "train.jsonl"),
        eval_index=str(dataset / "eval.jsonl"),
        epochs=3,
        max_steps=0,
    )
    out = finetune(args)
    meta = json.loads((out / "adapter_config.json").read_text())
    assert meta["best_eval_loss"] < meta["base_eval_loss"]
    x, m = _ids()
    a = load_with_adapters(str(pretrained), str(out)).sequence_logps(x, m).item()
    b = load_mlx_model(out / "fused").sequence_logps(x, m).item()
    assert a == pytest.approx(b, rel=1e-4)


def test_qlora_trains_on_quantized_base(dataset, pretrained, tmp_path):  # noqa: F811
    args = Namespace(
        init_from=str(pretrained),
        out_dir=str(tmp_path / "q"),
        rank=4,
        alpha=8.0,
        lr=3e-3,
        batch_size=2,
        dtype="fp32",
        quantize=8,
        fuse=False,
        seed=0,
        train_index=str(dataset / "train.jsonl"),
        eval_index=str(dataset / "eval.jsonl"),
        epochs=2,
        max_steps=0,
    )
    meta = json.loads((finetune(args) / "adapter_config.json").read_text())
    assert meta["best_eval_loss"] < meta["base_eval_loss"]


def test_mlx_dpo_increases_margin(pretrained, tmp_path, abc_text):
    from notagen.tokenizer import Patchilizer

    tok = Patchilizer(patch_length=10_000)
    tagged = tok.decode(tok.encode_train(abc_text)[:-1]).replace(chr(1), "")
    (tmp_path / "c.abc").write_text(tagged)
    (tmp_path / "r.abc").write_text(tagged.replace("|", "||").replace("C", "E"))
    (tmp_path / "p.json").write_text(
        json.dumps({"chosen": [str(tmp_path / "c.abc")], "rejected": [str(tmp_path / "r.abc")]})
    )
    args = Namespace(
        init_from=str(pretrained),
        out_dir=str(tmp_path / "dpo"),
        rank=8,
        alpha=16.0,
        lr=1e-3,
        batch_size=1,
        dtype="fp32",
        quantize=None,
        fuse=False,
        seed=0,
        data_index=str(tmp_path / "p.json"),
        steps=20,
        beta=0.1,
        lam=10.0,
    )
    out = dpo(args)
    log = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert log[-1]["reward_margin"] > log[0]["reward_margin"]
