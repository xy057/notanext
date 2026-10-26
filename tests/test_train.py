import json
import shutil

import pytest
import torch
from conftest import TINY

from notagen.train import PRETRAIN_KEYS, DPOTrainConfig, LMTrainConfig, dpop_loss, lr_lambda, train_dpo, train_lm


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, abc_text):
    """An augmented-data layout: <folder>/<key>/<name>_<key>.abc for all 15 keys + a .jsonl index."""
    root = tmp_path_factory.mktemp("data")
    lines = abc_text.splitlines(keepends=True)
    head, body = lines[:10], lines[10:]
    index = []
    for i in range(12):
        name = f"piece{i}"
        text = "".join(head + body[i : i + 20 + i])
        for key in PRETRAIN_KEYS:
            (root / key).mkdir(exist_ok=True)
            (root / key / f"{name}_{key}.abc").write_text(text)
        index.append({"path": str(root / name), "key": "G"})
    (root / "train.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index[:10]))
    (root / "eval.jsonl").write_text("".join(json.dumps(e) + "\n" for e in index[10:]))
    return root


def _cfg(dataset, out, **kw):
    base = dict(
        mode="pretrain",
        train_index=str(dataset / "train.jsonl"),
        eval_index=str(dataset / "eval.jsonl"),
        out_dir=str(out),
        model=TINY.to_dict(),
        batch_size=2,
        lr=3e-3,
        epochs=3,
        warmup_steps=5,
        schedule="cosine",
        precision="fp32",
        num_workers=0,
        device="cpu",
    )
    base.update(kw)
    return LMTrainConfig(**base)


def test_warmup_reaches_the_optimizer_in_use():
    f = lr_lambda("constant", 1000, 10_000, 0.1)
    assert f(0) == pytest.approx(1 / 1000) and f(999) == 1.0 and f(5000) == 1.0
    g = lr_lambda("cosine", 10, 110, 0.1)
    assert g(10) == pytest.approx(1.0) and g(110) == pytest.approx(0.1)


def test_pretrain_learns_and_resumes(dataset, tmp_path):
    out = tmp_path / "pre"
    train_lm(_cfg(dataset, out))
    log = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    evals = [r["eval_loss"] for r in log if "eval_loss" in r]
    assert len(evals) == 3 and evals[-1] < evals[0]
    assert (out / "best" / "model.safetensors").exists() and (out / "last.pth").exists()
    first = [r for r in log if "lr" in r][0]
    assert first["lr"] == pytest.approx(3e-3 / 5)  # warmup applied from step 1

    ck = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
    train_lm(_cfg(dataset, out, epochs=4, resume=True))
    ck2 = torch.load(out / "last.pth", map_location="cpu", weights_only=False)
    assert ck2["epoch"] == 4 and ck2["step"] > ck["step"]
    # legacy loaders read ``checkpoint['model']``
    from notagen.checkpoint import load_model

    shutil.copy(
        out / "last.pth", tmp_path / "weights_notagen_x_p_size_16_p_length_48_p_layers_2_c_layers_2_h_size_128.pth"
    )
    load_model(
        tmp_path / "weights_notagen_x_p_size_16_p_length_48_p_layers_2_c_layers_2_h_size_128.pth", trust_pickle=True
    )


def test_finetune_from_pretrained(dataset, tmp_path):
    pre = tmp_path / "pre"
    train_lm(_cfg(dataset, pre, epochs=1))
    ft = tmp_path / "ft"
    train_lm(
        _cfg(
            dataset,
            ft,
            mode="finetune",
            init_from=str(pre / "best"),
            epochs=1,
            lr=1e-3,
            accumulation_steps=2,
            grad_checkpointing=True,
        )
    )
    assert (ft / "best" / "model.safetensors").exists()


def test_dpop_loss_matches_legacy_formula():
    pp, pn, rp, rn = (torch.tensor([v]) for v in (-10.0, -12.0, -9.0, -11.5))
    legacy = -torch.nn.functional.logsigmoid(0.1 * (((pp - pn) - (rp - rn)) - 10 * max(0, rp - pp)))
    torch.testing.assert_close(dpop_loss(pp, pn, rp, rn, 0.1, 10.0), legacy)


def test_dpo_runs_and_moves_toward_chosen(dataset, tmp_path, abc_text):
    pre = tmp_path / "pre"
    train_lm(_cfg(dataset, pre, epochs=1))
    from notagen.tokenizer import Patchilizer

    tok = Patchilizer(patch_length=10_000)
    tagged = tok.decode(tok.encode_train(abc_text)[:-1]).replace(chr(1), "")
    chosen, rejected = tmp_path / "c.abc", tmp_path / "r.abc"
    chosen.write_text(tagged)
    rejected.write_text(tagged.replace("|", "||").replace("C", "E"))
    (tmp_path / "pref.json").write_text(json.dumps({"chosen": [str(chosen)], "rejected": [str(rejected)]}))
    out = tmp_path / "dpo"
    train_dpo(
        DPOTrainConfig(
            data_index=str(tmp_path / "pref.json"),
            init_from=str(pre / "best"),
            out_dir=str(out),
            model=TINY.to_dict(),
            steps=20,
            lr=1e-4,
            precision="fp32",
            device="cpu",
            batch_size=2,
        )
    )
    log = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert log[-1]["reward_margin"] > log[0]["reward_margin"]
    assert (out / "final" / "model.safetensors").exists()
