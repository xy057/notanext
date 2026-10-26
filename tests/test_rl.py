"""RL machinery on a tiny model with a toy reward (no CLaMP 2 needed)."""

import json

import numpy as np
import torch
from test_train import _cfg, dataset  # noqa: F401  (fixture)

from notagen.reward import PieceScore
from notagen.rl import RLConfig, piece_loss, train_rl
from notagen.train import train_lm


class ToyScorer:
    """Reward = share of 'c' among generated chars (a stand-in for the CLaMP 2 reward)."""

    def score_group(self, texts, oks, prompt):
        out = []
        for t in texts:
            body = t[len("".join(prompt.lines)) :]
            out.append(PieceScore(ok=True, restored=True, reward=body.count("c") / max(1, len(body))))
        return out, np.zeros((0, 768))


def test_rl_increases_toy_reward(dataset, tmp_path):  # noqa: F811
    pre = tmp_path / "pre"
    train_lm(_cfg(dataset, pre, epochs=3))
    out = tmp_path / "rl"
    cfg = RLConfig(
        init_from=str(pre / "best"),
        out_dir=str(out),
        prompts=["Classical|Mozart, Wolfgang Amadeus|Keyboard"],
        steps=25,
        group_size=8,
        lr=2e-3,
        beta=0.0,
        max_chars=1200,
        precision="fp32",
        device="cpu",
        grad_checkpointing=True,
        temperature=1.0,
    )
    train_rl(cfg, scorer=ToyScorer())
    log = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    first, last = np.mean([r["reward"] for r in log[:5]]), np.mean([r["reward"] for r in log[-5:]])
    assert last > first * 1.3, (first, last)
    assert (out / "final" / "model.safetensors").exists()


def test_kl_is_zero_against_itself(tiny_pair):
    _, _, m = tiny_pair
    patches = [[1] * 15 + [2]] + [[ord(c) for c in "[r:0/3][V:1]abc|"]] * 3
    loss, kl, n = piece_loss(
        m, m, patches, n_prompt=1, adv=0.0, beta=1.0, norm=1.0, device=torch.device("cpu"), precision="fp32"
    )
    assert abs(kl) < 1e-6 and n == 48 and abs(loss.item()) < 1e-6
