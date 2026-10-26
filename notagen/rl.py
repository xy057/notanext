"""Online RL from released weights (roadmap Phase 4): group-relative policy gradient with a CLaMP 2 reward.

Each step, for each prompt, the current policy samples a group of ``group_size`` pieces. Every piece
is scored by :class:`notagen.reward.Scorer` (structure checks + CLaMP 2 text/reference fidelity -
within-group similarity), and the advantage is the group-normalized reward (GRPO). The loss is

    sum_i [ -A_i * sum_t log p(t)  +  beta * sum_t KL_t ] / (G * mean tokens)

over the *sampled* chars of each piece (prompt patches excluded), with the k3 per-token KL estimate
against the frozen starting model. The policy is updated once per batch, so the data is on-policy and
no ratio clipping is needed. Generation uses a half-precision copy of the policy, synced every step.

This needs no training data: prompts plus the CLaMP 2 checkpoint are enough. A reference-feature
folder can be added to also reward the paper's CLaMP 2 score.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model, save_model
from .clamp2 import CLaMP2Encoder
from .device import pick_device
from .generation import Generator, TorchBackend
from .reward import Prompt, RewardWeights, Scorer
from .sampling import SamplingParams, TorchSampler
from .tokenizer import Patchilizer
from .train import autocast_ctx, freeze_unused, make_optimizer, resolve_precision


@dataclass
class RLConfig:
    init_from: str = ""
    out_dir: str = "runs/rl"
    clamp2: str = "weights/clamp2"
    prompts: list = field(default_factory=list)  # "period|composer|instrumentation"
    reference: str = ""  # optional folder of ground-truth CLaMP 2 .npy features
    steps: int = 50
    prompts_per_step: int = 1
    group_size: int = 8
    lr: float = 1e-6
    beta: float = 0.04  # KL weight
    grad_clip: float = 1.0
    top_k: int = 9
    top_p: float = 0.9
    temperature: float = 1.2
    max_chars: int = 20000
    precision: str = "auto"
    grad_checkpointing: bool = True
    weights: dict = field(default_factory=lambda: asdict(RewardWeights()))
    seed: int = 0
    device: str = "auto"
    save_every: int = 0


def _gen_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def piece_loss(
    policy, ref, patches: list[list[int]], n_prompt: int, adv: float, beta: float, norm: float, device, precision
):
    """Loss for one sampled piece; returns (loss tensor, kl sum, n tokens)."""
    x = torch.tensor([patches], device=device).reshape(1, -1)
    m = torch.ones(1, len(patches), dtype=torch.long, device=device)
    with autocast_ctx(device, precision):
        lp, mask = policy.token_logps(x, m)
    with torch.no_grad(), autocast_ctx(device, precision):
        rlp, _ = ref.token_logps(x, m)
    keep = mask.clone()
    keep[: max(0, n_prompt - 1)] = False  # targets of prompt patches were not sampled
    lp, rlp = lp[keep], rlp[keep].detach()
    d = rlp - lp
    kl = torch.exp(d) - d - 1
    loss = (-adv * lp.sum() + beta * kl.sum()) / norm
    return loss, kl.sum().item(), int(keep.sum())


def train_rl(cfg: RLConfig, scorer: Scorer | None = None) -> Path:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)
    precision = resolve_precision(cfg.precision, device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rl_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    prompts = [Prompt.parse(p) for p in cfg.prompts]
    if not prompts:
        raise SystemExit("give at least one --prompts 'period|composer|instrumentation'")

    policy = load_model(cfg.init_from, device=device)
    # Copy before enabling checkpointing: deepcopy would otherwise share the patched forward closures.
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    gen_model = copy.deepcopy(ref).to(dtype=_gen_dtype(device)).eval()
    # Dropout stays off (eval mode; gradients still flow): with it on, the policy and the frozen
    # reference disagree before any update and the KL term is pure noise.
    policy.eval()
    freeze_unused(policy)
    if cfg.grad_checkpointing:
        policy.gradient_checkpointing_enable()
    optimizer = make_optimizer(policy, cfg.lr, 0.0, device)

    from .evaluate import load_reference

    centroid, _ = load_reference(cfg.reference or None)
    weights = RewardWeights(**cfg.weights)
    scorer = scorer or Scorer(CLaMP2Encoder.load(cfg.clamp2, device=device), weights, centroid)
    tok = Patchilizer.from_config(policy.cfg)
    sgen = torch.Generator(device=device).manual_seed(cfg.seed)
    backend = TorchBackend(gen_model, TorchSampler(SamplingParams(cfg.top_k, cfg.top_p, cfg.temperature), sgen))
    generator = Generator(backend, tok, max_chars=cfg.max_chars, stream=False)
    log = open(out / "log.jsonl", "a")

    for step in range(1, cfg.steps + 1):
        t0 = time.time()
        step_prompts = [prompts[(step - 1 + k) % len(prompts)] for k in range(cfg.prompts_per_step)]
        rec = {"step": step, "prompts": [str(p) for p in step_prompts]}
        groups = []
        gen_model.eval()
        for p in step_prompts:
            pieces = list(generator.generate(p.lines, cfg.group_size, cfg.group_size))
            scores, _ = scorer.score_group([x.text for x in pieces], [x.ok for x in pieces], p)
            groups.append((p, pieces, scores))
        t_gen = time.time() - t0

        optimizer.zero_grad(set_to_none=True)
        kl_total, tok_total, used = 0.0, 0, 0
        rewards_all = []
        for p, pieces, scores in groups:
            r = np.array([s.reward for s in scores], dtype=np.float64)
            rewards_all += r.tolist()
            std = r.std()
            if std < 1e-6:
                continue  # no learning signal in this group
            adv = (r - r.mean()) / (std + 1e-4)
            n_prompt = len(tok.encode_prompt(p.lines))
            # Dr.GRPO-style constant normalizer: the group's mean sampled-char count, not per-piece length.
            chars = [sum(c != 0 for patch in x.patches[n_prompt:] for c in patch) for x in pieces if x.patches]
            norm = cfg.group_size * len(step_prompts) * max(1.0, float(np.mean(chars)) if chars else 1.0)
            for x, a in zip(pieces, adv):
                if x.patches is None or abs(a) < 1e-8:
                    continue
                loss, kl, nt = piece_loss(policy, ref, x.patches, n_prompt, float(a), cfg.beta, norm, device, precision)
                loss.backward()
                kl_total += kl
                tok_total += nt
                used += 1
        gnorm = float("nan")
        if used:
            gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip).item()
            optimizer.step()
            with torch.no_grad():  # sync the generation copy
                for gp, pp in zip(gen_model.parameters(), policy.parameters()):
                    gp.copy_(pp.to(gp.dtype))

        all_scores = [s for _, _, sc in groups for s in sc]

        def mean(k, all_scores=all_scores):
            v = [getattr(s, k) for s in all_scores if not np.isnan(getattr(s, k))]
            return float(np.mean(v)) if v else float("nan")

        rec.update(
            reward=float(np.mean(rewards_all)),
            text_sim=mean("text_sim"),
            ref_sim=mean("ref_sim"),
            group_sim=mean("group_sim"),
            aligned=float(np.mean([s.aligned for s in all_scores])),
            restored=float(np.mean([s.restored for s in all_scores])),
            kl_per_token=kl_total / max(1, tok_total),
            grad_norm=gnorm,
            pieces_used=used,
            t_gen=t_gen,
            t_step=time.time() - t0,
        )
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if cfg.save_every and step % cfg.save_every == 0:
            save_model(policy, out / f"step{step}")

    log.close()
    return save_model(policy, out / "final")


def main(argv=None):
    ap = argparse.ArgumentParser("notagen rl")
    ap.add_argument("--config", help="JSON with any RLConfig fields")
    for name, f in RLConfig.__dataclass_fields__.items():
        if name in ("weights", "prompts"):
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(flag, type=type(f.default), default=None)
    ap.add_argument("--prompts", nargs="+")
    for name in RewardWeights.__dataclass_fields__:
        ap.add_argument("--w-" + name, type=float, default=None, help=f"reward weight '{name}'")
    a = ap.parse_args(argv)
    kw = json.loads(Path(a.config).read_text()) if a.config else {}
    weights = {**asdict(RewardWeights()), **kw.get("weights", {})}
    for name in RewardWeights.__dataclass_fields__:
        v = getattr(a, "w_" + name)
        if v is not None:
            weights[name] = v
    for k, v in vars(a).items():
        if v is not None and k != "config" and not k.startswith("w_"):
            kw[k] = v
    kw["weights"] = weights
    if not kw.get("prompts"):
        from .evaluate import KEYBOARD_8

        kw["prompts"] = KEYBOARD_8
    return train_rl(RLConfig(**kw))
