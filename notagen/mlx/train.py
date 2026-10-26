"""LoRA fine-tuning and DPO in MLX (Apple Silicon), roadmap Phase 2b stretch.

``notagen mlx-train finetune``: next-char LM loss on your own augmented ABC data (same ``.jsonl``
index format as ``notagen train finetune``). ``notagen mlx-train dpo``: DPOP on a preference JSON
(``notagen prefs`` / ``RL/data.py``). LoRA adapters go on the transformer Linear layers of both
decoders; with a quantized base (``--quantize 4|8``) this is QLoRA. The reference model for DPO is
the base model itself: its log-probs are computed once before the adapters are attached.

Outputs: ``adapters.safetensors`` + ``adapter_config.json``, and with ``--fuse`` a full MLX model
directory (loadable by ``--backend mlx``). Fusing needs an unquantized base.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from . import _DTYPES, _quantize, load_mlx_model
from .model import NotaGenMLX

TARGETS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        out_dims, w_in = base.weight.shape
        in_dims = w_in * 32 // base.bits if isinstance(base, nn.QuantizedLinear) else w_in
        self.scale = alpha / rank
        self.lora_a = mx.random.uniform(-1, 1, (in_dims, rank)) * (1 / math.sqrt(in_dims))
        self.lora_b = mx.zeros((rank, out_dims))

    def __call__(self, x):
        return self.base(x) + ((x @ self.lora_a.astype(x.dtype)) @ self.lora_b.astype(x.dtype)) * self.scale


def _blocks(model: NotaGenMLX):
    return [*model.patch_decoder.h, *model.char_decoder.h]


def _get(block, path):
    a, b = path.split(".")
    return getattr(getattr(block, a), b)


def _set(block, path, value):
    a, b = path.split(".")
    setattr(getattr(block, a), b, value)


def apply_lora(model: NotaGenMLX, rank: int = 8, alpha: float = 16.0) -> NotaGenMLX:
    model.freeze()
    for blk in _blocks(model):
        for t in TARGETS:
            _set(blk, t, LoRALinear(_get(blk, t), rank, alpha))
    for blk in _blocks(model):
        for t in TARGETS:
            _get(blk, t).unfreeze(keys=["lora_a", "lora_b"], recurse=False)
    return model


def fuse_lora(model: NotaGenMLX) -> NotaGenMLX:
    for blk in _blocks(model):
        for t in TARGETS:
            m = _get(blk, t)
            if not isinstance(m, LoRALinear):
                continue
            if isinstance(m.base, nn.QuantizedLinear):
                raise ValueError("cannot fuse adapters into a quantized base; train on an unquantized base to fuse")
            lin = nn.Linear(m.base.weight.shape[1], m.base.weight.shape[0])
            delta = (m.lora_a @ m.lora_b).T * m.scale
            lin.weight = (m.base.weight.astype(mx.float32) + delta).astype(m.base.weight.dtype)
            lin.bias = m.base.bias
            _set(blk, t, lin)
    return model


def _save(model: NotaGenMLX, out: Path, meta: dict, fuse: bool):
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(meta, indent=2))
    if fuse:
        fuse_lora(model)
        d = out / "fused"
        d.mkdir(exist_ok=True)
        model.save_weights(str(d / "weights.safetensors"))
        (d / "config.json").write_text(
            json.dumps(
                {"model_type": "notagen", **model.cfg.to_dict(), "mlx": {"dtype": meta["dtype"], "quantize": None}},
                indent=2,
            )
        )


def _load_base(path: str, dtype: str, quantize: int | None) -> NotaGenMLX:
    model = load_mlx_model(path, dtype=dtype)
    if quantize:
        _quantize(model, quantize)
    mx.eval(model.parameters())
    return model


def _to_mx(patches, masks, ps):
    return mx.array(patches.reshape(len(patches), -1, ps).numpy()), mx.array(masks.numpy())


def finetune(args) -> Path:
    from torch.utils.data import DataLoader

    from ..tokenizer import Patchilizer
    from ..train import AugmentedABCDataset, collate, read_index

    random.seed(args.seed)
    mx.random.seed(args.seed)
    model = _load_base(args.init_from, args.dtype, args.quantize)
    tok = Patchilizer.from_config(model.cfg, tunebody_detect="contains_v")
    train_entries = read_index(args.train_index)
    eval_entries = read_index(args.eval_index) if args.eval_index else train_entries[: max(1, len(train_entries) // 10)]
    train_dl = DataLoader(
        AugmentedABCDataset(train_entries, tok, "near_original"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        drop_last=True,
    )
    eval_dl = DataLoader(
        AugmentedABCDataset(eval_entries, tok, "near_original"), batch_size=args.batch_size, collate_fn=collate
    )
    apply_lora(model, args.rank, args.alpha)
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    ps = model.cfg.patch_size
    loss_and_grad = nn.value_and_grad(model, lambda m, p, k: m.loss(p, k))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    def evaluate():
        tot, n = 0.0, 0
        for p, k in eval_dl:
            tot += model.loss(*_to_mx(p, k, ps)).item()
            n += 1
        return tot / max(1, n)

    step, best = 0, float("inf")
    base_eval = evaluate()
    log.write(json.dumps({"step": 0, "eval_loss": base_eval}) + "\n")
    print(f"step 0 eval_loss={base_eval:.4f}")
    for epoch in range(1, args.epochs + 1):
        t0, run = time.time(), 0.0
        for p, k in train_dl:
            loss, grads = loss_and_grad(model, *_to_mx(p, k, ps))
            opt.update(model, grads)
            mx.eval(model.trainable_parameters(), opt.state, loss)
            run += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        ev = evaluate()
        rec = {
            "step": step,
            "epoch": epoch,
            "train_loss": run / max(1, len(train_dl)),
            "eval_loss": ev,
            "seconds": time.time() - t0,
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec))
        best = min(best, ev)
        if args.max_steps and step >= args.max_steps:
            break
    meta = {
        "kind": "lora",
        "base": args.init_from,
        "rank": args.rank,
        "alpha": args.alpha,
        "targets": TARGETS,
        "dtype": args.dtype,
        "quantize": args.quantize,
        "base_eval_loss": base_eval,
        "best_eval_loss": best,
    }
    _save(model, out, meta, args.fuse)
    return out


def dpo(args) -> Path:
    from ..tokenizer import Patchilizer
    from ..train import collate

    random.seed(args.seed)
    model = _load_base(args.init_from, args.dtype, args.quantize)
    tok = Patchilizer.from_config(model.cfg, tunebody_detect="startswith_r")
    prefs = json.loads(Path(args.data_index).read_text())
    files = sorted(set(prefs["chosen"]) | set(prefs["rejected"]))
    import torch

    enc = {f: torch.tensor(tok.encode(Path(f).read_text(encoding="utf-8")), dtype=torch.long) for f in files}
    ps = model.cfg.patch_size

    def batch(paths):
        p, k = collate([(enc[f], torch.ones(len(enc[f]), dtype=torch.long)) for f in paths])
        return _to_mx(p, k, ps)

    ref = {f: model.sequence_logps(*batch([f])).item() for f in files}  # base model = reference
    apply_lora(model, args.rank, args.alpha)
    pairs = [(c, r) for c in prefs["chosen"] for r in prefs["rejected"]]

    def loss_fn(m, cp, cm, rp, rm, ref_pos, ref_neg):
        pol_pos, pol_neg = m.sequence_logps(cp, cm), m.sequence_logps(rp, rm)
        logits = (pol_pos - pol_neg) - (ref_pos - ref_neg)
        penalty = mx.maximum(ref_pos - pol_pos, 0)
        return mx.mean(-nn.log_sigmoid(args.beta * (logits - args.lam * penalty))), (pol_pos, pol_neg)

    loss_and_grad = nn.value_and_grad(model, lambda *a: loss_fn(*a)[0])
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        ch, rj = zip(*(pairs[rng.randrange(len(pairs))] for _ in range(args.batch_size)))
        cp, cm = batch(ch)
        rp, rm = batch(rj)
        ref_pos, ref_neg = mx.array([ref[f] for f in ch]), mx.array([ref[f] for f in rj])
        loss, grads = loss_and_grad(model, cp, cm, rp, rm, ref_pos, ref_neg)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        if step % 10 == 0 or step == 1:
            _, (pp, pn) = loss_fn(model, cp, cm, rp, rm, ref_pos, ref_neg)
            margin = (args.beta * ((pp - ref_pos) - (pn - ref_neg))).mean().item()
            rec = {"step": step, "loss": loss.item(), "reward_margin": margin}
            log.write(json.dumps(rec) + "\n")
            log.flush()
    meta = {
        "kind": "lora-dpo",
        "base": args.init_from,
        "rank": args.rank,
        "alpha": args.alpha,
        "targets": TARGETS,
        "dtype": args.dtype,
        "quantize": args.quantize,
        "beta": args.beta,
        "lam": args.lam,
    }
    _save(model, out, meta, args.fuse)
    return out


def load_with_adapters(base: str, adapter_dir: str) -> NotaGenMLX:
    meta = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    model = _load_base(base, meta["dtype"], meta["quantize"])
    apply_lora(model, meta["rank"], meta["alpha"])
    model.load_weights(str(Path(adapter_dir) / "adapters.safetensors"), strict=False)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser("notagen mlx-train")
    sub = ap.add_subparsers(dest="kind", required=True)
    for kind in ("finetune", "dpo"):
        p = sub.add_parser(kind)
        p.add_argument("--init-from", required=True, help="weights (.pth, safetensors dir, or MLX dir)")
        p.add_argument("--out-dir", required=True)
        p.add_argument("--rank", type=int, default=8)
        p.add_argument("--alpha", type=float, default=16.0)
        p.add_argument("--lr", type=float, default=1e-4 if kind == "finetune" else 5e-5)
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--dtype", default="bf16", choices=sorted(_DTYPES))
        p.add_argument("--quantize", type=int, choices=[4, 8], help="QLoRA: quantize the frozen base")
        p.add_argument("--fuse", action="store_true", help="also write a fused full MLX model")
        p.add_argument("--seed", type=int, default=0)
        if kind == "finetune":
            p.add_argument("--train-index", required=True)
            p.add_argument("--eval-index", default="")
            p.add_argument("--epochs", type=int, default=3)
            p.add_argument("--max-steps", type=int, default=0)
        else:
            p.add_argument("--data-index", required=True)
            p.add_argument("--steps", type=int, default=500)
            p.add_argument("--beta", type=float, default=0.1)
            p.add_argument("--lam", type=float, default=10.0)
    a = ap.parse_args(argv)
    print((finetune if a.kind == "finetune" else dpo)(a))
