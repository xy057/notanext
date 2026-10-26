"""Training: pre-training / fine-tuning (next-char LM) and CLaMP-DPO (DPOP), roadmap Phase 3.

Differences from the legacy ``pretrain/``, ``finetune/`` and ``RL/`` scripts:

* the LR schedule actually applies. The legacy scripts built the scheduler on one AdamW instance
  and then replaced ``optimizer`` with a new one, so warmup never reached the optimizer in use;
* ``torch.amp`` autocast on any device: bf16 needs no loss scaling, and fp16 uses ``GradScaler``;
* optional cosine decay, gradient clipping, fused AdamW, and gradient checkpointing;
* data loading runs in DataLoader worker processes (encoding stays on the fly because stream-mode
  cuts and key transposition are random per draw);
* checkpoints resume exactly (model, optimizer, scheduler, scaler, step, RNG). They are written
  in the legacy ``.pth`` layout so old tools still read them, plus a ``safetensors`` export of
  the best weights;
* DPO pre-computes reference log-probs once (ref model in eval mode, then freed), supports
  batch > 1, and logs reward margins.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import load_model, read_legacy_state_dict, save_model
from .config import PRESETS, ModelConfig
from .device import pick_device
from .model import NotaGenLMHeadModel
from .tokenizer import Patchilizer

# ----------------------------------------------------------------------------- data

PRETRAIN_KEYS = ["C#", "F#", "B", "E", "A", "D", "G", "C", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]


def _finetune_key(mode_key: str, rng: random.Random) -> str:
    """Fine-tune transposition: within +-3 semitones of the original key, peaked at 0 (legacy)."""
    from abctoolkit.transpose import Key2index, Key2Mode

    index2key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
    mode2key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list}
    ori_index = Key2index[mode2key[mode_key]]
    available = [(ori_index + offset) % 12 for offset in range(-3, 4)]
    probs = [1 / 16, 2 / 16, 3 / 16, 4 / 16, 3 / 16, 2 / 16, 1 / 16]
    edges = [0] + [sum(probs[: i + 1]) for i in range(len(probs))]
    r = rng.random()
    des = available[-1]
    for i in range(len(edges) - 1):
        if edges[i] <= r < edges[i + 1]:
            des = available[i]
    if des == 1:
        return "Db" if rng.random() < 0.8 else "C#"
    if des == 11:
        return "B" if rng.random() < 0.8 else "Cb"
    if des == 6:
        return "F#" if rng.random() < 0.5 else "Gb"
    return index2key[des]


class AugmentedABCDataset(Dataset):
    """Rows of a ``.jsonl`` index (``{"path": ..., "key": ...}``) -> one transposed, encoded piece."""

    def __init__(self, entries, patchilizer: Patchilizer, key_sampling: str):
        self.entries, self.tok, self.key_sampling = entries, patchilizer, key_sampling

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        key = random.choice(PRETRAIN_KEYS) if self.key_sampling == "uniform" else _finetune_key(e["key"], random)
        folder, name = os.path.dirname(e["path"]), os.path.split(e["path"])[-1]
        with open(os.path.join(folder, key, name + "_" + key + ".abc"), encoding="utf-8") as f:
            ids = self.tok.encode_train(f.read())
        return torch.tensor(ids, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)


def collate(batch):
    patches, masks = zip(*batch)
    return (
        torch.nn.utils.rnn.pad_sequence(patches, batch_first=True, padding_value=0),
        torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0),
    )


def read_index(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ----------------------------------------------------------------------------- shared setup


@dataclass
class Runtime:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world: int = 1

    @property
    def main(self):
        return self.rank == 0


def setup_runtime(device="auto") -> Runtime:
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        local = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dev = torch.device("cuda", local)
            dist.init_process_group("nccl")
        else:
            dev = torch.device("cpu")
            dist.init_process_group("gloo")
        return Runtime(dev, int(os.environ["RANK"]), local, world)
    return Runtime(pick_device(device))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autocast_ctx(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device.type, dtype=dtype)


def resolve_precision(precision: str, device: torch.device) -> str:
    if precision != "auto":
        return precision
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if device.type == "mps":
        return "bf16"
    return "fp32"


def make_optimizer(model, lr: float, weight_decay: float, device: torch.device):
    params = [p for p in model.parameters() if p.requires_grad]
    kw = {"fused": True} if device.type in ("cuda", "mps") else {}
    try:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kw)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def lr_lambda(schedule: str, warmup: int, total: int, min_ratio: float):
    def f(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def freeze_unused(model: NotaGenLMHeadModel):
    """The patch decoder's one-row ``wte`` is never used; freezing it lets DDP skip
    ``find_unused_parameters`` (the legacy scripts paid for that search every step)."""
    model.patch_level_decoder.base.wte.weight.requires_grad_(False)


class Logger:
    def __init__(self, out_dir: Path, enabled: bool, wandb_project: str | None, run_name: str, config: dict):
        self.enabled = enabled
        self.file = out_dir / "log.jsonl" if enabled else None
        self.wandb = None
        if enabled and wandb_project:
            import wandb

            self.wandb = wandb.init(project=wandb_project, name=run_name, config=config)

    def log(self, step: int, **metrics):
        if not self.enabled:
            return
        with open(self.file, "a") as f:
            f.write(json.dumps({"step": step, "time": time.time(), **metrics}) + "\n")
        if self.wandb:
            self.wandb.log(metrics, step=step)


# ----------------------------------------------------------------------------- LM training


@dataclass
class LMTrainConfig:
    mode: str = "finetune"  # pretrain | finetune
    train_index: str = ""
    eval_index: str = ""
    out_dir: str = "runs/finetune"
    model: dict = field(default_factory=lambda: PRESETS["large"].to_dict())
    init_from: str = ""  # finetune: pretrained weights (.pth / safetensors dir)
    batch_size: int = 1
    accumulation_steps: int = 1
    lr: float = 1e-5
    weight_decay: float = 0.01  # torch AdamW default, as in the legacy scripts
    epochs: int = 64
    warmup_steps: int = 1000
    schedule: str = "constant"  # constant (legacy intent) | cosine
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0  # 0 disables (legacy LM scripts did not clip)
    precision: str = "auto"  # auto | bf16 | fp16 | fp32
    grad_checkpointing: bool = False
    patch_sampling_batch_size: int = 0
    num_workers: int = 4
    seed: int = 0
    eval_split: float = 0.1  # used when eval_index is empty
    resume: bool = False
    weights_name: str = ""  # legacy-style checkpoint filename (default: best.pth)
    wandb_project: str = ""
    device: str = "auto"
    max_steps: int = 0  # stop early (smoke tests)


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _init_lm_model(cfg: LMTrainConfig, mcfg: ModelConfig, rt: Runtime) -> NotaGenLMHeadModel:
    if cfg.mode == "finetune" or cfg.init_from:
        if not cfg.init_from:
            raise SystemExit("finetune needs --init-from (pretrained weights)")
        model = load_model(cfg.init_from, cfg=mcfg if Path(cfg.init_from).suffix == ".pth" else None)
        model.train()
    else:
        model = NotaGenLMHeadModel(mcfg)
    return model


def train_lm(cfg: LMTrainConfig) -> Path:
    rt = setup_runtime(cfg.device)
    seed_all(cfg.seed + rt.rank)
    out = Path(cfg.out_dir)
    if rt.main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    mcfg = ModelConfig.from_dict(cfg.model)
    tok = Patchilizer.from_config(mcfg, tunebody_detect="startswith_v" if cfg.mode == "pretrain" else "contains_v")
    train_entries = read_index(cfg.train_index)
    eval_entries = read_index(cfg.eval_index) if cfg.eval_index else []
    if not eval_entries:
        random.Random(cfg.seed).shuffle(train_entries)
        n_eval = max(1, int(len(train_entries) * cfg.eval_split))
        eval_entries, train_entries = train_entries[:n_eval], train_entries[n_eval:]

    key_sampling = "uniform" if cfg.mode == "pretrain" else "near_original"
    train_ds = AugmentedABCDataset(train_entries, tok, key_sampling)
    eval_ds = AugmentedABCDataset(eval_entries, tok, key_sampling)
    train_sampler = DistributedSampler(train_ds, rt.world, rt.rank, seed=cfg.seed) if rt.world > 1 else None
    eval_sampler = DistributedSampler(eval_ds, rt.world, rt.rank, shuffle=False) if rt.world > 1 else None
    loader_kw = dict(
        batch_size=cfg.batch_size,
        collate_fn=collate,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )
    train_dl = DataLoader(train_ds, sampler=train_sampler, shuffle=train_sampler is None, **loader_kw)
    eval_dl = DataLoader(eval_ds, sampler=eval_sampler, shuffle=False, **{**loader_kw, "drop_last": False})

    model = _init_lm_model(cfg, mcfg, rt).to(rt.device)
    freeze_unused(model)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    precision = resolve_precision(cfg.precision, rt.device)
    optimizer = make_optimizer(model, cfg.lr, cfg.weight_decay, rt.device)
    steps_per_epoch = max(1, len(train_dl) // cfg.accumulation_steps)
    total_steps = cfg.max_steps or steps_per_epoch * cfg.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(cfg.schedule, cfg.warmup_steps, total_steps, cfg.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler(rt.device.type, enabled=precision == "fp16")

    state = {"epoch": 0, "step": 0, "best_epoch": 0, "min_eval_loss": float("inf")}
    last_path = out / "last.pth"
    if cfg.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(read_legacy_state_dict(last_path, trust_pickle=True))
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["lr_sched"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        state.update({k: ck[k] for k in state if k in ck})
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        if rt.main:
            print(f"Resumed from epoch {state['epoch']} (step {state['step']})")

    ddp = model
    if rt.world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[rt.local_rank] if rt.device.type == "cuda" else None
        )

    logger = Logger(out, rt.main, cfg.wandb_project or None, out.name, asdict(cfg))
    if rt.main:
        n = sum(p.numel() for p in model.parameters())
        print(f"params={n / 1e6:.1f}M device={rt.device} precision={precision} steps/epoch={steps_per_epoch}")

    def reduce_mean(x: float) -> float:
        if rt.world == 1:
            return x
        t = torch.tensor([x], device=rt.device)
        dist.all_reduce(t)
        return t.item() / rt.world

    def run_eval() -> float:
        ddp.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for patches, masks in eval_dl:
                with autocast_ctx(rt.device, precision):
                    loss = ddp(patches.to(rt.device), masks.to(rt.device))
                total += loss.item()
                n += 1
        ddp.train()
        return reduce_mean(total / max(1, n))

    best_name = cfg.weights_name or "best.pth"
    stop = False
    for epoch in range(state["epoch"] + 1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ddp.train()
        running, n_micro, t0 = 0.0, 0, time.time()
        for i, (patches, masks) in enumerate(train_dl):
            patches, masks = patches.to(rt.device, non_blocking=True), masks.to(rt.device, non_blocking=True)
            boundary = (i + 1) % cfg.accumulation_steps == 0
            sync = ddp.no_sync() if rt.world > 1 and not boundary else _nullcontext()
            with sync, autocast_ctx(rt.device, precision):
                loss = ddp(patches, masks, cfg.patch_sampling_batch_size) / cfg.accumulation_steps
            scaler.scale(loss).backward()
            running += loss.item()
            n_micro += 1
            if not boundary:
                continue
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            else:
                gnorm = float("nan")
            lr_used = scheduler.get_last_lr()[0]
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % 10 == 0 or state["step"] == 1:
                logger.log(
                    state["step"],
                    train_loss=running * cfg.accumulation_steps / n_micro,
                    lr=lr_used,
                    grad_norm=gnorm,
                    epoch=epoch,
                    seqs_per_s=n_micro * cfg.batch_size * rt.world / (time.time() - t0),
                )
            if cfg.max_steps and state["step"] >= cfg.max_steps:
                stop = True
                break

        train_loss = reduce_mean(running * cfg.accumulation_steps / max(1, n_micro))
        eval_loss = run_eval()
        state["epoch"] = epoch
        if rt.main:
            logger.log(state["step"], epoch=epoch, epoch_train_loss=train_loss, eval_loss=eval_loss)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} eval_loss={eval_loss:.4f}")
            ck = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "model_config": mcfg.to_dict(),
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
                **state,
            }
            if eval_loss < state["min_eval_loss"]:
                state["best_epoch"], state["min_eval_loss"] = epoch, eval_loss
                ck.update(best_epoch=epoch, min_eval_loss=eval_loss)
                torch.save(ck, out / best_name)
                save_model(model, out / "best")
            torch.save(ck, last_path)
        if rt.world > 1:
            dist.barrier()
        if stop:
            break

    if rt.main:
        print(f"best epoch {state['best_epoch']} eval_loss={state['min_eval_loss']:.4f} -> {out / 'best'}")
    if rt.world > 1:
        dist.destroy_process_group()
    return out / "best"


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------- DPO


@dataclass
class DPOTrainConfig:
    data_index: str = ""  # {"chosen": [...], "rejected": [...]} from RL/data.py
    init_from: str = ""
    out_dir: str = "runs/dpo"
    model: dict = field(default_factory=lambda: PRESETS["large"].to_dict())
    beta: float = 0.1
    lam: float = 10.0  # DPOP positive-likelihood penalty
    lr: float = 1e-6
    steps: int = 10000
    batch_size: int = 1
    grad_clip: float = 1.0
    precision: str = "auto"
    grad_checkpointing: bool = False
    seed: int = 0
    weights_name: str = ""
    wandb_project: str = ""
    device: str = "auto"


def dpop_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta, lam):
    """DPO with the DPOP penalty, per pair (legacy ``RL/train.py`` objective)."""
    logits = (pol_pos - pol_neg) - (ref_pos - ref_neg)
    penalty = torch.clamp(ref_pos - pol_pos, min=0)
    return -F.logsigmoid(beta * (logits - lam * penalty))


def _per_sequence_logps(model: NotaGenLMHeadModel, patches, masks):
    """Summed log-prob per sequence for a right-padded batch. Returns ``[b]``."""
    return torch.stack(
        [
            model.sequence_logps(
                patches[i : i + 1, : int(masks[i].sum()) * model.cfg.patch_size],
                masks[i : i + 1, : int(masks[i].sum())],
            )
            for i in range(len(patches))
        ]
    )


def train_dpo(cfg: DPOTrainConfig) -> Path:
    rt = setup_runtime(cfg.device)
    seed_all(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    mcfg = ModelConfig.from_dict(cfg.model)
    tok = Patchilizer.from_config(mcfg, tunebody_detect="startswith_r")
    prefs = json.loads(Path(cfg.data_index).read_text())
    precision = resolve_precision(cfg.precision, rt.device)

    # Encode each file once (fixing any random stream cut), then score it with the frozen reference.
    files = sorted(set(prefs["chosen"]) | set(prefs["rejected"]))
    encoded = {}
    for path in files:
        try:
            encoded[path] = torch.tensor(tok.encode(Path(path).read_text(encoding="utf-8")), dtype=torch.long)
        except Exception as e:  # unreadable / unencodable files are skipped, as before
            print(f"skip {path}: {e}")
    ref = load_model(cfg.init_from, cfg=mcfg if Path(cfg.init_from).suffix == ".pth" else None, device=rt.device).eval()
    ref_logps = {}
    with torch.no_grad(), autocast_ctx(rt.device, precision):
        for path, ids in encoded.items():
            ref_logps[path] = (
                ref.sequence_logps(ids[None].to(rt.device), torch.ones(1, len(ids), dtype=torch.long, device=rt.device))
                .float()
                .item()
            )
    del ref
    if rt.device.type == "cuda":
        torch.cuda.empty_cache()

    pairs = [(c, r) for c in prefs["chosen"] for r in prefs["rejected"] if c in encoded and r in encoded]
    if not pairs:
        raise SystemExit("no usable preference pairs")
    policy = load_model(cfg.init_from, cfg=mcfg if Path(cfg.init_from).suffix == ".pth" else None, device=rt.device)
    policy.train()
    freeze_unused(policy)
    if cfg.grad_checkpointing:
        policy.gradient_checkpointing_enable()
    optimizer = make_optimizer(policy, cfg.lr, 0.01, rt.device)
    scaler = torch.amp.GradScaler(rt.device.type, enabled=precision == "fp16")
    logger = Logger(out, True, cfg.wandb_project or None, out.name, asdict(cfg))
    rng = random.Random(cfg.seed)

    def batch_of(paths):
        seqs = [encoded[p] for p in paths]
        patches, masks = collate([(s, torch.ones(len(s), dtype=torch.long)) for s in seqs])
        return (
            patches.reshape(len(seqs), -1).to(rt.device),
            masks.to(rt.device),
            torch.tensor([ref_logps[p] for p in paths], device=rt.device),
        )

    running = 0.0
    for step in range(1, cfg.steps + 1):
        chosen, rejected = zip(*(pairs[rng.randrange(len(pairs))] for _ in range(cfg.batch_size)))
        cp, cm, ref_pos = batch_of(chosen)
        rp, rm, ref_neg = batch_of(rejected)
        with autocast_ctx(rt.device, precision):
            pol_pos = _per_sequence_logps(policy, cp, cm).float()
            pol_neg = _per_sequence_logps(policy, rp, rm).float()
        loss = dpop_loss(pol_pos, pol_neg, ref_pos, ref_neg, cfg.beta, cfg.lam).mean()
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running += loss.item()
        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                margin = (cfg.beta * ((pol_pos - ref_pos) - (pol_neg - ref_neg))).mean().item()
                penalty = torch.clamp(ref_pos - pol_pos, min=0).mean().item()
            logger.log(
                step,
                train_loss=running / step,
                reward_margin=margin,
                dpop_penalty=penalty,
                chosen_logp=pol_pos.mean().item(),
                rejected_logp=pol_neg.mean().item(),
            )

    torch.save({"model": policy.state_dict(), "model_config": mcfg.to_dict()}, out / (cfg.weights_name or "dpo.pth"))
    return save_model(policy, out / "final")


# ----------------------------------------------------------------------------- CLI


def _parser_from_dataclass(dc, prog):
    ap = argparse.ArgumentParser(prog)
    ap.add_argument("--config", help="JSON file with any of the options below")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="model size (overrides config 'model')")
    for name, f in dc.__dataclass_fields__.items():
        if name == "model":
            continue
        default = f.default
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(flag, type=type(default) if default is not None else str, default=None)
    return ap


def _build(dc, args):
    kw = json.loads(Path(args.config).read_text()) if args.config else {}
    kw.update({k: v for k, v in vars(args).items() if v is not None and k not in ("config", "preset")})
    if args.preset:
        kw["model"] = PRESETS[args.preset].to_dict()
    return dc(**kw)


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("pretrain", "finetune", "dpo"):
        raise SystemExit("usage: notagen train {pretrain,finetune,dpo} [--options]  (see --help)")
    kind, rest = argv[0], argv[1:]
    if kind == "dpo":
        cfg = _build(DPOTrainConfig, _parser_from_dataclass(DPOTrainConfig, "notagen train dpo").parse_args(rest))
        return train_dpo(cfg)
    args = _parser_from_dataclass(LMTrainConfig, f"notagen train {kind}").parse_args(rest)
    cfg = _build(LMTrainConfig, args)
    cfg.mode = kind
    return train_lm(cfg)


if __name__ == "__main__":
    main()
