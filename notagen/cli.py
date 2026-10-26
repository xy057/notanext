"""``notagen`` command line. Replaces editing each stage's ``config.py`` by hand."""

from __future__ import annotations

import argparse
import sys


def _add_sampling(p):
    p.add_argument("--weights", required=True, help="legacy .pth, .safetensors, or a converted model dir")
    p.add_argument("--backend", default="auto", choices=["auto", "torch", "mlx"])
    p.add_argument("--device", default="auto", help="torch backend: auto|cuda|mps|cpu")
    p.add_argument("--dtype", default="auto", help="auto|fp32|bf16|fp16")
    p.add_argument("--sampler", default="fast", choices=["fast", "legacy"])
    p.add_argument("--seed", type=int)
    p.add_argument("--top-k", type=int, default=9)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow full pickle loading of a legacy .pth (only for files you trust)",
    )


def _opts(a):
    from .infer import InferenceOptions

    return InferenceOptions(
        a.weights,
        a.backend,
        a.device,
        a.dtype,
        a.sampler,
        a.seed,
        a.top_k,
        a.top_p,
        a.temperature,
        a.batch_size,
        a.trust_pickle,
    )


def cmd_generate(a):
    from .infer import batch_inference

    prompt = None
    if a.prompt:
        parts = a.prompt.split("|")
        if len(parts) != 3:
            sys.exit('--prompt must be "period|composer|instrumentation"')
        prompt = ["%" + s.strip() + "\n" for s in parts]
    original, interleaved = batch_inference(_opts(a), a.num_samples, a.out_root, prompt)
    print(f"original:    {original}\ninterleaved: {interleaved}")


def cmd_convert(a):
    from .checkpoint import load_model, save_model

    model = load_model(a.src, trust_pickle=a.trust_pickle)
    if a.to == "safetensors":
        print(save_model(model, a.out))
    else:
        from .mlx import convert_to_mlx

        print(convert_to_mlx(model, a.out, dtype=a.dtype, quantize=a.quantize))


def main(argv=None):
    ap = argparse.ArgumentParser("notagen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="batch inference (inference/inference.py)")
    _add_sampling(g)
    g.add_argument("--num-samples", type=int, default=1000)
    g.add_argument("--out-root", default="output", help="writes <out-root>/{original,interleaved}/<name>/")
    g.add_argument("--prompt", help='"period|composer|instrumentation" (fine-tuned / RL / NotaGen-X only)')
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("convert", help="convert weights to safetensors or MLX")
    c.add_argument("src")
    c.add_argument("out")
    c.add_argument("--to", choices=["safetensors", "mlx"], default="safetensors")
    c.add_argument("--dtype", default="bf16", help="mlx: fp32|bf16|fp16")
    c.add_argument("--quantize", type=int, choices=[4, 8], help="mlx: quantize Linear weights")
    c.add_argument("--trust-pickle", action="store_true")
    c.set_defaults(fn=cmd_convert)

    b = sub.add_parser("bench", help="generation throughput benchmark")
    b.set_defaults(fn=None)

    t = sub.add_parser("train", help="pretrain / finetune / dpo (see notagen/train.py)")
    t.set_defaults(fn=None)
    for name, help_ in [
        ("eval", "generate for a prompt set; structure + CLaMP 2 report"),
        ("rl", "online RL from released weights with a CLaMP 2 reward"),
        ("clamp2", "CLaMP 2 feature extraction / scoring"),
        ("prefs", "build a CLaMP-DPO preference set (RL/data.py)"),
        ("export", "package weights + model card for release"),
        ("mlx-train", "LoRA fine-tune / DPO in MLX (Apple Silicon)"),
    ]:
        sub.add_parser(name, help=help_, add_help=False).set_defaults(fn=None)

    a, rest = ap.parse_known_args(argv)
    passthrough = {
        "eval": "evaluate",
        "rl": "rl",
        "clamp2": "clamp2",
        "prefs": "prefs",
        "export": "export",
        "mlx-train": "mlx.train",
    }
    if a.cmd in passthrough:
        import importlib

        return importlib.import_module(f".{passthrough[a.cmd]}", __package__).main(rest)
    if a.cmd == "bench":
        from .bench import main as bench_main

        return bench_main(rest)
    if a.cmd == "train":
        from .train import main as train_main

        return train_main(rest)
    if rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    a.fn(a)


if __name__ == "__main__":
    main()
