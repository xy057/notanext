"""``notagen export``: package weights for release (roadmap Phase 6). Writes local folders only;
uploading is left to the user (e.g. ``huggingface-cli upload``).

    <out>/<name>/            model.safetensors + config.json + README.md (model card)
    <out>/<name>-mlx-bf16/   MLX weights (and -mlx-8bit, -mlx-4bit with --quantize)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .checkpoint import load_model, save_model

CARD = """---
license: mit
library_name: notagen
tags:
- music
- symbolic-music
- abc-notation
{extra_tags}---

# {title}

{summary}

Converted from `{source}` (SHA-256 `{sha}`) with `notagen export`. {fidelity}

| | |
|---|---|
| Architecture | NotaGen hierarchical patch/char decoder (GPT-2 blocks) |
| Patch decoder | {patch_num_layers} layers, hidden {hidden_size}, {patch_length} patches x {patch_size} chars |
| Char decoder | {char_num_layers} layers |
| Parameters | {params:.1f}M |
| Format | {fmt} |

## Usage

```bash
pip install "notagen[mlx]"   # from this repository
notagen generate --weights {usage_path} --prompt "Romantic|Chopin, Frederic|Keyboard" --num-samples 4
```

Prompts are `period|composer|instrumentation`; valid combinations are listed in `gradio/prompts.txt`.
{eval_section}
## Citation

```
@article{{wang2025notagen,
  title={{NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms}},
  author={{Wang, Yashan and Wu, Shangda and Hu, Jianhuai and others}},
  journal={{arXiv preprint arXiv:2502.18008}},
  year={{2025}}
}}
```
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _eval_section(report_path: str | None) -> str:
    if not report_path:
        return ""
    r = json.loads(Path(report_path).read_text())["overall"]
    rows = "\n".join(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in r.items())
    return f"""
## Evaluation (`notagen eval`)

Structure checks and CLaMP 2 metrics on generated pieces (see `docs/MODERNIZATION_ROADMAP.md`):

| metric | value |
|---|---|
{rows}
"""


def export(
    src: str,
    out_dir: str,
    name: str,
    title: str,
    summary: str,
    quantize=(8, 4),
    report: str | None = None,
    trust_pickle=False,
) -> list[Path]:
    src_p = Path(src)
    model = load_model(src_p, trust_pickle=trust_pickle)
    sha = _sha256(src_p) if src_p.is_file() else "(directory)"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    c = model.cfg
    common = dict(
        title=title,
        summary=summary,
        source=src_p.name,
        sha=sha,
        params=params,
        eval_section=_eval_section(report),
        **c.to_dict(),
    )
    outs = []
    base = save_model(model, Path(out_dir) / name)
    same = (
        "The tensors are bit-identical to the original checkpoint; only the container changed "
        "(no optimizer state, no pickle)."
    )
    (base / "README.md").write_text(
        CARD.format(extra_tags="", fmt="safetensors, fp32", usage_path=name, fidelity=same, **common)
    )
    outs.append(base)
    try:
        from .mlx import convert_to_mlx
    except ImportError:
        return outs
    for q in (None, *quantize):
        suffix = "mlx-bf16" if q is None else f"mlx-{q}bit"
        d = convert_to_mlx(model, Path(out_dir) / f"{name}-{suffix}", dtype="bf16", quantize=q)
        fmt = "MLX safetensors, bf16" + ("" if q is None else f", {q}-bit quantized Linear layers (group 64)")
        lossy = (
            "Weights are cast to bf16"
            + ("" if q is None else f" and the transformer Linear layers quantized to {q} bits")
            + "; outputs differ slightly from fp32 (see the roadmap's quality table)."
        )
        (d / "README.md").write_text(
            CARD.format(
                extra_tags="- mlx\n", fmt=fmt, usage_path=f"{name}-{suffix} --backend mlx", fidelity=lossy, **common
            )
        )
        outs.append(d)
    return outs


def main(argv=None):
    ap = argparse.ArgumentParser("notagen export")
    ap.add_argument("src", help="legacy .pth or safetensors dir")
    ap.add_argument("--out", default="weights/export")
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default="NotaGen")
    ap.add_argument(
        "--summary",
        default="Symbolic (ABC) classical music generation conditioned on period, composer and instrumentation.",
    )
    ap.add_argument("--quantize", type=int, nargs="*", default=[8, 4])
    ap.add_argument("--report", help="report.json from notagen eval, added to the model card")
    ap.add_argument("--trust-pickle", action="store_true")
    a = ap.parse_args(argv)
    for d in export(a.src, a.out, a.name, a.title, a.summary, tuple(a.quantize), a.report, a.trust_pickle):
        size = sum(f.stat().st_size for f in d.iterdir()) / 1e9
        print(f"{d}  ({size:.2f} GB)")
