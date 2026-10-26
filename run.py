#!/usr/bin/env python3
"""Launch the NotaGen Gradio demo in one command.

    python run.py                          # finds weights, picks the fastest backend, serves on 127.0.0.1:7861
    python run.py --open                   # ... and opens it in your browser
    python run.py --weights path/to/weights_notagenx_...pth --backend torch
    python run.py --host 0.0.0.0 --port 8000   # reachable from other machines on your network

Weights are looked up in this order: --weights, $NOTAGEN_WEIGHTS, then the first of
    weights/export/notagenx-mlx-bf16 (Apple Silicon), weights/export/notagenx,
    weights/weights_notagenx*.pth, gradio/<INFERENCE_WEIGHTS_PATH from gradio/config.py>.
Nothing is downloaded automatically; if no weights are found, the download link is printed.
"""

from __future__ import annotations

import argparse
import os
import platform
import runpy
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "gradio"
HF_URL = "https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth"


def _apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _config_weights() -> Path | None:
    cfg = DEMO_DIR / "config.py"
    ns: dict = {}
    try:
        exec(cfg.read_text(), ns)
    except Exception:
        return None
    path = ns.get("INFERENCE_WEIGHTS_PATH")
    return (DEMO_DIR / path) if path else None


def find_weights(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("NOTAGEN_WEIGHTS"):
        return Path(os.environ["NOTAGEN_WEIGHTS"]).expanduser().resolve()
    candidates = []
    if _apple_silicon():
        candidates.append(ROOT / "weights/export/notagenx-mlx-bf16")
    candidates.append(ROOT / "weights/export/notagenx")
    candidates += sorted((ROOT / "weights").glob("weights_notagenx*.pth"))
    if (cfg := _config_weights()) is not None:
        candidates.append(cfg)
    return next((c.resolve() for c in candidates if c.exists()), None)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Launch the NotaGen Gradio demo.")
    ap.add_argument("--weights", help="legacy .pth, safetensors folder, or MLX folder (from `notagen export`)")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"],
                    help="auto = MLX on Apple Silicon when installed, else PyTorch")
    ap.add_argument("--dtype", default="auto", help="auto | bf16 | fp16 | fp32")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve on your network")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--open", action="store_true", help="open the demo in your browser")
    a = ap.parse_args(argv)

    weights = find_weights(a.weights)
    if weights is None or not weights.exists():
        missing = f" ({weights})" if weights else ""
        sys.exit(
            f"No NotaGen weights found{missing}.\n"
            f"Download NotaGen-X (6.2 GB) into ./weights/:\n  {HF_URL}\n"
            "or point at your own file:  python run.py --weights /path/to/weights.pth"
        )

    os.environ.update(
        NOTAGEN_WEIGHTS=str(weights),
        NOTAGEN_BACKEND=a.backend,
        NOTAGEN_DTYPE=a.dtype,
        GRADIO_SERVER_NAME=a.host,
        GRADIO_SERVER_PORT=str(a.port),
    )
    url = f"http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}"
    print(f"NotaGen demo\n  weights: {weights}\n  backend: {a.backend}\n  url:     {url}\n", flush=True)
    if a.open:
        threading.Timer(4.0, webbrowser.open, args=(url,)).start()

    # demo.py expects to run from gradio/ (prompts.txt, abc2xml.py, `import inference`). Drop the repo
    # root from sys.path so that `import gradio` resolves to the installed package, not this folder.
    os.chdir(DEMO_DIR)
    sys.path = [str(DEMO_DIR)] + [p for p in sys.path if Path(p or ".").resolve() != ROOT]
    sys.argv = [str(DEMO_DIR / "demo.py")]
    runpy.run_path(str(DEMO_DIR / "demo.py"), run_name="__main__")


if __name__ == "__main__":
    main()
