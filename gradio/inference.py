"""Demo inference on the shared ``notagen`` package (MLX on Apple Silicon, else PyTorch).

Keeps ``inference_patch(period, composer, instrumentation)`` for ``demo.py``; generated text is
still printed as it streams, which the demo captures from stdout.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

from notagen.infer import InferenceOptions, generate_for_demo, load  # noqa: E402
from notagen.stage import model_config_from  # noqa: E402

_opts = InferenceOptions(
    weights=config.INFERENCE_WEIGHTS_PATH,
    backend=getattr(config, "BACKEND", "auto"),
    top_k=config.TOP_K,
    top_p=config.TOP_P,
    temperature=config.TEMPERATURE,
    model_config=model_config_from(config),
)
_backend = _tok = None


def inference_patch(period, composer, instrumentation):
    global _backend, _tok
    if _backend is None:
        _backend, _tok = load(_opts)
    return generate_for_demo(
        _backend, _tok, period, composer, instrumentation, on_text=lambda t: print(t, end="", flush=True)
    )


if __name__ == "__main__":
    print(inference_patch("Classical", "Beethoven, Ludwig van", "Keyboard"))
