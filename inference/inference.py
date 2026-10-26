"""Batch inference. Settings stay in ``config.py``; the work is done by ``notagen.infer``.

Equivalent CLI:  notagen generate --weights <path> --num-samples 1000 --out-root ../output
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

from notagen.infer import InferenceOptions, batch_inference  # noqa: E402
from notagen.stage import model_config_from  # noqa: E402

# New optional settings; add them to config.py to override.
BACKEND = getattr(config, "BACKEND", "auto")  # auto | torch | mlx
BATCH_SIZE = getattr(config, "BATCH_SIZE", 8)  # pieces generated in parallel
SAMPLER = getattr(config, "SAMPLER", "fast")  # fast | legacy (bit-compatible NumPy sampling)
PROMPT_LINES = getattr(config, "PROMPT_LINES", [])  # e.g. ['%Romantic\n', '%Chopin, Frederic\n', '%Keyboard\n']


def inference_patch(prompt_lines=PROMPT_LINES, pieces=config.NUM_SAMPLES):
    opts = InferenceOptions(
        weights=config.INFERENCE_WEIGHTS_PATH,
        backend=BACKEND,
        sampler=SAMPLER,
        top_k=config.TOP_K,
        top_p=config.TOP_P,
        temperature=config.TEMPERATURE,
        batch_size=BATCH_SIZE,
        model_config=model_config_from(config),
    )
    return batch_inference(opts, pieces, out_root="../output", prompt_lines=list(prompt_lines))


if __name__ == "__main__":
    inference_patch()
