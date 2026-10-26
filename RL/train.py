"""CLaMP-DPO (DPOP) entry point. Settings stay in ``config.py``; training runs in ``notagen.train``.

Reference log-probs are computed once up front, so only the policy model sits in GPU memory.
Equivalent CLI:  notagen train dpo --help
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

from notagen.stage import model_config_from  # noqa: E402
from notagen.train import DPOTrainConfig, train_dpo  # noqa: E402

if __name__ == "__main__":
    train_dpo(
        DPOTrainConfig(
            data_index=config.DATA_INDEX_PATH,
            init_from=config.PRETRAINED_PATH,
            out_dir=getattr(config, "OUT_DIR", "."),  # legacy location: WEIGHTS_PATH in this dir
            model=model_config_from(config).to_dict(),
            beta=config.BETA,
            lam=config.LAMBDA,
            lr=config.LEARNING_RATE,
            steps=config.OPTIMIZATION_STEPS,
            weights_name=config.WEIGHTS_PATH,
            wandb_project="notagen" if config.WANDB_LOGGING else "",
            batch_size=getattr(config, "BATCH_SIZE", 1),
            precision=getattr(config, "PRECISION", "auto"),
            grad_checkpointing=getattr(config, "GRAD_CHECKPOINTING", False),
        )
    )
