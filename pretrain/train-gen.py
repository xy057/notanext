"""Pre-training entry point. Settings stay in ``config.py``; training runs in ``notagen.train``.

Single device:   python train-gen.py
Multi-GPU:       torchrun --nproc_per_node 8 train-gen.py   (or: accelerate launch --multi_gpu train-gen.py)
Equivalent CLI:  notagen train pretrain --help
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config  # noqa: E402

from notagen.stage import model_config_from  # noqa: E402
from notagen.train import LMTrainConfig, train_lm  # noqa: E402

if __name__ == "__main__":
    train_lm(
        LMTrainConfig(
            mode="pretrain",
            train_index=config.DATA_TRAIN_INDEX_PATH,
            eval_index=config.DATA_EVAL_INDEX_PATH,
            out_dir=getattr(config, "OUT_DIR", "."),  # legacy location: WEIGHTS_PATH in this dir
            model=model_config_from(config).to_dict(),
            batch_size=config.BATCH_SIZE,
            accumulation_steps=config.ACCUMULATION_STEPS,
            lr=config.LEARNING_RATE,
            epochs=config.NUM_EPOCHS,
            patch_sampling_batch_size=config.PATCH_SAMPLING_BATCH_SIZE,
            resume=config.LOAD_FROM_CHECKPOINT,
            weights_name=config.WEIGHTS_PATH,
            wandb_project="notagen" if config.WANDB_LOGGING else "",
            # New settings (add to config.py to change):
            schedule=getattr(config, "SCHEDULE", "constant"),  # constant | cosine
            warmup_steps=getattr(config, "WARMUP_STEPS", 1000),
            grad_clip=getattr(config, "GRAD_CLIP", 1.0),  # 0 = off, as in the legacy script
            precision=getattr(config, "PRECISION", "auto"),  # auto | bf16 | fp16 | fp32
            grad_checkpointing=getattr(config, "GRAD_CHECKPOINTING", False),
            num_workers=getattr(config, "NUM_WORKERS", 4),
        )
    )
