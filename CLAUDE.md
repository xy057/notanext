# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NotaGen is a research codebase for symbolic (ABC notation) classical music generation, trained in three stages: pre-training → fine-tuning (with `period-composer-instrumentation` prompts) → reinforcement learning via **CLaMP-DPO** (DPO where preference pairs are ranked by CLaMP 2 similarity to ground truth). Paper: arXiv 2502.18008. Weights live on HuggingFace (`ElectricAlexis/NotaGen`) and are not in the repo. The modernization plan and its status are in `docs/MODERNIZATION_ROADMAP.md`.

## Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[mlx,demo,train,dev]"
.venv/bin/python -m pytest          # ~45 tests, ~20 s, no weights or GPU needed
.venv/bin/ruff check notagen tests
```

`transformers==4.40.0` is a **dev-only** dependency: the new code is plain PyTorch, and transformers is only needed to run the frozen legacy model in `tests/legacy_ref/` as a regression oracle.

## Layout

- `notagen/`: the single implementation everything uses.
  - `config.py`: `ModelConfig` dataclass + `PRESETS` (small/medium/large/x); `from_legacy_name()` parses released checkpoint filenames.
  - `tokenizer.py`: `Patchilizer`. The old per-stage copies differed in how they detect the tune body; that is now the `tunebody_detect` option (`startswith_v` = pretrain, `contains_v` = finetune/inference, `startswith_r` = RL input that already has `[r:]` tags).
  - `model.py`: plain-PyTorch GPT-2 patch/char decoders with **the legacy state-dict key names** (released `.pth` files load with `strict=True`), SDPA attention, and a static KV cache (`KVCache` + `CacheView` with per-row start positions).
  - `generation.py`: `Generator` (batched, continuous-batching port of `inference/inference.py`'s control flow) over a `Backend` protocol; `TorchBackend` lives here.
  - `mlx/`: native MLX backend (`NotaGenMLX`, `MLXBackend`, `convert_to_mlx`, `load_mlx_model`). Conv1D weights are transposed to `nn.Linear` so `mlx.nn.quantize` works.
  - `sampling.py`: `TorchSampler` (on device) and `LegacySampler` (the original NumPy chain, bit-compatible with seeded legacy runs).
  - `train.py`: LM pre-train/fine-tune and DPOP trainers. `checkpoint.py`: legacy `.pth` ↔ `safetensors` + `config.json`. `metrics.py`, `abc.py` (rest restoration), `infer.py`, `bench.py`, `cli.py`.
- Stage dirs (`pretrain/`, `finetune/`, `RL/`, `inference/`, `gradio/`) are now **thin wrappers** that read their `config.py` and call the package. They no longer contain model code. `RL/data.py`, `data/*`, and `clamp2/*` are unchanged; `clamp2/utils.py` is unrelated (CLaMP 2 / M3 models).
- `tests/legacy_ref/<stage>_utils.py`: frozen pre-refactor copies, used only as oracles. Don't edit them.

## Running things

CLI (anywhere): `notagen generate|convert|bench|train`. Stage scripts still work from their own directory and keep their `config.py` settings; new optional settings (`BACKEND`, `BATCH_SIZE`, `SAMPLER`, `SCHEDULE`, `PRECISION`, ...) are read with `getattr` defaults.

| Stage | Command | Notes |
|---|---|---|
| Data: MusicXML → ABC → interleaved/augmented | `python 1_batch_xml2abc.py`, `python 2_data_preprocess.py` (in `data/`) | unchanged |
| Pre-train | `notagen train pretrain --preset large --train-index ... ` or `python train-gen.py` (in `pretrain/`) | multi-GPU: `torchrun --nproc_per_node N ...` |
| Fine-tune | `notagen train finetune --init-from <weights> ...` or `python train-gen.py` (in `finetune/`) | |
| Batch inference | `notagen generate --weights <w> --num-samples 1000 --out-root output` or `python inference.py` (in `inference/`) | `--backend auto` = MLX on Apple Silicon, else torch |
| CLaMP 2 features / score | `python extract_clamp2.py`, `python statistics.py` (in `clamp2/`) | unchanged |
| DPO preference set | `python data.py` (in `RL/`) | unchanged |
| DPO training | `notagen train dpo --data-index ... --init-from ...` or `python train.py` (in `RL/`) | |
| Convert weights | `notagen convert <src> <dst> [--to mlx --quantize 8]` | |
| Throughput | `notagen bench --backend mlx --batch 1 8 32` | random weights |
| Demo | `python demo.py` (in `gradio/`) | weights path in `gradio/config.py` |

Output folder names are unchanged (`output/{original,interleaved}/<weights stem>_k_{K}_p_{P}_temp_{T}`); feature extraction must use `interleaved/`, and `RL/data.py` uses both.

## Architecture notes

- `Patchilizer` tokenizes ABC at character level (< 128) into 16-char **patches**: metadata line by line, tune body bar by bar. Ids: 0 pad, 1 BOS, 2 EOS.
- The patch-level GPT-2 autoregresses over patch embeddings; the char-level GPT-2 generates the 16 chars of each patch from the patch hidden state. With the KV cache, each new patch is one incremental patch step plus ≤16 char steps.
- **Stream mode**: tune-body lines carry `[r:i/remaining]`; when the context hits `patch_length`, generation re-seeds from metadata + the last half of the lines (`Generator._reseed`), and the patch cache is rebuilt for that row. A re-seed can leave a partial patch, which becomes the char decoder's prefix.
- The first tune-body patch is regenerated with a forced `[r:0/` prefix (legacy behaviour, kept).
- Training data omits whole-rest bars; `notagen.abc.rest_unreduce` restores them for CLaMP 2.
- CLaMP-DPO (`RL/data.py`): ranks generated pieces by CLaMP 2 cosine similarity to the mean ground-truth feature, takes the top/bottom `data_select_portion`, and drops chosen pieces with misaligned bars, ungrouped staves of one instrument, or CLaMP 2 similarity > 0.95 to any training piece. The DPOP loss is `-logsigmoid(beta * (logits - lambda * max(0, ref_pos - pol_pos)))`; reference log-probs are now computed once up front.

## Gotchas

- Parity tests compare against the legacy model in fp32 on CPU. **MLX GPU fp32 matmuls are ~TF32 precision** (~1e-3 relative on M-series), so strict MLX parity tests run on `mx.cpu`.
- The legacy LR schedules never applied (the scheduler was attached to an AdamW that was then replaced). The new trainer applies warmup; `schedule="constant"` reproduces the intended legacy behaviour.
- `torchrun` may hang on machines whose hostname resolves to an unreachable address; set `MASTER_ADDR=127.0.0.1` and launch ranks with `env://`, or use `--rdzv-endpoint 127.0.0.1:PORT` on a normal network.
- NotaGen-large (516M) needs ~24GB for training. Inference fits on a 16GB+ Mac with MLX (8-bit/4-bit quantization available).
