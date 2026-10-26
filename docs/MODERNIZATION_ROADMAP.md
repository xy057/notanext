# NotaGen Modernization Roadmap

Written 2026-10-23 · Status updated 2026-10-26

## Progress (2026-10-26)

| Phase | State | What was checked |
|---|---|---|
| 0 Baseline & harness | **Done** (random-weight fixtures) | 46 `pytest` tests. The pre-refactor `utils.py` copies are frozen in `tests/legacy_ref/` as oracles. Tokenizer output matches every stage variant, including random stream cuts. Loss, gradients, RL log-probs and char probabilities match legacy in fp32. Generation is identical to the legacy loop under seeded legacy sampling, and a scripted model drives the `[r:0/` hack, hundreds of re-seeds and EOS/limit endings through both. Metrics: alignment, stave grouping, diversity, near-copy rate. |
| 1 One package | **Done** | `notagen/` + `pyproject.toml` + `notagen` CLI. Stage scripts are wrappers that keep their `config.py`. Legacy `.pth` load (`weights_only=True`) and `safetensors` round-trip. The released filenames all parse to the right presets. |
| 2 Inference performance | **Done** | KV cache, on-device sampling, continuous batching, bf16/fp16, SDPA. Numbers below. |
| 2b MLX backend | **Done** (inference). LoRA/DPO in MLX not started | Strict parity with PyTorch on MLX's CPU stream: greedy decoding exact, seeded legacy sampling exact. 8-bit conversion round-trips. |
| 3 Training infrastructure | **Done** | LR schedule fixed. torch.amp, clipping, fused AdamW, gradient checkpointing, exact resume, DataLoader workers, DPO with precomputed reference and batch > 1. Checked by tests and smoke runs: CPU, MPS bf16, 2-rank DDP (gloo). |
| 4–6 | Not started | |

**Verified on released NotaGen-X weights** (`weights_notagenx_…_h_size_1280.pth`; SHA-256 `3df0bc38…4cff` matches the Hugging Face ETag; 515.9M params; loads with `weights_only=True`). Run with `NOTAGEN_WEIGHTS=<path> pytest tests/test_real_weights.py`:
- forward loss and patch hidden states match the legacy model (fp32, CPU);
- generated text is **identical** to the legacy loop for 2 prompts (Romantic/Chopin/Keyboard and Classical/Mozart/Chamber), both greedy and seeded `top_k=9, top_p=0.9, T=1.2`;
- MLX (fp32, CPU stream) is identical to PyTorch on the same 4 cases;
- golden hashes are recorded in `tests/fixtures/golden_weights_notagenx_…json` for future regression runs.

**Output quality across backends**, NotaGen-X, 24 full pieces per config (3 prompts × 8: Chopin/Keyboard, Mozart/Chamber, Schubert/Art Song), `top_k=9, top_p=0.9, T=1.2`, batch 8. The fp32 row with the legacy sampler is numerically the legacy pipeline:

| config | ended (EOS) | rests restored | bars aligned | staves grouped | → MusicXML | mean chars | distinct bars | chars/s |
|---|---|---|---|---|---|---|---|---|
| PyTorch-MPS fp32, legacy sampler | 24/24 | 24/24 | 22/24 | 24/24 | 24/24 | 6514 | 0.72 | 868 |
| PyTorch-MPS fp16 | 24/24 | 24/24 | 20/24 | 24/24 | 24/24 | 6479 | 0.74 | 1425 |
| MLX bf16 | 24/24 | 24/24 | 22/24 | 24/24 | 23/24 | 7437 | 0.72 | 1256 |
| MLX 8-bit | 24/24 | 24/24 | 24/24 | 24/24 | 24/24 | 6768 | 0.73 | 1700 |
| MLX 4-bit | 24/24 | 24/24 | 21/24 | 24/24 | 24/24 | 6924 | 0.71 | 2139 |

With 24 samples, a difference of 2–3 pieces is within sampling noise, so these structural metrics don't separate the configs. "Distinct bars" is the share of unique bar strings, a rough proxy for variety. The table can't detect musical-quality differences: that needs CLaMP 2 scores or listening, and neither has been done. `chars/s` is end-to-end at batch 8, including re-seeds and continuous-batching refills.

**Not verified yet:**
- CLaMP 2 scores, which need CLaMP 2 weights and ground-truth features;
- the "re-fine-tune matches the original loss curve" exit check;
- CUDA runs. Everything here ran on an M5 Pro (64GB): CPU, MPS and MLX.

**Generation throughput**, NotaGen-large, random weights, patches/s (higher is better):

| | batch 1, short context | batch 1, ~720-patch context | batch 32 |
|---|---|---|---|
| legacy loop, PyTorch-MPS fp16 | 6.3 | 4.6 | (one piece at a time) |
| KV cache, PyTorch-MPS fp16 | 24.2 | 23.5 | 670 |
| MLX bf16 | 38.7 | 39.2 | 862 |
| MLX 8-bit | 53.1 | – | 665 |
| MLX 4-bit | 67.4 | – | 674 |

At batch 1 the PyTorch path is limited by kernel-launch overhead (about 116 small layer calls per patch). That is why MLX's lazy graphs win there. Quantization speeds up batch 1, which is memory-bound, but not batch 32, which is compute-bound.

**Behaviour changes (intended):**
- Warmup now applies. It never did in the legacy LR schedule.
- DPO reference log-probs are computed in eval mode. The legacy reference ran with dropout on.
- DPO encodes each file once, so any stream cut is fixed per file instead of re-drawn for each pair.
- The Gradio demo uses the batch re-seed. The old one raised `NameError` past 1024 patches.
- `TorchSampler` drops the ~1e-10 probability that the legacy sampler leaked to filtered tokens. `--sampler legacy` keeps the old behaviour.

---

This roadmap is for bringing NotaGen from its early-2025 research-code state up to current (late 2026) practice. It is ordered so that each phase can be checked before the next one starts, and so that the phases that need **no retraining** come first. Every phase has an exit criterion. A phase isn't done because the code merged; it's done when that check passes.

---

## 1. Where the code stands today

| Area | Current state | Why it matters |
|---|---|---|
| Generation | `NotaGenLMHeadModel.generate` re-encodes the **whole patch context** (up to 1024 patches × 20 layers) to produce each 16-char patch. `CharLevelDecoder.generate` also re-runs from scratch for every character. There's no KV cache at any level. | Most of the inference compute is redundant. Long pieces get slower with every patch. |
| Sampling | For every character: GPU→CPU copy, then NumPy `top_k → normalize → top_p → normalize → temperature`, then a copy back. | 16 host↔device syncs per patch, and the GPU sits idle while the CPU samples. |
| Batching | One piece at a time (`NUM_SAMPLES=1000` runs serially). | CLaMP-DPO data generation is the slowest step in the loop. |
| Devices | Hard-coded `cuda`, with a CPU fallback. | It can't use Apple Silicon (neither MPS nor native MLX) or other backends. |
| Precision | fp16 + `torch.cuda.amp.GradScaler` (deprecated API). | bf16 is the norm on current GPUs and doesn't need loss scaling. |
| Code layout | 5 drifted copies of `utils.py` / `config.py`. Scripts depend on `from config import *` and the cwd. Settings are edited in source files. | Fixes don't reach every copy, nothing can be reused as a library, and runs can't be reproduced. |
| Checkpoints | `.pth` pickles. The hyperparameters live **only in the filename**. | Pickle files are unsafe to load, and if a file is renamed, its config is lost. |
| Training | Constant LR after warmup. No grad clipping. The dataset re-patchilizes every sample on the fly. Multi-GPU uses DDP via `accelerate`. | It leaves stability and throughput on the table. |
| DPO (`RL/train.py`) | Batch size 1. A full reference-model copy sits in GPU memory. The reference log-probs are recomputed every step. | Memory doubles, and every step spends compute on reference passes that could be done once. |
| Tests / eval | There's no test suite. Evaluation is the CLaMP 2 score plus listening. | No change can be verified, including any change in this roadmap. |
| Dependencies | `transformers==4.40.0`, `torch==2.3.0`, `gradio==5.17.1`. | These are about 2.5 years old and some are hard to install on new Python and CUDA versions. |

---

## 2. Guiding principles

1. **Released weights stay loadable through Phase 4.** Every change before Phase 5 must load the existing HF checkpoints (NotaGen-small/medium/large, the fine-tuned and RL3 versions, NotaGen-X) bit-compatibly.
2. **Measure before optimizing.** Phase 0 builds the yardstick, and every later phase reports against it.
3. **Make it cheap before making it bigger.** Fast batched inference (Phase 2) is what makes online RL (Phase 4) affordable, and online RL is the cheapest real quality gain. A new architecture (Phase 5) is the most expensive gain, so it comes last.
4. **No silent behavior changes.** Any change that alters sampled output under a fixed seed needs an explicit sign-off in the regression harness.

---

## 3. Phase overview

```
Phase 0  Baseline + regression harness            ── no retraining
Phase 1  Consolidate into one package             ── no retraining
Phase 2  Inference performance                    ── no retraining
Phase 2b Native MLX backend (Apple Silicon)       ── no retraining (converts weights)
Phase 3  Training infrastructure                  ── no retraining (validated by a short re-fine-tune)
Phase 4  Post-training upgrades (RL)              ── fine-tune from released weights
Phase 5  Architecture v2                          ── full retraining
Phase 6  Release + eval reporting                 ── runs alongside every phase
```

Dependencies: 0 → 1 → {2, 2b, 3} → 4 → 5. Phases 2, 2b and 3 can run in parallel once Phase 1 has landed. 2b is easier after 2's cache design exists.

---

## Phase 0: Baseline and regression harness

**Goal:** a reproducible definition of "nothing broke" and "this got better".

Work items:
- **Golden-output tests.** For each released checkpoint, fix the prompts (period/composer/instrumentation and the unconditional case) and record:
  - the logits of a forward pass on a fixed ABC file (tolerance-checked);
  - greedy (`top_k=1`) generations for N patches, compared for exact match.
- **Metrics suite** (`eval/`), each reported for a fixed prompt set × K samples:
  - *Validity:* how many outputs parse with `abctoolkit` and convert to MusicXML through `abc2xml.py`.
  - *Structural correctness:* how often bar durations are misaligned and how often staves of the same instrument end up ungrouped. These checks reuse the logic from the `RL/data.py` filters.
  - *Fidelity:* CLaMP 2 similarity to the ground-truth mean (the existing `clamp2/statistics.py`).
  - *Diversity:* mean pairwise CLaMP 2 distance among the samples for one prompt, and near-copy rate against training data (Levenshtein > 0.95). This guards against reward hacking in Phase 4.
  - *Performance:* patches/s, chars/s, peak memory, and time to first patch on the reference hardware (one CUDA GPU and one Apple Silicon machine).
- Record a pinned environment lockfile that reproduces the paper-era numbers.

**Exit:** `pytest` reproduces the golden outputs on the current code, and a baseline metrics report is committed.

Effort: **S–M**

---

## Phase 1: Consolidate into one package

**Goal:** one source of truth for the tokenizer and model, and runs you can configure without editing source files.

Work items:
- Create a `notagen/` package:
  - `notagen/tokenizer.py`: one `Patchilizer`. The drift between the copies (for example, `'[V:' in line` vs `line.startswith('[V:')`) becomes an explicit, documented option. It is not silently unified, because the golden tests from Phase 0 decide which behavior each checkpoint needs.
  - `notagen/model.py`: `PatchLevelDecoder`, `CharLevelDecoder`, `NotaGenLMHeadModel`, keeping the **same state-dict keys**.
  - `notagen/sampling.py`, `notagen/abc/` (interleave, rest restore, xml2abc/abc2xml wrappers), and `notagen/config.py` with typed dataclasses.
- Turn the stage directories into thin entry points: `notagen train pretrain|finetune|dpo`, `notagen generate`, `notagen clamp2 extract|score`, `notagen data xml2abc|preprocess|abc2xml`. Config comes from YAML plus CLI overrides, so the cwd no longer matters.
- **Checkpoint format:** read the legacy `.pth`, write `safetensors` plus a `config.json` that holds the architecture hyperparameters. Load with `weights_only=True`. Keep the filename-derived naming only as an optional output label.
- Packaging: `pyproject.toml`, `uv` lockfile, `ruff`, `pytest` in CI (CPU-only golden tests on NotaGen-small).
- Move the dependencies up to current versions (torch 2.x latest, transformers latest, gradio latest). The golden tests are the gate for this.

**Exit:** all five stages run through the package, and the Phase 0 golden tests pass for every released checkpoint on the upgraded stack.

Effort: **M**

---

## Phase 2: Inference performance (no retraining)

**Goal:** generation that is an order of magnitude faster and batched, on CUDA, MPS and CPU.

Work items:
- **Patch-level KV cache.** Encode the prompt once, then run each new patch as a single-position incremental step. When stream mode slides the window and re-seeds from metadata + recent lines, recompute the cache (GPT-2's absolute positions mean the cache can't be shifted). That rebuild happens about once per window, not once per patch.
- **Char-level KV cache.** Each patch is ≤17 positions, so the main gain is removing per-step Python and embedding overhead.
- **Sampling on the GPU.** top-k, top-p and temperature as fused torch ops that keep the existing `safe_normalize_probs` semantics. Use a seeded `torch.Generator` so tests stay deterministic.
- **Batched generation.** B pieces at once, with a finished-mask for each sequence, padding-aware attention masks, and stream-mode window handling per sequence.
- **Kernels and precision:** bf16 (fp16 on MPS where bf16 is slow), PyTorch SDPA attention, optional `torch.compile` with static cache shapes.
- **Quantization:** replace the bitsandbytes path in `gradio/inference.py` (CUDA-only) with `torchao` int8 weight-only (int4 optional), which also runs on Mac.
- Port the Gradio demo onto the shared package. Optionally add streaming output that renders bars as they're generated.

**Exit:**
- Greedy outputs match Phase 0 exactly (fp32), and sampled-output metrics stay within noise (bf16).
- At least 10× patches/s on a long-piece benchmark vs baseline. That target is a goal to measure against, not a promise.
- NotaGen-X runs on Apple Silicon.

Effort: **M–L.** This is the single highest-leverage phase.

---

## Phase 2b: Native MLX backend (Apple Silicon)

**Goal:** a first-class Apple Silicon path that doesn't go through PyTorch-MPS: faster local generation, lower memory, and a NotaGen-X demo that runs well on an ordinary MacBook.

Why MLX as well as MPS: PyTorch-MPS is a compatibility layer. Ops fall back to the CPU, bf16 is uneven, and quantized kernels are thin. MLX is built for unified memory (weights aren't copied between host and device), ships fast 4/8-bit quantized matmuls, and compiles graphs lazily. This model is small, autoregressive and memory-bound, which is the workload where MLX gains the most.

Work items:
- **`notagen/mlx/` model port** in `mlx.nn`: `PatchLevelDecoder`, `CharLevelDecoder` and the GPT-2 blocks. The model is about 400 lines and uses only standard blocks (embedding, LayerNorm, attention, MLP), so a hand port is small. Use `mx.fast.scaled_dot_product_attention` and `mx.fast.layer_norm`.
- **KV cache and sampling in MLX**, matching the Phase 2 design: patch-level incremental cache, rebuilt when the stream-mode window slides; char-level cache; top-k/top-p/temperature with `mx.random.categorical` and a fixed key so outputs are reproducible.
- **Weight converter** `notagen convert --to mlx`: legacy `.pth` or Phase 1 `safetensors` → MLX `safetensors` + `config.json`. Handle GPT-2's `Conv1D` transposed weight layout explicitly. Optional `--quantize 4|8` through `mlx.nn.quantize`. Prebuilt converted weights can be published to the Hub (Phase 6).
- **Batched generation** with per-sequence finished masks, so local CLaMP-DPO sample generation becomes feasible on a Mac.
- **Backend selection:** `notagen generate --backend {torch,mlx}`, defaulting to MLX on Apple Silicon when it's installed. The tokenizer, ABC post-processing (rest restore, interleaving) and abc2xml stay shared because they're pure Python.
- **Gradio demo** uses the MLX backend on Macs.
- *Stretch:* MLX training for **LoRA fine-tuning and DPO** on small/medium models, so a Mac user can fine-tune on their own corpus. Full pretraining stays on CUDA.

**Exit:**
- **Parity:** fp32 MLX logits match PyTorch fp32 on the Phase 0 fixtures within tolerance, and greedy generations match exactly for N patches, including across a stream-window boundary.
- **Quality:** bf16 and 8-bit sampled-output metrics stay within noise of PyTorch. 4-bit is reported separately and labeled as lossy.
- **Speed:** patches/s and peak memory are reported against PyTorch-MPS on the same Mac. MLX is the default only if it's faster.
- CI runs the parity tests on a macOS (Apple Silicon) runner.

Depends on: Phase 1 (safetensors + `config.json`, shared tokenizer) and the Phase 0 fixtures. It can run in parallel with Phases 2 and 3, though porting the cache after Phase 2's torch version exists is easier because the design is already settled.

Effort: **M** for inference, **+M** for the LoRA/DPO stretch.

---

## Phase 3: Training infrastructure (same architecture)

**Goal:** modern, stable, efficient training loops for the existing model, ready for Phases 4 and 5.

Work items:
- bf16 autocast via the `torch.amp` API. Drop `GradScaler` unless training in fp16.
- Cosine (or WSD) LR schedule with warmup, gradient clipping, fused AdamW, and an optional gradient-checkpointing flag.
- Distributed: FSDP2 or DDP through `accelerate`, with checkpoints that resume exactly (model, optimizer, scheduler, RNG, data position).
- **Pre-tokenized sharded dataset.** Patchilize once into memory-mapped shards so the training loop no longer re-parses ABC. Use dataloader workers and length-bucketing.
- **DPO trainer rewrite:**
  - batch > 1 with proper padding;
  - **precompute reference log-probs offline**, so the reference model doesn't have to stay in GPU memory;
  - log reward margins, chosen/rejected log-probs and the DPOP penalty term.
- Logging: make wandb optional. Add a local fallback (TensorBoard/CSV).

**Exit:** a short fine-tune from the released NotaGen-large pretrained weights matches the original fine-tune loss curve within tolerance, and is faster per step with lower memory. One DPO iteration reproduces the RL1 metric direction.

Effort: **M**

---

## Phase 4: Post-training upgrades (fine-tune from released weights)

**Goal:** the cheapest real quality gains. These need the fast batched inference from Phase 2.

### 4a. A reward that is more than CLaMP 2 similarity
CLaMP 2 similarity to the **mean** ground-truth feature rewards "averageness". Taken alone, it pushes the model toward bland, mode-collapsed pieces. Build a composite score:
- CLaMP 2 similarity, as today;
- hard rule checks from `RL/data.py` turned into penalties: validity, bar-duration alignment, staff grouping, near-copy of training data;
- optional per-prompt conditioning fidelity, meaning how close the piece lands to *that composer's/period's* feature centroid rather than the global one;
- a diversity regularizer measured within the sample group.

### 4b. Online RL
Replace the offline "generate 1000 → pick top/bottom portion → DPOP" loop with an online, group-based policy-gradient method (GRPO-style):
- for each prompt, sample a group of G pieces in one batch (this needs Phase 2);
- use group-normalized advantages from the composite reward, with a KL penalty to the fine-tuned reference;
- the rule-based rewards are cheap, CLaMP 2 scoring runs batched on the same GPU, and no learned reward model is needed.

Keep the improved offline DPO from Phase 3 as the fallback/baseline arm. Compare the two at equal generation budget.

### 4c. Guardrails
- Track the diversity and near-copy metrics every N steps, and stop early if diversity collapses.
- Run a blind listening test (A/B against NotaGen-X) before claiming a quality gain.

**Exit:** better validity, structure and CLaMP 2 scores than NotaGen-X, with diversity not worse, and a listening-test preference result reported either way.

Effort: **L.** It is mostly compute for sampling.

---

## Phase 5: Architecture v2 (full retraining)

**Goal:** a NotaGen-v2 model family. It is **not** weight-compatible with v1.

### 5a. Run ablations first
Before committing to a full pretrain, compare candidates at **small scale** (about 110M, the NotaGen-small size) on the same data and token budget, with the Phase 0 metrics:

| Candidate | Change | Hypothesis |
|---|---|---|
| Llama-style blocks | RoPE, RMSNorm, SwiGLU, GQA in place of GPT-2 blocks (the code already imports `LlamaModel`) | Better loss per FLOP. RoPE extends to longer context more gracefully. |
| Longer patch context | 2048–8192 patches, using RoPE + SDPA/FlashAttention | Whole movements fit in context, which reduces how often stream-mode re-seeding is needed or removes it. |
| Char decoder | Keep the small autoregressive decoder vs a lighter decoder / multi-token head | Most inference time after Phase 2 may be in the char decoder. |
| Patch embedding | One-hot × Linear(16·128) vs summed per-position char embeddings | Same expressiveness and fewer FLOPs. Probably neutral on quality. |
| Tokenization | Character patches (current) vs bar-aware or BPE patching | Char-level is robust for ABC syntax. BPE shortens sequences. This is a real trade-off and needs measuring. |

Pick the winner by the eval suite, not by training loss alone.

### 5b. Conditioning
Today the conditioning is three comment lines (`%period`, `%composer`, `%instrumentation`) with a closed vocabulary. Options:
- keep the comment prompts for compatibility;
- add **text conditioning** through the CLaMP 2 text encoder (CLaMP 2 is a multilingual text–music model already in this repo). That would allow free-text prompts such as "a melancholic nocturne for cello and piano" through cross-attention or a prefix.

### 5c. Data
- The original 1.6M-piece pre-training corpus is not fully public. Rebuild from public-domain and openly licensed sources (for example PDMX and other public-domain score collections) and record the provenance and license of every source.
- Deduplicate, then reuse the existing interleave + 15-key transposition augmentation.
- Hold out the fine-tuning composers' works so that "near-copy" stays measurable.

### 5d. Then the full pipeline
Pretrain → fine-tune → Phase 4 RL on the chosen architecture, all on the Phase 3 infrastructure.

**Exit:** NotaGen-v2 beats the Phase 4 v1 model on the eval suite and the listening test at comparable parameter count, or the ablation report shows it isn't worth it. Either result is valid.

Effort: **XL.** It is gated by data licensing and GPU budget.

---

## Phase 6: Release and reporting (runs alongside every phase)

- Publish to the HF Hub in the `safetensors` + `config.json` format, with `from_pretrained`-style loading. Publish MLX-converted weights too (bf16 plus 8-bit/4-bit quantized). Keep the legacy `.pth` links in the README.
- Write a model card for each release: training data provenance, eval-suite numbers, known failure modes, and the license (the repo is MIT; state the data licenses separately).
- Update `README.md` and `CLAUDE.md` to the new commands at the end of each phase.

---

## 4. Alternatives considered (and why they aren't the main track)

| Alternative | Why not primary |
|---|---|
| Fine-tune a general open LLM on ABC text | Text conditioning comes for free, but at 10–100× the parameters and inference cost. The BPE tokenizer splits ABC tokens unevenly, and the specialized hierarchical model is already strong per parameter. It's worth one Phase 5a-style ablation if compute allows. |
| Use an off-the-shelf DPO/RL library (e.g. TRL) as-is | The two-level patch/char model doesn't fit the standard causal-LM interfaces. Borrow the algorithms and keep a small custom trainer. |
| Switch to audio or MIDI generation | That is a different product. Symbolic ABC output is NotaGen's core value (editable, printable scores). |

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Consolidation breaks checkpoint behavior (the drifted `Patchilizer` copies) | Phase 0 golden tests for each checkpoint, plus explicit tokenizer options. |
| The KV cache adds subtle mask or position bugs in stream mode | Exact greedy-match tests across window boundaries. |
| The MLX and PyTorch model code drift apart (the same problem as today's five `utils.py` copies) | Cross-framework parity tests in CI on every change to either model. Phase 5 architectures must land in both or be explicitly marked torch-only. |
| RL reward hacking / mode collapse | Composite reward, diversity metric with early stopping, listening tests. |
| Architecture v2 doesn't beat v1 | Ablate at 110M first. Phase 5 is optional; Phases 0–4 deliver value regardless. |
| Training data licensing | Track provenance per source, and use public-domain/openly licensed corpora only. |
| GPU budget | Phases 0–3 need no large training runs. Phase 4 runs on 1–2 GPUs. Commit to Phase 5 only after the ablations. |

---

## 6. Suggested first milestone

**Phases 0 + 1 + the KV-cache and GPU-sampling items from Phase 2.** This milestone delivers the visible win (much faster generation, runs on Mac) on the existing weights, plus the test harness that every later phase depends on.
