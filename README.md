# 🎵 NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms

<p align="center">
  <!-- ArXiv -->
  <a href="https://arxiv.org/abs/2502.18008">
    <img src="https://img.shields.io/badge/NotaGen_Paper-ArXiv-%23B31B1B?logo=arxiv&logoColor=white" alt="Paper">
  </a>
  &nbsp;&nbsp;
  <!-- HuggingFace -->
  <a href="https://huggingface.co/ElectricAlexis/NotaGen">
    <img src="https://img.shields.io/badge/NotaGen_Weights-HuggingFace-%23FFD21F?logo=huggingface&logoColor=white" alt="Weights">
  </a>
  &nbsp;&nbsp;
  <!-- HuggingFace Space -->
  <a href="https://huggingface.co/spaces/ElectricAlexis/NotaGen">
    <img src="https://img.shields.io/badge/NotaGen_Space-Huggingface-✨️?logo=huggingface&logoColor=white" alt="Space">
  </a>
  &nbsp;&nbsp;
  <!-- Web Demo -->
  <a href="https://electricalexis.github.io/notagen-demo/">
    <img src="https://img.shields.io/badge/NotaGen_Demo-Web-%23007ACC?logo=google-chrome&logoColor=white" alt="Demo">
  </a>
</p>

<p align="center">
  <img src="notagen.png" alt="NotaGen" width="50%">
</p>


## 📖 Overview
**NotaGen** is a symbolic music generation model that explores the potential of producing **high-quality classical sheet music**. Inspired by the success of Large Language Models (LLMs), NotaGen adopts a three-stage training paradigm:
- 🧠 **Pre-training** on 1.6M musical pieces
- 🎯 **Fine-tuning** on ~9K classical compositions with `period-composer-instrumentation` prompts
- 🚀 **Reinforcement Learning** using our novel **CLaMP-DPO** method (no human annotations or pre-defined rewards required.)

Check our [demo page](https://electricalexis.github.io/notagen-demo/) and enjoy music composed by NotaGen!

## ⚙️ Environment Setup

Python 3.10+ with PyTorch 2.4+. It runs on NVIDIA GPUs (CUDA), on Apple Silicon (MLX, or PyTorch on MPS), and on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,demo,train,clamp]"
```

The extras are optional. `mlx` installs only on Apple Silicon. `demo` is Gradio 5. `train` adds Weights & Biases logging. `clamp` adds CLaMP 2 for scoring and RL. A plain `pip install -r requirements.txt` also works.

### 🚀 Quick start

1. Download [NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth) (6.2 GB) into `weights/`.
2. Start the demo:

   ```bash
   python run.py --open
   ```

   `run.py` finds the weights, uses MLX on Apple Silicon and PyTorch elsewhere, and serves on `127.0.0.1:7861`. Options:
   - `--weights` picks a different weights file or folder;
   - `--backend mlx|torch` overrides the backend choice;
   - `--host 0.0.0.0` serves on your network instead of only this machine;
   - `--port` changes the port.

3. Or generate from the command line:

   ```bash
   notagen generate --weights weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth \
     --prompt "Romantic|Chopin, Frederic|Keyboard" --num-samples 8 --batch-size 8 --out-root output
   ```

Valid `period|composer|instrumentation` combinations are listed in [gradio/prompts.txt](gradio/prompts.txt).

### What's new in this version

All stages now share one package (`notagen/`). The per-stage scripts below still work as before: edit the `config.py` in that folder and run the script from inside it. The `notagen` command offers the same features without editing files (`notagen --help`):

| Command | What it does |
|---|---|
| `notagen generate` | Batch inference (= `inference/inference.py`) |
| `notagen train pretrain\|finetune\|dpo` | Training (= `pretrain/`, `finetune/`, `RL/train.py`) |
| `notagen clamp2 extract\|score` | CLaMP 2 features and score (= `clamp2/extract_clamp2.py`, `clamp2/statistics.py`) |
| `notagen prefs` | CLaMP-DPO preference set (= `RL/data.py`) |
| `notagen eval` | Generate for a prompt set; report structure checks and CLaMP 2 metrics |
| `notagen rl` | Online RL from released weights, with a CLaMP 2 reward (no training data needed) |
| `notagen mlx-train finetune\|dpo` | LoRA / QLoRA fine-tuning and DPO on Apple Silicon |
| `notagen convert` / `notagen export` | Convert weights to `safetensors` or MLX (bf16, 8-bit, 4-bit), with model cards |
| `notagen bench` | Generation throughput |

Main changes from the original scripts:
- **Faster generation.** A KV cache plus batched generation make it about 4–10× faster for a single piece and over 100× faster in throughput.
- **Same results.** With the same seed, the output is identical to the original code on the released NotaGen-X weights. The test suite in `tests/` checks this.
- **Native Apple Silicon.** MLX runs inference, including 4/8-bit quantization (NotaGen-X is 0.3 GB at 4-bit), and LoRA training.
- **Modern training.** bf16 runs on any GPU, gradient checkpointing and exact resume are available, and the learning-rate warmup is now actually applied (it never was in the original scripts).
- **Safer, smaller checkpoints.** `safetensors` checkpoints include their own `config.json`, and legacy `.pth` files are loaded without unpickling arbitrary objects.
- **CLaMP 2 built in.** Features, scores, preference sets, evaluation and an RL reward are computed in-process.

Benchmarks, verification results, and every intended behaviour change are in [docs/MODERNIZATION_ROADMAP.md](docs/MODERNIZATION_ROADMAP.md). To run the tests:

```bash
pip install -e ".[dev]" && pytest   # about 1 minute; tests that need real weights are skipped unless they're present
```

## 🏋️ NotaGen Model Weights

### Pre-training
We provide pre-trained weights of different scales:
|  Models         |  Parameters  |  Patch-level Decoder Layers  |  Character-level Decoder Layers  |  Hidden Size  |  Patch Length (Context Length)  |
|  ----           |  ----  |  ---- |  ----  |  ----  |  ----  |
|  [NotaGen-small](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth)  | 110M   |  12   |  3     |  768   |  2048  |
|  [NotaGen-medium](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_16_c_layers_3_h_size_1024_lr_0.0001_batch_4.pth) | 244M   |  16   |  3     |  1024  |  2048  |
|  [NotaGen-large](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth)  | 516M   |  20   |  6     |  1280  |  1024  |

**Notice**: The pre-trained weights cannot be used for conditional generation based on 'period-composer-instrumentation'.

### Fine-tuning

We fine-tuned NotaGen-large on a corpus of approximately 9k classical pieces. You can download the weights [here](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain-finetune_p_size_16_p_length_1024_p_layers_c_layers_6_20_h_size_1280_lr_1e-05_batch_1.pth).

### Reinforcement-Learning

After pre-training and fine-tuning, we optimized NotaGen-large with 3 iterations of CLaMP-DPO. You can download the weights [here](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagen_pretrain-finetune-RL3_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-06_batch_1.pth).

### 🌟 NotaGen-X

Inspired by Deepseek-R1, we further optimized the training procedures of NotaGen and released a better version --- [NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth). Compared to the version in the paper, NotaGen-X incorporates the following improvements:

- We introduced a post-training stage between pre-training and fine-tuning, refining the model with a classical-style subset of the pre-training dataset.
- We removed the key augmentation in the Fine-tune stage, making the instrument range of the generated compositions more reasonable.
- After RL, we utilized the resulting checkpoint to gather a new set of post-training data. Starting from the pre-trained checkpoint, we conducted another round of post-training, fine-tuning, and reinforcement learning.

If you want to add a new composer style to NotaGen-X, please refer to issue [#18](https://github.com/ElectricAlexis/NotaGen/issues/18) for more instructions :D

### Using the weights

All released `.pth` files load directly, and the model size is read from the filename. To get smaller, pickle-free files:

```bash
notagen export weights/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth --name notagenx
# -> weights/export/notagenx            safetensors, fp32, 2.1 GB (bit-identical tensors, no optimizer state)
#    weights/export/notagenx-mlx-bf16   1.0 GB
#    weights/export/notagenx-mlx-8bit   0.55 GB
#    weights/export/notagenx-mlx-4bit   0.30 GB
```

Any of these folders can be passed wherever weights are expected (`--weights`, `INFERENCE_WEIGHTS_PATH`, `run.py`).

## 🎹 Demo

### Online Gradio Demo

We developed an [online gradio demo](https://huggingface.co/spaces/ElectricAlexis/NotaGen) on Huggingface Space for NotaGen-X. You can input **"Period-Composer-Instrumentation"** as the prompt to have NotaGen generate music, preview the audio / pdf scores, and download them :D

<p align="center">
  <img src="gradio/illustration_online.png" alt="NotaGen Gradio Demo">
</p>

### Local Gradio Demo

We developed a local Gradio demo for NotaGen-X. You can input **"Period-Composer-Instrumentation"** as the prompt to have NotaGen generate music！

<p align="center">
  <img src="gradio/illustration.png" alt="NotaGen Gradio Demo">
</p>

Start it with `python run.py --open` (see [Quick start](#-quick-start)). It runs on an NVIDIA GPU with about 8 GB of memory, or on any Apple Silicon Mac through MLX: the 4-bit export needs about 0.3 GB for weights. After generation you can save the ABC notation and MusicXML files. For details, see [gradio/README.md](gradio/README.md).

### Online Colab Notebook

Thanks for [@deeplearn-art](https://github.com/deeplearn-art/NotaGen)'s contribution of a [Google Colab notebook for NotaGen](https://colab.research.google.com/drive/1yJA1wG0fiwNeehdQxAUw56i4bTXzoVVv?usp=sharing)! You can run it and access to a Gradio public link to play with this demo. 🤩

### ComfyUI

Thanks for [@billwuhao](https://github.com/billwuhao/ComfyUI_NotaGen)'s contribution of [a ComfyUI node for NotaGen](https://github.com/billwuhao/ComfyUI_NotaGen)! It can automatically convert generated .abc to .xml, .mp3, and .png formats. You can listen to the generated music and see the sheet music too! Please visit the [repository page](https://github.com/billwuhao/ComfyUI_NotaGen) for more information. 🤩 

<p align="center">
  <img src="https://github.com/billwuhao/ComfyUI_NotaGen/blob/master/images/2025-03-10_06-24-03.png" alt="NotaGen ComfyUI">
</p>


## 🛠️ Data Pre-processing & Post-processing

For converting **ABC notation** files from / to **MusicXML** files, please view [data/README.md](https://github.com/ElectricAlexis/NotaGen/blob/main/data/README.md) for instructions.

To illustrate the specific data format, we provide a small dataset of **Schubert's lieder** compositions from the [OpenScore Lieder](https://github.com/OpenScore/Lieder), which includes:
- 🗂️ Interleaved ABC folders
- 🗂️ Augmented ABC folders
- 📄 Data index files for training and evaluation

You can download it [here](https://drive.google.com/drive/folders/1iVLkcywzXGcHFodce9nDQyEmK4UDmBtY?usp=sharing) and put it under ```data/```.

In the instructions of **Fine-tuning** and **Reinforcement Learning** below, we will use this dataset as an example of our implementation. **It won't include the "period-composer-instrumentation" conditioning**, just for showing how to adapt the pretrained NotaGen to a specific music style.


## 🧠 Pre-train
If you want to use your own data to pre-train a blank **NotaGen** model, please:
1. Preprocess the data and generate the data index files following the instructions in [data/README.md](https://github.com/ElectricAlexis/NotaGen/blob/main/data/README.md)
2. Modify the parameters in ```pretrain/config.py```

Use this command for pre-training:
```bash
cd pretrain/
torchrun --nproc_per_node 8 train-gen.py          # or: accelerate launch --multi_gpu train-gen.py
```

Single-device training is `python train-gen.py`. The equivalent CLI is `notagen train pretrain --preset small --train-index ... --eval-index ...` (see `--help`).

Optional settings you can add to `config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `PRECISION` | `"auto"` | `bf16` on GPUs that support it, else `fp16` (with loss scaling) or `fp32` |
| `SCHEDULE` | `"constant"` | `constant` (warmup then flat, the original intent) or `cosine` |
| `WARMUP_STEPS` | `1000` | Learning-rate warmup steps |
| `GRAD_CLIP` | `1.0` | Gradient-norm clipping; `0` turns it off, as in the original scripts |
| `GRAD_CHECKPOINTING` | `False` | Trade compute for memory |
| `NUM_WORKERS` | `4` | Data-loading worker processes |
| `OUT_DIR` | `"."` | Where checkpoints and logs go |

Outputs, written to `OUT_DIR`:
- `WEIGHTS_PATH`: the best checkpoint, in the original `.pth` layout;
- `best/`: the same weights as `safetensors` + `config.json`;
- `last.pth`: used to resume;
- `log.jsonl`: training logs.

With `LOAD_FROM_CHECKPOINT = True`, training resumes exactly from `last.pth`: model, optimizer, schedule and random state.

## 🎯 Fine-tune

Here we give an example on fine-tuning **NotaGen-large** with the **Schubert's lieder** data mentioned above.

**Notice:** Fine-tuning **NotaGen-large** needs about **24GB of GPU memory**. You can lower that with `GRAD_CHECKPOINTING = True` in `finetune/config.py`, use **NotaGen-small** or **NotaGen-medium** (change the model configuration in `finetune/config.py`), or, on a Mac, fine-tune LoRA adapters with MLX:

```bash
cd finetune/   # the index files' paths are relative to this folder
notagen mlx-train finetune --init-from <weights> --train-index ../data/schubert_augmented_train.jsonl \
  --eval-index ../data/schubert_augmented_eval.jsonl --out-dir runs/schubert-lora --quantize 4
```

`--quantize 4` trains adapters on a 4-bit base model (QLoRA). To also write a full merged model, train without `--quantize` and add `--fuse`.

### Configuration
- In ```finetune/config.py```:
  - Modify the ```DATA_TRAIN_INDEX_PATH``` and ```DATA_EVAL_INDEX_PATH```:
    ```python
    # Configuration for the data
    DATA_TRAIN_INDEX_PATH = "../data/schubert_augmented_train.jsonl" 
    DATA_EVAL_INDEX_PATH  = "../data/schubert_augmented_eval.jsonl"
    ```
  - Download pre-trained NotaGen weights, and modify the ```PRETRAINED_PATH```:
    ```python
    PRETRAINED_PATH = "../pretrain/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth"  # Use NotaGen-large
    ```
  - ```EXP_TAG``` is for differentiating the models. It will be integrated into the ckpt's name. Here we set it to ```schubert```.
  - You can also modify other parameters like the learning rate.

### Execution
Use this command for fine-tuning:
```bash
cd finetune/
CUDA_VISIBLE_DEVICES=0 python train-gen.py
```

The optional settings and outputs are the same as for pre-training. The fine-tuned weights are saved as `finetune/<WEIGHTS_PATH>`, with a `safetensors` copy in `finetune/best/`.

## 🚀 Reinforcement Learning (CLaMP-DPO)

Here we give an example on how to use **CLaMP-DPO** to enhance the model fine-tuned with **Schubert's lieder** data.

### ⚙️ [CLaMP 2](https://github.com/sanderwood/clamp2) Setup

Download model weights and put them under the ```clamp2/```folder:
- [CLaMP 2 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth)
- [M3 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

**For the `notagen` commands** (the "CLI" lines in each step below), you need less. The M3 weights and the 1.1 GB XLM-RoBERTa weights aren't needed, because the CLaMP 2 checkpoint overwrites both. Set up this layout:

```
weights/clamp2/weights_clamp2_h_size_768_..._m3_True.pth
weights/clamp2/xlm-roberta-base/{config.json, tokenizer.json, tokenizer_config.json, sentencepiece.bpe.model}
```

The four small files (about 14 MB) come from [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base). The CLI computes the same features as `extract_clamp2.py`, batched and in memory, on CUDA, MPS or CPU. Each step below shows the original script and the CLI equivalent.

### 🔍 Extract Ground Truth Features
Modify ```input_dir``` and ```output_dir``` in ```clamp2/extract_clamp2.py```:
```python
input_dir = '../data/schubert_interleaved'  # interleaved abc folder
output_dir = 'feature/schubert_interleaved'  # feature folder
```
Extract the features:
```
cd clamp2/
python extract_clamp2.py
```
CLI: `notagen clamp2 extract data/schubert_interleaved clamp2/feature/schubert_interleaved`

### 🔄 CLaMP-DPO

Here we give an example of an iteration of **CLaMP-DPO** from the initial model fine-tuned on **Schubert's lieder** data.

#### 1. Inference
- Modify the ```INFERENCE_WEIGHTS_PATH``` to path of the fine-tuned weights and ```NUM_SAMPLES``` to generate in ```inference/config.py```:
  ```python
    INFERENCE_WEIGHTS_PATH = '../finetune/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth'              
    NUM_SAMPLES = 1000                                               
  ```
- Inference:
  ```
  cd inference/
  python inference.py
  ```
  This will generate an ```output/```folder with two subfolders: ```original``` and ```interleaved```. The ```original/``` subdirectory stores the raw inference outputs from the model, while the ```interleaved/``` subdirectory contains data post-processed with rest measure completion, compatible with CLaMP 2. Each of these subdirectories will contain a model-specific folder, named as a combination of the model's name and its sampling parameters.

  Optional settings in `inference/config.py`:
  - `BATCH_SIZE` (default 8): how many pieces generate in parallel;
  - `BACKEND` (`auto`, `mlx` or `torch`);
  - `SAMPLER` (`fast`, or `legacy` for NumPy sampling identical to the original code);
  - `PROMPT_LINES`, for example `['%Romantic\n', '%Chopin, Frederic\n', '%Keyboard\n']`, for prompted models.

  CLI: `notagen generate --weights <weights> --num-samples 1000 --out-root output` (same folder names).

#### 2. Extract Generated Data Features

Modify ```input_dir``` and ```output_dir``` in ```clamp2/extract_clamp2.py```:
```python
input_dir = '../output/interleaved/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'  # interleaved abc folder
output_dir = 'feature/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'  # feature folder
```
Extract the features:
```
cd clamp2/
python extract_clamp2.py
```
CLI: `notagen clamp2 extract output/interleaved/<run> clamp2/feature/<run>`

#### 3. Statistics on Average CLaMP 2 Score (Optional)
If you're interested in the **Average CLaMP 2 Score** of the current model, modify the parameters in ```clamp2/statistics.py```:
```python
gt_feature_folder = 'feature/schubert_interleaved'
output_feature_folder = 'feature/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'
```
Then run this script:
```
cd clamp2/
python statistics.py
```
CLI: `notagen clamp2 score clamp2/feature/schubert_interleaved clamp2/feature/<run>` (also reports diversity and the near-copy rate).

#### 4. Construct Preference Data
Modify the parameters in ```RL/data.py```:
```python
gt_feature_folder = '../clamp2/feature/schubert_interleaved'
output_feature_folder = '../clamp2/feature/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'
output_original_abc_folder = '../output/original/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'
output_interleaved_abc_folder = '../output/interleaved/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1_k_9_p_0.9_temp_1.2'
data_index_path = 'schubert_RL1.json'  # Data for the first iteration of RL
data_select_portion = 0.1              
```
In this script, the **CLaMP 2 Score** of each generated piece will be calculated and sorted. The portion of data in the chosen and rejected sets is determined by ```data_select_portion```. Additionally, there are also three rules to exclude problematic sheets from the chosen set: 
- Sheets with duration alignment problems are excluded; 
- Sheets that may plagiarize from ground truth data (CLaMP 2 cosine similarity > 0.95 to any ground-truth piece) are excluded; 
- Sheets where staves for the same instrument are not grouped together are excluded.

The preference data file will be named as ```data_index_path```, which records the file paths in chosen and rejected sets.

Run this script:
```
cd RL/
python data.py
```
CLI, which computes the features itself. The JSON stores paths as you give them, so run it from `RL/` like `data.py`:
```bash
cd RL/
notagen prefs --interleaved ../output/interleaved/<run> --original ../output/original/<run> \
  --gt-features ../clamp2/feature/schubert_interleaved --portion 0.1 --out schubert_RL1.json \
  --weights ../weights/clamp2
```
Without ground truth, use `--prompt "period|composer|instrumentation"` instead of `--gt-features`. This ranks pieces by CLaMP 2 similarity to a text description of the prompt.

#### 5. DPO Training

Modify the parameters in ```RL/config.py```:
```python
DATA_INDEX_PATH = 'schubert_RL1.json'  # Preference data path
PRETRAINED_PATH = '../finetune/weights_notagen_schubert_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth'  # The model to go through DPO optimization
EXP_TAG = 'schubert-RL1'              # Model tag for differentiation
```
You can also modify other parameters like ```OPTIMIZATION_STEPS``` and DPO hyper-parameters. New optional settings:
- `BATCH_SIZE`: preference pairs per step (default 1);
- `PRECISION`;
- `GRAD_CHECKPOINTING`.

The reference model's log-probabilities are computed once before training, so only one model is held in GPU memory.

Run this script:
```
cd RL/
CUDA_VISIBLE_DEVICES=0 python train.py
```
CLI, run from `RL/`: `notagen train dpo --data-index schubert_RL1.json --init-from <fine-tuned weights> --out-dir .`. On a Mac, `notagen mlx-train dpo` takes the same arguments and trains LoRA adapters.

After training, a model named ```weights_notagen_schubert-RL1_beta_0.1_lambda_10_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-06.pth``` will be saved under ```RL/```, with a `safetensors` copy in `RL/final/`. For the second round of CLaMP-DPO, please go back to the first inference stage, and let the new model to generate pieces.

For this small experiment on **Schubert's lieder** data, we post our **Average CLaMP 2 Score** here for the fine-tuned model and models after each iteration of CLaMP-DPO, as a reference:

|  CLaMP-DPO Iteration (K) |  Average CLaMP 2 Score  | 
|  ----           |  ----  | 
|  0 (fine-tuned) | 0.324  |  
|  1              | 0.579  |
|  2              | 0.778  |

If you are interested in this method, have a try on your own style-specific dataset :D

## 📈 Evaluation and Online RL with CLaMP 2 (new)

These commands need only model weights and the CLaMP 2 files above (run them from the repo root):

```bash
# Generate 8 pieces for each of 8 keyboard composers; report structure checks + CLaMP 2 metrics
notagen eval --weights <weights> --n 8 --out runs/eval
# add --reference clamp2/feature/<ground truth> to also get the paper's CLaMP 2 score

# Online RL from released weights, rewarded by CLaMP 2 + structure checks
notagen rl --init-from <weights> --steps 120 --group-size 8 --lr 3e-6 --beta 0.04 --out-dir runs/rl
```

`notagen eval` reports:
- **Completion and structure:** finished, rests restored, bars aligned, staves grouped.
- **Text similarity:** CLaMP 2 similarity between each piece and a text description of its prompt.
- **Retrieval accuracy:** whether CLaMP 2 picks a piece's own prompt out of the whole set.
- **Diversity** within each prompt.

`notagen rl` samples a group of pieces per prompt and normalizes rewards within the group (GRPO-style). It uses a KL penalty to the starting model.

In a 120-step run on NotaGen-X, CLaMP 2 retrieval accuracy over 10 keyboard composers went from 0.39 to 0.49, and text similarity from 0.117 to 0.135, with structure unchanged. Limits of that result:
- At that sample size the gain isn't statistically significant.
- The two held-out composers didn't improve.
- CLaMP 2 was both the reward and the judge, so this doesn't show the music got better.

See [docs/MODERNIZATION_ROADMAP.md](docs/MODERNIZATION_ROADMAP.md) for the full numbers.

## 📚 Citation

If you find **NotaGen** or **CLaMP-DPO** useful in your work, please cite our paper.

```bibtex
@misc{wang2025notagenadvancingmusicalitysymbolic,
      title={NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms}, 
      author={Yashan Wang and Shangda Wu and Jianhuai Hu and Xingjian Du and Yueqi Peng and Yongxin Huang and Shuai Fan and Xiaobing Li and Feng Yu and Maosong Sun},
      year={2025},
      eprint={2502.18008},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.18008}, 
}
```
