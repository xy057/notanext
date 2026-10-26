## Local Gradio Demo

1. Set up the environment (from the repository root):

  ```
  python -m venv .venv && source .venv/bin/activate
  pip install -e ".[demo,mlx]"        # mlx installs only on Apple Silicon
  ```

2. Download [NotaGen-X](https://huggingface.co/ElectricAlexis/NotaGen/blob/main/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth) and put it under ```weights/``` (or ```gradio/```, where the original instructions put it; ```run.py``` checks both).

3. Start the demo from the repository root:

  ```
  python run.py --open
  ```

  The original way still works: `cd gradio/ && python demo.py`, with the weights path set in `config.py`.

4. Then you can view the demo page at http://localhost:7861.

  <p align="center">
  <img src="illustration.png" alt="NotaGen Gradio Demo">
  </p>
  
  You can choose period, composer, and instrumentation as a prompt combination for NotaGen's conditional generation. After generation completes, you can save the ABC notation and MusicXML files locally.
  
  It is with some regret that the current combination of prompts is limited to 112, which is constrained by the number of pieces of music under each prompt in the fine-tuning dataset. We hope to expand the combinations and forms of prompts in the future.

### Options

| `run.py` flag | Environment variable (for `demo.py`) | Default |
|---|---|---|
| `--weights` | `NOTAGEN_WEIGHTS` | auto-detected (`weights/export/notagenx-mlx-bf16` on Apple Silicon, then `weights/export/notagenx`, then `weights/weights_notagenx*.pth`, then `gradio/` + `INFERENCE_WEIGHTS_PATH`) |
| `--backend auto\|mlx\|torch` | `NOTAGEN_BACKEND` | `auto`: MLX on Apple Silicon, else PyTorch (CUDA, MPS, or CPU) |
| `--dtype` | `NOTAGEN_DTYPE` | `auto`: bf16 (MLX / CUDA), fp16 (MPS), fp32 (CPU) |
| `--host` | `GRADIO_SERVER_NAME` | `127.0.0.1` for `run.py`. Plain `demo.py` keeps the original `0.0.0.0`, which is reachable from your network. |
| `--port` | `GRADIO_SERVER_PORT` | `7861` |

Memory: NotaGen-X needs about 8 GB on an NVIDIA GPU. On a Mac, the MLX exports from `notagen export` need about 1.0 GB (bf16), 0.55 GB (8-bit) or 0.3 GB (4-bit) for weights.
