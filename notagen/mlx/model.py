"""NotaGen in MLX (Apple Silicon), mirroring ``notagen.model`` with a KV cache.

Layout differences from the PyTorch model, handled by :func:`from_torch_state_dict`:

* GPT-2 ``Conv1D`` weights (``[in, out]``) become ``nn.Linear`` weights (``[out, in]``), so
  ``mlx.nn.quantize`` can quantize them;
* the one-hot patch embedding becomes a ``[patch_size * vocab, hidden]`` lookup table plus bias.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..config import ModelConfig


class KVCache:
    def __init__(self, n_layers, batch, max_len, n_heads, head_dim, dtype):
        shape = (batch, max_len, n_heads, head_dim)
        self.k = [mx.zeros(shape, dtype=dtype) for _ in range(n_layers)]
        self.v = [mx.zeros(shape, dtype=dtype) for _ in range(n_layers)]
        self.batch, self.max_len = batch, max_len


class CacheView:
    def __init__(self, cache: KVCache, start: mx.array, kv_len: int, rows: mx.array | None = None):
        self.cache, self.start, self.kv_len, self.rows = cache, start, kv_len, rows


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.hidden_size
        self.n_head, self.head_dim = cfg.num_heads, c // cfg.num_heads
        self.scale = self.head_dim**-0.5
        self.c_attn = nn.Linear(c, 3 * c)
        self.c_proj = nn.Linear(c, c)

    def __call__(self, x, layer: int, view: CacheView):
        b, t, c = x.shape
        qkv = self.c_attn(x).reshape(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        cache = view.cache
        pos = view.start[:, None] + mx.arange(t)[None]  # [b,t]
        rows = view.rows if view.rows is not None else mx.arange(b)
        cache.k[layer][rows[:, None], pos] = k.astype(cache.k[layer].dtype)
        cache.v[layer][rows[:, None], pos] = v.astype(cache.v[layer].dtype)
        if view.rows is None:
            ks, vs = cache.k[layer][:, : view.kv_len], cache.v[layer][:, : view.kv_len]
        else:
            ks, vs = cache.k[layer][view.rows, : view.kv_len], cache.v[layer][view.rows, : view.kv_len]
        mask = mx.arange(view.kv_len)[None, None, :] <= pos[:, :, None]  # [b,t,kv]
        y = mx.fast.scaled_dot_product_attention(
            q.transpose(0, 2, 1, 3),
            ks.transpose(0, 2, 1, 3),
            vs.transpose(0, 2, 1, 3),
            scale=self.scale,
            mask=mask[:, None],
        )
        return self.c_proj(y.transpose(0, 2, 1, 3).reshape(b, t, c))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.hidden_size, 4 * cfg.hidden_size)
        self.c_proj = nn.Linear(4 * cfg.hidden_size, cfg.hidden_size)

    def __call__(self, x):
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def __call__(self, x, layer, view):
        x = x + self.attn(self.ln_1(x), layer, view)
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, cfg: ModelConfig, n_layers: int, n_positions: int, vocab_size: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(n_positions, cfg.hidden_size)
        self.h = [Block(cfg) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def __call__(self, inputs_embeds, view: CacheView):
        t = inputs_embeds.shape[1]
        x = inputs_embeds + self.wpe(view.start[:, None] + mx.arange(t)[None])
        for i, block in enumerate(self.h):
            x = block(x, i, view)
        return self.ln_f(x)


class PatchEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.table = mx.zeros((cfg.patch_size * cfg.vocab_size, cfg.hidden_size))
        self.bias = mx.zeros((cfg.hidden_size,))
        self._offsets = mx.arange(cfg.patch_size) * cfg.vocab_size

    def __call__(self, patches):
        return self.table[patches + self._offsets].sum(-2) + self.bias


class NotaGenMLX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = PatchEmbedding(cfg)
        self.patch_decoder = GPT2(cfg, cfg.patch_num_layers, cfg.patch_length, vocab_size=1)
        self.char_decoder = GPT2(cfg, cfg.char_num_layers, cfg.patch_size + 1, cfg.vocab_size)

    @property
    def dtype(self):
        return self.patch_embedding.table.dtype

    def encode_patches(self, patches, view: CacheView):
        return self.patch_decoder(self.patch_embedding(patches), view)

    def char_step(self, inputs_embeds, view: CacheView):
        """Logits ``[b, t, vocab]``; the LM head is tied to the char embedding."""
        return self.char_decoder.wte.as_linear(self.char_decoder(inputs_embeds, view))

    def embed_chars(self, ids):
        return self.char_decoder.wte(ids)

    def new_patch_cache(self, batch):
        c = self.cfg
        return KVCache(c.patch_num_layers, batch, c.patch_length, c.num_heads, c.head_dim, self.dtype)

    def new_char_cache(self, batch):
        c = self.cfg
        return KVCache(c.char_num_layers, batch, c.patch_size + 1, c.num_heads, c.head_dim, self.dtype)


def _map_gpt2(prefix_src: str, prefix_dst: str, sd: dict, out: dict, n_layers: int):
    out[f"{prefix_dst}.wte.weight"] = sd[f"{prefix_src}.wte.weight"]
    out[f"{prefix_dst}.wpe.weight"] = sd[f"{prefix_src}.wpe.weight"]
    for name in ("weight", "bias"):
        out[f"{prefix_dst}.ln_f.{name}"] = sd[f"{prefix_src}.ln_f.{name}"]
    for i in range(n_layers):
        s, d = f"{prefix_src}.h.{i}", f"{prefix_dst}.h.{i}"
        for ln in ("ln_1", "ln_2"):
            for name in ("weight", "bias"):
                out[f"{d}.{ln}.{name}"] = sd[f"{s}.{ln}.{name}"]
        for lin in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            out[f"{d}.{lin}.weight"] = sd[f"{s}.{lin}.weight"].T  # Conv1D [in,out] -> Linear [out,in]
            out[f"{d}.{lin}.bias"] = sd[f"{s}.{lin}.bias"]


def from_torch_state_dict(sd: dict, cfg: ModelConfig) -> dict:
    """Map a PyTorch NotaGen state dict (numpy arrays) to MLX parameter names."""
    out = {
        "patch_embedding.table": sd["patch_level_decoder.patch_embedding.weight"].T,
        "patch_embedding.bias": sd["patch_level_decoder.patch_embedding.bias"],
    }
    _map_gpt2("patch_level_decoder.base", "patch_decoder", sd, out, cfg.patch_num_layers)
    _map_gpt2("char_level_decoder.base.transformer", "char_decoder", sd, out, cfg.char_num_layers)
    return out
