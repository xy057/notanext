"""NotaGen hierarchical patch/char decoder in plain PyTorch.

Replaces the per-stage ``transformers.GPT2Model`` based copies. Parameter names match the
legacy ``state_dict`` exactly (``patch_level_decoder.base.h.0.attn.c_attn.weight``,
``char_level_decoder.base.transformer.wte.weight`` ...), so released ``.pth`` checkpoints load
unchanged. What is new:

* a static KV cache with per-row lengths, used by both decoders during generation;
* SDPA attention (Flash / memory-efficient kernels where the backend has them);
* a gather-based patch embedding that is numerically the same as one-hot x Linear.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .tokenizer import BOS_ID, SPECIAL_ID


class Conv1D(nn.Module):
    """GPT-2's transposed linear layer: ``weight`` is ``[in, out]``."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_out))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).view(*x.shape[:-1], -1)


class KVCache:
    """Preallocated per-layer K/V of shape ``[batch, max_len, heads, head_dim]``."""

    def __init__(self, n_layers, batch, max_len, n_heads, head_dim, device, dtype):
        shape = (batch, max_len, n_heads, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layers)]
        self.batch, self.max_len = batch, max_len


@dataclass
class CacheView:
    """Where a forward call reads/writes the cache.

    ``rows``  - cache rows for the batch entries (``None`` = all rows in order).
    ``start`` - per-entry position of the first new token (device tensor ``[b]``).
    ``kv_len`` - host int, ``max(start) + T``: how many cache slots to attend over.
    """

    cache: KVCache
    start: torch.Tensor
    kv_len: int
    rows: torch.Tensor | None = None


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.hidden_size
        self.n_head, self.head_dim = cfg.num_heads, c // cfg.num_heads
        self.c_attn = Conv1D(c, 3 * c)
        self.c_proj = Conv1D(c, c)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, layer: int, pad_mask=None, view: CacheView | None = None):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).view(b, t, 3, self.n_head, self.head_dim).unbind(2)  # [b,t,h,d]
        drop = self.attn_dropout if self.training else 0.0

        if view is None:
            q, k, v = (z.transpose(1, 2) for z in (q, k, v))
            if pad_mask is None:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
            else:
                causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
                mask = causal[None] & pad_mask.bool()[:, None, :]
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], dropout_p=drop)
        else:
            cache = view.cache
            pos = view.start[:, None] + torch.arange(t, device=x.device)  # [b,t]
            rows = view.rows if view.rows is not None else torch.arange(b, device=x.device)
            cache.k[layer][rows[:, None], pos] = k.to(cache.k[layer].dtype)
            cache.v[layer][rows[:, None], pos] = v.to(cache.v[layer].dtype)
            if view.rows is None:
                ks, vs = cache.k[layer][:, : view.kv_len], cache.v[layer][:, : view.kv_len]
            else:
                ks, vs = cache.k[layer][view.rows, : view.kv_len], cache.v[layer][view.rows, : view.kv_len]
            mask = torch.arange(view.kv_len, device=x.device)[None, None, :] <= pos[:, :, None]  # [b,t,kv]
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2), attn_mask=mask[:, None]
            )

        y = y.transpose(1, 2).reshape(b, t, c)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = Conv1D(cfg.hidden_size, 4 * cfg.hidden_size)
        self.c_proj = Conv1D(4 * cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, layer, pad_mask=None, view=None):
        x = x + self.attn(self.ln_1(x), layer, pad_mask, view)
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    """``transformers.GPT2Model`` with the same parameter names, driven by ``inputs_embeds``."""

    def __init__(self, cfg: ModelConfig, n_layers: int, n_positions: int, vocab_size: int):
        super().__init__()
        self.n_layers = n_layers
        self.wte = nn.Embedding(vocab_size, cfg.hidden_size)
        self.wpe = nn.Embedding(n_positions, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList(Block(cfg) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self._init_weights()

    def _init_weights(self):
        # Matches GPT2PreTrainedModel._init_weights, including the scaled residual projections.
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * self.n_layers))

    def forward(self, inputs_embeds, pad_mask=None, view: CacheView | None = None):
        t = inputs_embeds.shape[1]
        if view is None:
            pos = torch.arange(t, device=inputs_embeds.device)[None]
        else:
            pos = view.start[:, None] + torch.arange(t, device=inputs_embeds.device)
        x = self.drop(inputs_embeds + self.wpe(pos))
        for i, block in enumerate(self.h):
            x = block(x, i, pad_mask, view)
        return self.ln_f(x)


class GPT2LMHead(nn.Module):
    """``transformers.GPT2LMHeadModel`` layout: ``transformer`` + weight-tied ``lm_head``."""

    def __init__(self, cfg: ModelConfig, n_layers: int, n_positions: int, vocab_size: int):
        super().__init__()
        self.transformer = GPT2(cfg, n_layers, n_positions, vocab_size)
        self.lm_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, inputs_embeds, pad_mask=None, view=None):
        return self.lm_head(self.transformer(inputs_embeds, pad_mask, view))


class PatchLevelDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Linear(cfg.patch_size * cfg.vocab_size, cfg.hidden_size)
        nn.init.normal_(self.patch_embedding.weight, std=0.02)
        # vocab_size=1: the legacy GPT2Model had an unused one-row wte; kept for state-dict compatibility.
        self.base = GPT2(cfg, cfg.patch_num_layers, cfg.patch_length, vocab_size=1)
        self.register_buffer("_offsets", torch.arange(cfg.patch_size) * cfg.vocab_size, persistent=False)

    def embed(self, patches):
        """``one_hot(patches) @ W.T + b`` computed as a gather-and-sum over the 16 slots."""
        idx = patches + self._offsets  # [b, n, patch_size]
        w = self.patch_embedding.weight.t()  # [patch_size*vocab, hidden]
        return F.embedding(idx, w).sum(-2) + self.patch_embedding.bias

    def forward(self, patches, masks=None, view: CacheView | None = None):
        """``patches``: ``[b, n, patch_size]`` ids. Returns ``[b, n, hidden]``."""
        return self.base(self.embed(patches), pad_mask=masks, view=view)


class CharLevelDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.base = GPT2LMHead(cfg, cfg.char_num_layers, cfg.patch_size + 1, cfg.vocab_size)

    def _inputs(self, encoded_patches, target_patches):
        target_patches = torch.cat((torch.full_like(target_patches[:, :1], BOS_ID), target_patches), dim=1)
        labels = target_patches.masked_fill(target_patches == SPECIAL_ID, -100)
        target_masks = (labels != -100).long()
        emb = self.base.transformer.wte(target_patches)
        emb = torch.cat((encoded_patches.unsqueeze(1), emb[:, 1:]), dim=1)
        return target_patches, labels, target_masks, emb

    def forward(self, encoded_patches, target_patches, patch_sampling_batch_size: int = 0, generator=None):
        """Mean next-char cross-entropy over the non-pad chars of every patch (legacy ``.loss``)."""
        target_patches, labels, target_masks, emb = self._inputs(encoded_patches, target_patches)
        if 0 < patch_sampling_batch_size < target_patches.shape[0]:
            sel = torch.randperm(target_patches.shape[0], generator=generator)[:patch_sampling_batch_size]
            sel = sel.sort().values.to(emb.device)
            labels, target_masks, emb = labels[sel], target_masks[sel], emb[sel]
        logits = self.base(emb, pad_mask=target_masks)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100
        )

    def sequence_logps(self, encoded_patches, target_patches):
        """Summed log-probability of every non-pad char (the RL/DPO copy's ``forward``)."""
        target_patches, labels, target_masks, emb = self._inputs(encoded_patches, target_patches)
        logits = self.base(emb, pad_mask=target_masks)[:, :-1].float()
        token_logps = logits.log_softmax(-1).gather(-1, target_patches[:, 1:, None]).squeeze(-1)
        return token_logps[target_masks[:, 1:] == 1].sum()

    def step(self, inputs_embeds, view: CacheView):
        return self.base(inputs_embeds, view=view)


class NotaGenLMHeadModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_level_decoder = PatchLevelDecoder(cfg)
        self.char_level_decoder = CharLevelDecoder(cfg)

    @property
    def device(self):
        return self.patch_level_decoder.patch_embedding.weight.device

    @property
    def dtype(self):
        return self.patch_level_decoder.patch_embedding.weight.dtype

    def _encode_for_targets(self, patches, masks):
        patches = patches.reshape(len(patches), -1, self.cfg.patch_size)
        encoded = self.patch_level_decoder(patches, masks)
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks = masks.clone()
        masks[:, 0] = 0
        return encoded[left_shift_masks == 1], patches[masks == 1]

    def forward(self, patches, masks, patch_sampling_batch_size: int = 0):
        """Training loss. ``patches``: ``[b, n*patch_size]`` or ``[b, n, patch_size]``; ``masks``: ``[b, n]``."""
        encoded, targets = self._encode_for_targets(patches, masks)
        return self.char_level_decoder(encoded, targets, patch_sampling_batch_size)

    def sequence_logps(self, patches, masks):
        encoded, targets = self._encode_for_targets(patches, masks)
        return self.char_level_decoder.sequence_logps(encoded, targets)

    # ---- generation primitives (used by notagen.generation.TorchBackend) ----

    def new_patch_cache(self, batch, dtype=None) -> KVCache:
        c = self.cfg
        return KVCache(
            c.patch_num_layers, batch, c.patch_length, c.num_heads, c.head_dim, self.device, dtype or self.dtype
        )

    def new_char_cache(self, batch, dtype=None) -> KVCache:
        c = self.cfg
        return KVCache(
            c.char_num_layers, batch, c.patch_size + 1, c.num_heads, c.head_dim, self.device, dtype or self.dtype
        )

    def gradient_checkpointing_enable(self):
        for dec in (self.patch_level_decoder.base, self.char_level_decoder.base.transformer):
            for block in dec.h:
                block._orig_forward = block.forward
                block.forward = _checkpointed(block._orig_forward)


def _checkpointed(fn):
    from torch.utils.checkpoint import checkpoint

    def wrapper(*args, **kw):
        if torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, **kw)
        return fn(*args, **kw)

    return wrapper
