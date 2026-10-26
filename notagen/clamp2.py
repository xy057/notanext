"""CLaMP 2 music/text features, in-process and batched (replaces ``clamp2/extract_clamp2.py``).

The model is the one in ``clamp2/utils.py`` (M3 patch encoder = BERT over 64-char bar patches, plus
XLM-RoBERTa for text, each followed by a 768-d projection). Parameter names match its checkpoint.
What changed:

* features are computed in memory for lists of strings, with segments of many pieces batched
  together, on CUDA, MPS or CPU, so the metrics and RL rewards can call it directly;
* the patch embedding is a gather-and-sum instead of a one-hot matmul, and the forward no longer
  forces CPU float tensors (``.type(torch.FloatTensor)``) on every call;
* XLM-RoBERTa is built from its config: its pre-trained weights are overwritten by the CLaMP 2
  checkpoint anyway, so only the ~14 MB tokenizer/config are needed (not the 1.1 GB weights);
* text lines are de-duplicated in order instead of through ``set()``, whose ordering changes with
  Python's hash seed.

Feature semantics are unchanged: pieces longer than 512 patches are split into 512-patch segments
(the last one taken from the end), each segment is mean-pooled and projected, and the segment
features are averaged with weights equal to their new-patch counts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

PATCH_SIZE = 64
PATCH_LENGTH = 512
HIDDEN = 768
MAX_TEXT_LENGTH = 128
CHECKPOINT_NAME = (
    "weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_"
    "t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth"
)


class M3Patchilizer:
    """Bar-level patchilizer from ``clamp2/utils.py`` (ABC path only)."""

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.mask_token_id = 3

    def split_bars(self, body):
        bars = re.split(self.regexPattern, "".join(body))
        bars = list(filter(None, bars))
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        return [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]

    def bar2patch(self, bar, patch_size=PATCH_SIZE):
        patch = [self.bos_token_id] + [ord(c) for c in bar] + [self.eos_token_id]
        patch = patch[:patch_size]
        return patch + [self.pad_token_id] * (patch_size - len(patch))

    def encode(self, item, patch_size=PATCH_SIZE, add_special_patches=False):
        from unidecode import unidecode

        item = unidecode(item)
        lines = re.findall(r".*?\n|.*$", item)
        lines = list(filter(None, lines))
        patches = []
        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ":") or line.startswith("%%")):
                patches.append(line)
            else:
                bars = self.split_bars(line)
                if bars:
                    bars[-1] += "\n"
                    patches.extend(bars)
        if add_special_patches:
            patches = [chr(self.bos_token_id) * patch_size] + patches + [chr(self.eos_token_id) * patch_size]
        return [self.bar2patch(p, patch_size) for p in patches]


class M3PatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import BertConfig, BertModel

        cfg = BertConfig(
            vocab_size=1,
            hidden_size=HIDDEN,
            num_hidden_layers=12,
            num_attention_heads=HIDDEN // 64,
            intermediate_size=HIDDEN * 4,
            max_position_embeddings=PATCH_LENGTH,
        )
        self.patch_embedding = nn.Linear(PATCH_SIZE * 128, HIDDEN)
        self.base = BertModel(cfg)
        self.register_buffer("_offsets", torch.arange(PATCH_SIZE) * 128, persistent=False)

    def forward(self, patches, masks):
        # clamp: chars >= 128 cannot occur after unidecode, but stay safe for arbitrary input
        idx = patches.clamp(max=127) + self._offsets
        emb = F.embedding(idx, self.patch_embedding.weight.t()).sum(-2) + self.patch_embedding.bias
        return self.base(inputs_embeds=emb, attention_mask=masks)["last_hidden_state"]


class CLaMP2(nn.Module):
    def __init__(self, text_config_dir: str | Path):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.text_model = AutoModel.from_config(
            AutoConfig.from_pretrained(str(text_config_dir)), add_pooling_layer=True
        )
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, HIDDEN)
        self.music_model = M3PatchEncoder()
        self.music_proj = nn.Linear(HIDDEN, HIDDEN)

    @staticmethod
    def _pool(x, masks):
        m = masks.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1)

    def music_segments(self, patches, masks):
        return self.music_proj(self._pool(self.music_model(patches, masks), masks))

    def text_segments(self, ids, masks):
        h = self.text_model(ids, attention_mask=masks)["last_hidden_state"]
        return self.text_proj(self._pool(h, masks))


def strip_prompt_comments(abc: str) -> str:
    """Drop ``%period``-style comment lines (keeps ``%%`` directives), as ``extract_clamp2.py`` does."""
    return "".join(l for l in abc.splitlines(keepends=True) if not (l.startswith("%") and not l.startswith("%%")))


def _segments(n: int, max_len: int) -> list[tuple[int, int, int]]:
    """(start, end, weight) per segment: full chunks, then the last ``max_len`` items weighted by the remainder."""
    segs = [(i, min(i + max_len, n), max_len) for i in range(0, n, max_len)]
    rem = n % max_len
    if n > max_len or rem:
        s, e, _ = segs[-1]
        segs[-1] = (max(0, n - max_len), n, rem if rem else max_len)
    return segs


@dataclass
class CLaMP2Encoder:
    model: CLaMP2
    tokenizer: object
    device: torch.device
    batch_size: int = 16

    @classmethod
    def load(cls, weights_dir: str | Path = "weights/clamp2", device="auto", dtype=torch.float32, batch_size=16):
        from transformers import AutoTokenizer

        from .device import pick_device

        weights_dir = Path(weights_dir)
        dev = pick_device(device)
        model = CLaMP2(weights_dir / "xlm-roberta-base")
        ckpt = torch.load(weights_dir / CHECKPOINT_NAME, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in ckpt["model"].items() if not k.startswith("loss_fn")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # position_ids buffers differ between transformers versions; nothing else may be missing.
        missing = [k for k in missing if not k.endswith("position_ids")]
        unexpected = [k for k in unexpected if not k.endswith("position_ids")]
        if missing or unexpected:
            raise RuntimeError(f"CLaMP 2 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        model = model.to(dev, dtype).eval()
        tok = AutoTokenizer.from_pretrained(str(weights_dir / "xlm-roberta-base"))
        return cls(model, tok, dev, batch_size)

    # ---- encoding --------------------------------------------------------------------------

    def _run(self, items: list[torch.Tensor], max_len: int, pad, fn) -> np.ndarray:
        """Weighted-average segment features for each item (``items`` are id tensors)."""
        jobs = []  # (item index, weight, segment tensor)
        for i, ids in enumerate(items):
            for s, e, w in _segments(len(ids), max_len):
                jobs.append((i, w, ids[s:e]))
        feats = torch.zeros(len(items), HIDDEN, dtype=torch.float64)
        weights = torch.zeros(len(items), 1, dtype=torch.float64)
        for b in range(0, len(jobs), self.batch_size):
            chunk = jobs[b : b + self.batch_size]
            x = torch.stack([torch.cat([seg, pad.expand(max_len - len(seg), *pad.shape[1:])]) for _, _, seg in chunk])
            m = torch.stack([torch.arange(max_len) < len(seg) for _, _, seg in chunk]).long()
            with torch.no_grad():
                out = fn(x.to(self.device), m.to(self.device)).float().cpu().double()
            for (i, w, _), f in zip(chunk, out):
                feats[i] += w * f
                weights[i] += w
        return (feats / weights).float().numpy()

    def music_features(self, abc_texts: list[str]) -> np.ndarray:
        """Features of interleaved ABC pieces (``[n, 768]``)."""
        p = M3Patchilizer()
        items = [torch.tensor(p.encode(strip_prompt_comments(t), add_special_patches=True)) for t in abc_texts]
        pad = torch.zeros(1, PATCH_SIZE, dtype=torch.long)
        return self._run(items, PATCH_LENGTH, pad, self.model.music_segments)

    def text_features(self, texts: list[str]) -> np.ndarray:
        items = []
        for t in texts:
            lines = list(dict.fromkeys(l for l in t.split("\n") if l))
            ids = self.tokenizer(self.tokenizer.sep_token.join(lines), return_tensors="pt")["input_ids"][0]
            items.append(ids)
        pad = torch.full((1,), self.tokenizer.pad_token_id, dtype=torch.long)
        return self._run(items, MAX_TEXT_LENGTH, pad, self.model.text_segments)


def prompt_text(period: str, composer: str, instrumentation: str) -> str:
    """Text query for a NotaGen prompt. CLaMP 2 was trained on free-text metadata, so a short
    natural description works better than the raw ``%`` lines."""
    return f"{composer}\n{period} period\n{instrumentation}"


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return a @ b.T


# ---- CLI: ``notagen clamp2 extract|score`` (replaces extract_clamp2.py / statistics.py) ----------


def extract_folder(enc: CLaMP2Encoder, input_dir, output_dir, chunk: int = 64) -> int:
    """Write ``<output_dir>/<same relative path>.npy`` (shape ``[1, 768]``) for every ``.abc``/``.txt``,
    skipping existing outputs, as the legacy extractor did."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    todo = []
    for f in sorted(input_dir.rglob("*")):
        if f.suffix not in (".abc", ".txt"):
            continue
        out = output_dir / f.relative_to(input_dir).with_suffix(".npy")
        if not out.exists():
            todo.append((f, out))
    done = 0
    for kind in (".abc", ".txt"):
        items = [(f, o) for f, o in todo if f.suffix == kind]
        for i in range(0, len(items), chunk):
            batch = items[i : i + chunk]
            texts = [f.read_text(encoding="utf-8") for f, _ in batch]
            feats = enc.music_features(texts) if kind == ".abc" else enc.text_features(texts)
            for (_, o), v in zip(batch, feats):
                o.parent.mkdir(parents=True, exist_ok=True)
                np.save(o, v[None])
                done += 1
    return done


def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser("notagen clamp2")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="features for a folder of interleaved .abc (or .txt) files")
    e.add_argument("input_dir")
    e.add_argument("output_dir")
    s = sub.add_parser("score", help="paper's CLaMP 2 score: generated features vs ground-truth mean")
    s.add_argument("gt_dir")
    s.add_argument("gen_dir")
    for p in (e, s):
        p.add_argument("--weights", default="weights/clamp2")
        p.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        enc = CLaMP2Encoder.load(a.weights, device=a.device)
        print(f"wrote {extract_folder(enc, a.input_dir, a.output_dir)} feature files")
    else:
        from .metrics import clamp2_score, mean_pairwise_distance, near_copy_rate

        load = lambda d: np.stack([np.load(f).reshape(-1) for f in sorted(Path(d).rglob("*.npy"))])  # noqa: E731
        gt, gen = load(a.gt_dir), load(a.gen_dir)
        print(
            json.dumps(
                {
                    "clamp2_score": clamp2_score(gt, gen),
                    "diversity": mean_pairwise_distance(gen),
                    "near_copy_rate": near_copy_rate(gt, gen),
                    "n_gt": len(gt),
                    "n_gen": len(gen),
                },
                indent=2,
            )
        )
