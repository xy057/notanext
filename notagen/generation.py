"""Batched, KV-cached NotaGen generation.

The per-piece control flow (the ``[r:0/`` tune-body hack, EOS handling, stream-mode re-seeding,
length/time limits) is a faithful port of ``inference/inference.py``. Two things differ:

* every decoder runs incrementally against a KV cache instead of re-encoding the whole context
  for each patch / char;
* up to ``batch_size`` pieces are generated together, and a finished row is refilled with the
  next piece straight away (continuous batching).

The loop only talks to a *backend* (``prefill`` / ``feed`` / ``sample``), so the PyTorch and MLX
implementations share it.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch

from .tokenizer import BOS_ID, EOS_ID, SPECIAL_ID, Patchilizer

R0 = [ord(c) for c in "[r:0/"]


class Backend(Protocol):
    patch_size: int
    patch_length: int

    def start(self, batch_size: int) -> None: ...
    def prefill(self, row: int, patches: list[list[int]]) -> None: ...
    def feed(self, patches: dict[int, list[int]]) -> None: ...
    def sample(self, rows: list[int], prefixes: list[list[int]]) -> list[list[int]]: ...


@dataclass
class Piece:
    text: str
    seconds: float
    ok: bool
    reason: str = ""
    patches: list[list[int]] | None = None  # full id history (prompt + sampled), when not re-seeded


@dataclass
class _Row:
    index: int
    chars: list[str]
    n_patches: int
    start_time: float
    prefix: list[int] = field(default_factory=list)
    pending: list[int] | None = None
    tunebody: bool = False
    cut_index: int | None = None
    history: list[list[int]] | None = None


class Generator:
    def __init__(
        self,
        backend: Backend,
        patchilizer: Patchilizer,
        max_chars: int = 102400,
        time_limit: float = 20 * 60,
        on_text=None,
        stream: bool = True,
    ):
        self.backend = backend
        self.tok = patchilizer
        self.max_chars = max_chars
        self.time_limit = time_limit
        self.on_text = on_text  # callback(row_index, text) for streaming output
        self.stream = stream  # False: a piece that fills the context fails instead of re-seeding (RL)

    def generate(self, prompt_lines: list[str], num_pieces: int, batch_size: int = 1) -> Iterator[Piece]:
        """Yield ``num_pieces`` finished pieces (successful or failed) in completion order."""
        b = self.backend
        batch_size = max(1, min(batch_size, num_pieces))
        b.start(batch_size)
        prompt = self.tok.encode_prompt(prompt_lines)
        started = 0
        rows: dict[int, _Row] = {}

        def launch(slot: int):
            nonlocal started
            b.prefill(slot, prompt)
            rows[slot] = _Row(
                started, list("".join(prompt_lines)), len(prompt), time.time(), history=[list(p) for p in prompt]
            )
            started += 1

        for slot in range(batch_size):
            launch(slot)

        while rows:
            feed = {s: r.pending for s, r in rows.items() if r.pending is not None}
            if feed:
                b.feed(feed)
                for s in feed:
                    rows[s].pending = None

            slots = list(rows)
            preds = dict(zip(slots, b.sample(slots, [rows[s].prefix for s in slots])))

            hack = [s for s in slots if not rows[s].tunebody and self.tok.decode([preds[s]]).startswith("[r:")]
            for s in hack:
                rows[s].tunebody = True
            hack = [s for s in hack if len(rows[s].prefix) + len(R0) < self.tok.patch_size]
            if hack:
                again = b.sample(hack, [rows[s].prefix + R0 for s in hack])
                for s, g in zip(hack, again):
                    preds[s] = R0 + g

            for s in slots:
                done = self._advance(rows[s], preds[s], s)
                if done is not None:
                    yield done
                    del rows[s]
                    if started < num_pieces:
                        launch(s)

    def _advance(self, row: _Row, pred: list[int], slot: int) -> Piece | None:
        tok = self.tok
        if len(pred) >= 2 and pred[0] == BOS_ID and pred[1] == EOS_ID:
            if row.history is not None:
                row.history.append(row.prefix + [BOS_ID, EOS_ID] + [SPECIAL_ID] * (len(pred) - 2))
            return self._finish(row, True)

        text = tok.decode([pred])
        row.chars.extend(text)
        if self.on_text:
            self.on_text(row.index, text)

        ended = False
        for j in range(len(pred)):
            if ended:
                pred[j] = SPECIAL_ID
            if pred[j] == EOS_ID:
                ended = True

        patch = row.prefix + pred
        row.prefix = []
        row.n_patches += 1
        if row.history is not None:
            row.history.append(list(patch))

        if len(row.chars) > self.max_chars:
            return self._finish(row, False, "max_chars")
        if time.time() - row.start_time > self.time_limit:
            return self._finish(row, False, "time_limit")

        if row.n_patches >= tok.patch_length:
            if not self.stream:
                return self._finish(row, False, "context_full")
            return self._reseed(row, slot)
        row.pending = patch
        return None

    def _reseed(self, row: _Row, slot: int) -> Piece | None:
        """Stream mode: restart the context from metadata + the latest tune-body lines."""
        abc_code = "".join(row.chars)
        abc_lines = abc_code.split("\n")

        tunebody_index = None
        for i, line in enumerate(abc_lines):
            if line.startswith("[r:") or line.startswith("[V:"):
                tunebody_index = i
                break
        if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
            # The legacy loop ``break``s here and keeps the piece as it is.
            return self._finish(row, True, "context_full")

        metadata_lines = [line + "\n" for line in abc_lines[:tunebody_index]]
        tunebody_lines = abc_lines[tunebody_index:]
        if not abc_code.endswith("\n"):
            tunebody_lines = [l + "\n" for l in tunebody_lines[:-1]] + [tunebody_lines[-1]]
        else:
            tunebody_lines = [l + "\n" for l in tunebody_lines]

        if row.cut_index is None:
            row.cut_index = len(tunebody_lines) // 2

        ids = self.tok.encode_generate("".join(metadata_lines + tunebody_lines[-row.cut_index :]))
        prefix = ids[-1] if len(ids[-1]) < self.tok.patch_size else []
        complete = ids[:-1] if prefix else ids
        self.backend.prefill(slot, complete)
        row.history = None  # the context was rebuilt; the sampled history no longer is one sequence
        row.n_patches = len(complete)
        row.prefix = prefix
        return None

    def _finish(self, row: _Row, ok: bool, reason: str = "") -> Piece:
        return Piece("".join(row.chars), time.time() - row.start_time, ok, reason, row.history)


class TorchBackend:
    """Runs a ``NotaGenLMHeadModel`` with KV caches on any torch device."""

    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler
        self.patch_size = model.cfg.patch_size
        self.patch_length = model.cfg.patch_length
        self.device = model.device

    @torch.inference_mode()
    def start(self, batch_size: int) -> None:
        self.cache = self.model.new_patch_cache(batch_size)
        self.lens = [0] * batch_size
        self.hidden = torch.zeros(batch_size, self.model.cfg.hidden_size, device=self.device, dtype=self.model.dtype)

    @torch.inference_mode()
    def prefill(self, row: int, patches: list[list[int]]) -> None:
        from .model import CacheView

        x = torch.tensor([patches], device=self.device)
        view = CacheView(
            self.cache,
            torch.zeros(1, dtype=torch.long, device=self.device),
            len(patches),
            rows=torch.tensor([row], device=self.device),
        )
        out = self.model.patch_level_decoder(x, view=view)
        self.hidden[row] = out[0, -1]
        self.lens[row] = len(patches)

    @torch.inference_mode()
    def feed(self, patches: dict[int, list[int]]) -> None:
        from .model import CacheView

        bsz = self.cache.batch
        x = torch.zeros(bsz, 1, self.patch_size, dtype=torch.long)
        for r, p in patches.items():
            assert self.lens[r] < self.patch_length, "patch context overflow"
            x[r, 0] = torch.tensor(p)
        start = [min(l, self.patch_length - 1) for l in self.lens]
        view = CacheView(self.cache, torch.tensor(start, device=self.device), max(start) + 1)
        out = self.model.patch_level_decoder(x.to(self.device), view=view)
        idx = torch.tensor(list(patches), device=self.device)
        self.hidden[idx] = out[idx, 0]
        for r in patches:
            self.lens[r] += 1

    @torch.inference_mode()
    def sample(self, rows: list[int], prefixes: list[list[int]]) -> list[list[int]]:
        from .model import CacheView

        model, ps, dev = self.model, self.patch_size, self.device
        wte = model.char_level_decoder.base.transformer.wte
        b = len(rows)
        cache = model.new_char_cache(b)
        need = [ps - len(p) for p in prefixes]
        logits = torch.empty(b, model.cfg.vocab_size, device=dev, dtype=model.dtype)
        enc = self.hidden[torch.tensor(rows, device=dev)]

        # Prefill [encoded_patch, *prefix], grouped by prefix length.
        by_len: dict[int, list[int]] = {}
        for i, p in enumerate(prefixes):
            by_len.setdefault(len(p), []).append(i)
        for plen, members in by_len.items():
            idx = torch.tensor(members, device=dev)
            x = enc[idx].unsqueeze(1)
            if plen:
                x = torch.cat([x, wte(torch.tensor([prefixes[i] for i in members], device=dev))], dim=1)
            view = CacheView(cache, torch.zeros(len(members), dtype=torch.long, device=dev), plen + 1, rows=idx)
            logits[idx] = model.char_level_decoder.step(x, view)[:, -1]

        # Tokens stay on the device until the patch is complete: one host sync per patch, not per char.
        pos = [len(p) + 1 for p in prefixes]
        steps = []
        for step in range(max(need)):
            tokens = self.sampler(logits)
            steps.append(tokens)
            if step == max(need) - 1:
                break
            start = [min(p, ps) for p in pos]
            view = CacheView(cache, torch.tensor(start, device=dev), max(start) + 1)
            logits = model.char_level_decoder.step(wte(tokens)[:, None], view)[:, -1]
            pos = [p + 1 for p in pos]
        host = torch.stack(steps, 1).tolist()
        return [host[i][: need[i]] for i in range(b)]
