"""The new Generator must reproduce inference/inference.py's control flow and outputs."""

import random

import numpy as np
import pytest
import torch
from conftest import TINY

from notagen.generation import Generator, TorchBackend
from notagen.sampling import LegacySampler, SamplingParams
from notagen.tokenizer import Patchilizer

PROMPT = ["%Classical\n", "%Mozart, Wolfgang Amadeus\n", "%Keyboard\n", "L:1/8\n", "M:4/4\n", "K:G\n"]


def legacy_loop(generate, patchilizer, prompt_lines, patch_size, patch_length, max_chars):
    """inference_patch() from inference/inference.py for one piece, with the model call injected."""
    bos_patch = [patchilizer.bos_token_id] * (patch_size - 1) + [patchilizer.eos_token_id]
    prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
    byte_list = list("".join(prompt_lines))
    prompt_patches = [
        [ord(c) for c in patch] + [patchilizer.special_token_id] * (patch_size - len(patch)) for patch in prompt_patches
    ]
    prompt_patches.insert(0, bos_patch)
    input_patches = torch.tensor(prompt_patches).reshape(1, -1)
    failure_flag = False
    end_flag = False
    cut_index = None
    tunebody_flag = False
    while True:
        predicted_patch = generate(input_patches.unsqueeze(0))
        if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith("[r:"):
            tunebody_flag = True
            r0_patch = torch.tensor([ord(c) for c in "[r:0/"]).unsqueeze(0)
            temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
            predicted_patch = generate(temp_input_patches.unsqueeze(0))
            predicted_patch = [ord(c) for c in "[r:0/"] + predicted_patch
        if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
            end_flag = True
            break
        next_patch = patchilizer.decode([predicted_patch])
        for char in next_patch:
            byte_list.append(char)
        patch_end_flag = False
        for j in range(len(predicted_patch)):
            if patch_end_flag:
                predicted_patch[j] = patchilizer.special_token_id
            if predicted_patch[j] == patchilizer.eos_token_id:
                patch_end_flag = True
        predicted_patch = torch.tensor([predicted_patch])
        input_patches = torch.cat([input_patches, predicted_patch], dim=1)
        if len(byte_list) > max_chars:
            failure_flag = True
            break
        if input_patches.shape[1] >= patch_length * patch_size and not end_flag:
            abc_code = "".join(byte_list)
            abc_lines = abc_code.split("\n")
            tunebody_index = None
            for i, line in enumerate(abc_lines):
                if line.startswith("[r:") or line.startswith("[V:"):
                    tunebody_index = i
                    break
            if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                break
            metadata_lines = abc_lines[:tunebody_index]
            tunebody_lines = abc_lines[tunebody_index:]
            metadata_lines = [line + "\n" for line in metadata_lines]
            if not abc_code.endswith("\n"):
                tunebody_lines = [tunebody_lines[i] + "\n" for i in range(len(tunebody_lines) - 1)] + [
                    tunebody_lines[-1]
                ]
            else:
                tunebody_lines = [tunebody_lines[i] + "\n" for i in range(len(tunebody_lines))]
            if cut_index is None:
                cut_index = len(tunebody_lines) // 2
            abc_code_slice = "".join(metadata_lines + tunebody_lines[-cut_index:])
            input_patches = patchilizer.encode_generate(abc_code_slice)
            input_patches = [item for sublist in input_patches for item in sublist]
            input_patches = torch.tensor([input_patches]).reshape(1, -1)
    return "".join(byte_list), not failure_flag


# ---------- scripted fake model: same deterministic function of the flattened context ----------

CHUNKS = ["abc ", "def|", "[V:2]C,8|\n", "\n", "[r:3/2][V:1]", "gfed|", "A2B2 ", "c'4|]\n"]


def scripted(flat: list[int], need: int) -> list[int]:
    """Deterministic 'model': next chars depend only on the full flattened id context."""
    rnd = random.Random(hash(tuple(flat)) & 0xFFFFFFFF)
    if len(flat) > 16 * 20 and rnd.random() < 0.01:
        return ([1, 2] + [2] * 14)[:need]
    s = "".join(rnd.choice(CHUNKS) for _ in range(4))
    ids = [ord(c) for c in s][:need]
    if len(ids) < need:
        ids += [2] + [0] * (need - len(ids) - 1)
    return ids


class ScriptedBackend:
    def __init__(self, ps, pl):
        self.patch_size, self.patch_length = ps, pl

    def start(self, b):
        self.ctx = [[] for _ in range(b)]

    def prefill(self, row, patches):
        self.ctx[row] = [i for p in patches for i in p]

    def feed(self, patches):
        for r, p in patches.items():
            assert len(p) == self.patch_size
            self.ctx[r] += p

    def sample(self, rows, prefixes):
        return [scripted(self.ctx[r] + p, self.patch_size - len(p)) for r, p in zip(rows, prefixes)]


def scripted_legacy_generate(input_patches):
    flat = input_patches.flatten().tolist()
    return scripted(flat, 16 - len(flat) % 16)


@pytest.mark.parametrize("patch_length", [24, 40, 1024])
def test_control_flow_matches_legacy(patch_length):
    tok = Patchilizer(patch_length=patch_length)
    for seed in range(6):
        prompt = PROMPT + [f"%seed {seed}\n"]
        want, ok = legacy_loop(scripted_legacy_generate, tok, prompt, 16, patch_length, 4000)
        gen = Generator(ScriptedBackend(16, patch_length), tok, max_chars=4000)
        got = next(gen.generate(prompt, 1))
        assert got.text == want and got.ok == ok


def test_batched_scripted_pieces_equal_unbatched():
    tok = Patchilizer(patch_length=24)
    prompt = PROMPT
    single = Generator(ScriptedBackend(16, 24), tok, max_chars=3000)
    want = next(single.generate(prompt, 1)).text
    batched = Generator(ScriptedBackend(16, 24), tok, max_chars=3000)
    pieces = list(batched.generate(prompt, 5, batch_size=3))
    assert len(pieces) == 5 and all(p.text == want for p in pieces)


# ---------- real (random-weight) model: KV cache vs legacy full recompute ----------


@pytest.mark.parametrize("top_k,top_p,temp", [(9, 0.9, 1.2), (1, 1.0, 1.0)])
def test_real_model_matches_legacy(tiny_pair, top_k, top_p, temp):
    mod, old, new = tiny_pair
    tok = Patchilizer(patch_length=TINY.patch_length)
    prompt = PROMPT + ["[r:0/3][V:1]abcd efga|[V:2]C,8|\n", "[r:1/2][V:1]bagf edcB|[V:2]G,,8|\n"]

    def old_generate(x):
        with torch.no_grad():
            return old.generate(x, top_k=top_k, top_p=top_p, temperature=temp)

    for seed in range(3):
        np.random.seed(seed)
        want, ok = legacy_loop(old_generate, mod.Patchilizer(), prompt, 16, TINY.patch_length, 1500)
        np.random.seed(seed)
        backend = TorchBackend(new, LegacySampler(SamplingParams(top_k, top_p, temp)))
        got = next(Generator(backend, tok, max_chars=1500).generate(prompt, 1))
        assert got.text == want and got.ok == ok
