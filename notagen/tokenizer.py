"""Character-level ABC patch tokenizer (one copy, replacing the per-stage drifted ones).

Behaviour is kept identical to the legacy ``Patchilizer`` in each stage. The places
where the copies diverged are explicit options rather than silently unified:

* ``tunebody_detect`` - how ``encode_train`` finds the first tune-body line:
    ``"contains_v"``   ``'[V:' in line``           (finetune / inference / gradio copies)
    ``"startswith_v"`` ``line.startswith('[V:')``  (pretrain copy)
    ``"startswith_r"`` ``line.startswith('[r:')``  (RL copy: input already carries
                        ``[r:i/n]`` tags, so no tags are added)
"""

from __future__ import annotations

import bisect
import random
import re
from typing import Literal

SPECIAL_ID = 0
BOS_ID = 1
EOS_ID = 2

TunebodyDetect = Literal["contains_v", "startswith_v", "startswith_r"]


class Patchilizer:
    delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]

    def __init__(
        self,
        patch_size: int = 16,
        patch_length: int = 1024,
        stream: bool = True,
        tunebody_detect: TunebodyDetect = "contains_v",
    ):
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.stream = stream
        self.tunebody_detect = tunebody_detect
        self.regexPattern = "(" + "|".join(map(re.escape, self.delimiters)) + ")"
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.special_token_id = SPECIAL_ID

    @classmethod
    def from_config(cls, cfg, **kw) -> Patchilizer:
        return cls(patch_size=cfg.patch_size, patch_length=cfg.patch_length, stream=cfg.stream, **kw)

    @property
    def bos_patch(self) -> list[int]:
        return [BOS_ID] * (self.patch_size - 1) + [EOS_ID]

    @property
    def eos_patch(self) -> list[int]:
        return [BOS_ID] + [EOS_ID] * (self.patch_size - 1)

    def split_bars(self, body_lines):
        """Split a body of music into individual bars."""
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [
                            line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)
                        ]
                    if "V" not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  # absorb the trailing "barline + \n" piece
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except Exception:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=None, generate_last=False):
        patch_size = patch_size or self.patch_size
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(EOS_ID)
        return [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]

    def patch2chars(self, patch):
        """Convert a patch of ids into text, stopping at EOS.

        Ids 0/1 before EOS are kept as control characters, as in the legacy code.
        """
        out = ""
        for idx in patch:
            if idx == EOS_ID:
                break
            out += chr(idx)
        return out

    def patchilize_metadata(self, metadata_lines):
        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line)
        return metadata_patches

    def patchilize_tunebody(self, tunebody_lines, encode_mode="train"):
        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == "train":
            for bar in bars:
                tunebody_patches += self.split_patches(bar)
        elif encode_mode == "generate":
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar)
            tunebody_patches += self.split_patches(bars[-1], generate_last=True)
        return tunebody_patches

    def _is_tunebody(self, line: str) -> bool:
        if self.tunebody_detect == "contains_v":
            return "[V:" in line
        if self.tunebody_detect == "startswith_v":
            return line.startswith("[V:")
        return line.startswith("[r:")

    def encode_train(
        self, abc_text, patch_length=None, patch_size=None, add_special_patches=True, cut=True, rng=random
    ):
        patch_length = patch_length or self.patch_length
        patch_size = patch_size or self.patch_size

        lines = abc_text.split("\n")
        lines = list(filter(None, lines))
        lines = [line + "\n" for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if self._is_tunebody(line):
                tunebody_index = i
                break

        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        if self.stream and self.tunebody_detect != "startswith_r":
            tunebody_lines = [
                "[r:" + str(line_index) + "/" + str(len(tunebody_lines) - line_index - 1) + "]" + line
                for line_index, line in enumerate(tunebody_lines)
            ]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode="train")

        if add_special_patches:
            bos_patch = chr(BOS_ID) * (patch_size - 1) + chr(EOS_ID)
            eos_patch = chr(BOS_ID) + chr(EOS_ID) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream and len(metadata_patches) + len(tunebody_patches) > patch_length:
            available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if "\n" in patch]
            line_index_for_cut_index = list(range(len(available_cut_indexes)))
            end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
            biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
            available_cut_indexes = available_cut_indexes[: biggest_index + 1]

            if len(available_cut_indexes) == 1:
                choices = ["head"]
            elif len(available_cut_indexes) == 2:
                choices = ["head", "tail"]
            else:
                choices = ["head", "tail", "middle"]
            choice = rng.choice(choices)
            if choice == "head":
                patches = metadata_patches + tunebody_patches[0:]
            else:
                if choice == "tail":
                    cut_index = len(available_cut_indexes) - 1
                else:
                    cut_index = rng.choice(range(1, len(available_cut_indexes) - 1))

                line_index = line_index_for_cut_index[cut_index]
                stream_tunebody_lines = tunebody_lines[line_index:]

                stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode="train")
                if add_special_patches:
                    stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                patches = metadata_patches + stream_tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut or self.tunebody_detect == "startswith_r":
            patches = patches[:patch_length]

        return [[ord(c) for c in patch] + [SPECIAL_ID] * (patch_size - len(patch)) for patch in patches]

    # The RL copy called this ``encode``.
    encode = encode_train

    def encode_generate(self, abc_code, patch_length=None, patch_size=None, add_special_patches=True):
        """Encode a prompt / re-seed context. A trailing partial patch is left unpadded."""
        patch_length = patch_length or self.patch_length
        patch_size = patch_size or self.patch_size

        lines = abc_code.split("\n")
        lines = list(filter(None, lines))

        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith("[V:") or line.startswith("[r:"):
                tunebody_index = i
                break

        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_lines = [line + "\n" for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith("\n"):
                tunebody_lines = [tunebody_lines[i] + "\n" for i in range(len(tunebody_lines) - 1)] + [
                    tunebody_lines[-1]
                ]
            else:
                tunebody_lines = [tunebody_lines[i] + "\n" for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + "\n" for line in tunebody_lines]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode="generate")

        if add_special_patches:
            bos_patch = chr(BOS_ID) * (patch_size - 1) + chr(EOS_ID)
            metadata_patches = [bos_patch] + metadata_patches

        patches = metadata_patches + tunebody_patches
        patches = patches[:patch_length]

        id_patches = []
        for patch in patches:
            if len(patch) < patch_size and patch[-1] != chr(EOS_ID):
                id_patches.append([ord(c) for c in patch])
            else:
                id_patches.append([ord(c) for c in patch] + [SPECIAL_ID] * (patch_size - len(patch)))
        return id_patches

    def encode_prompt(self, prompt_lines) -> list[list[int]]:
        """BOS patch + metadata patches for ``%period`` / ``%composer`` / ``%instrumentation`` lines."""
        patches = self.patchilize_metadata(prompt_lines)
        patches = [[ord(c) for c in p] + [SPECIAL_ID] * (self.patch_size - len(p)) for p in patches]
        return [self.bos_patch] + patches

    def decode(self, patches):
        return "".join(self.patch2chars(patch) for patch in patches)
