import random

import pytest
from conftest import TINY, legacy_overrides
from legacy import load_legacy

from notagen.tokenizer import Patchilizer

CASES = [("pretrain", "startswith_v"), ("finetune", "contains_v"), ("inference", "contains_v")]


@pytest.mark.parametrize("stage,detect", CASES)
@pytest.mark.parametrize("patch_length", [1024, 40])
def test_encode_train_matches_legacy(abc_text, stage, detect, patch_length):
    mod = load_legacy(stage, **{**legacy_overrides(TINY), "PATCH_LENGTH": patch_length})
    old = mod.Patchilizer()
    new = Patchilizer(patch_length=patch_length, tunebody_detect=detect)
    for seed in range(8):  # stream mode picks head/tail/middle cuts at random
        random.seed(seed)
        a = old.encode_train(abc_text)
        random.seed(seed)
        b = new.encode_train(abc_text)
        assert a == b


def test_rl_encode_matches_legacy(abc_text):
    mod = load_legacy("RL", **{**legacy_overrides(TINY), "PATCH_LENGTH": 40})
    tagged = Patchilizer(patch_length=10_000).decode(Patchilizer(patch_length=10_000).encode_train(abc_text))
    # The RL copy expects text whose tune-body lines already carry [r:i/n] tags.
    tagged = "".join(c for c in tagged if ord(c) > 2)
    old, new = mod.Patchilizer(), Patchilizer(patch_length=40, tunebody_detect="startswith_r")
    for seed in range(8):
        random.seed(seed)
        a = old.encode(tagged)
        random.seed(seed)
        assert a == new.encode(tagged)


@pytest.mark.parametrize("cut", [None, 5, 17, 33])
def test_encode_generate_and_decode_match_legacy(abc_text, cut):
    mod = load_legacy("inference", **legacy_overrides(TINY))
    old, new = mod.Patchilizer(), Patchilizer(patch_length=TINY.patch_length)
    tagged = new.decode(Patchilizer(patch_length=10_000).encode_train(abc_text)[:-1]).replace(chr(1), "")
    text = tagged if cut is None else tagged[: len(tagged) - cut]
    assert old.encode_generate(text) == new.encode_generate(text)
    ids = new.encode_train(abc_text)
    assert old.decode(ids) == new.decode(ids)


def test_prompt_encoding(abc_text):
    new = Patchilizer()
    lines = ["%Classical\n", "%Mozart, Wolfgang Amadeus\n", "%Keyboard\n"]
    ids = new.encode_prompt(lines)
    assert ids[0] == new.bos_patch and all(len(p) == 16 for p in ids)
    assert new.decode(ids[1:]) == "".join(lines)
