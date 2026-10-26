import numpy as np

from notagen.abc import rest_unreduce, to_lines
from notagen.metrics import alignment_ok, mean_pairwise_distance, near_copy_rate, staves_grouped


def test_alignment(abc_text):
    lines = to_lines(abc_text)
    assert alignment_ok(lines)
    bad = [l.replace("[V:2]C,8|", "[V:2]C,6|") for l in lines]
    assert not alignment_ok(bad)


def test_rest_unreduce_roundtrip(abc_text):
    # Drop voice 2 from one bar (as training data omits whole-rest bars) and restore it.
    lines = to_lines(abc_text)
    i = next(i for i, l in enumerate(lines) if l.startswith("[V:1]"))
    tagged = [l if not l.startswith("[V:") else f"[r:{k}/0]" + l for k, l in enumerate(lines)]
    tagged[i] = tagged[i].split("[V:2]")[0] + "\n"
    restored = rest_unreduce(tagged)
    assert "[V:2]" in restored[i] and alignment_ok(restored)


def test_staves_grouped():
    ok = ['V:1 treble nm="Piano"\n', 'V:2 bass nm="Piano"\n', 'V:3 nm="Violin"\n']
    bad = ['V:1 treble nm="Piano"\n', 'V:2 nm="Violin"\n', 'V:3 bass nm="Piano"\n']
    assert staves_grouped(ok) and not staves_grouped(bad)


def test_diversity_and_copy_rate():
    rng = np.random.default_rng(0)
    same = np.repeat(rng.normal(size=(1, 8)), 4, 0)
    assert abs(mean_pairwise_distance(same)) < 1e-9
    assert mean_pairwise_distance(rng.normal(size=(6, 8))) > 0.5
    assert near_copy_rate(same[:1], same) == 1.0
