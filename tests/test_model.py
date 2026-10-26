import torch
from conftest import TINY, legacy_overrides
from legacy import load_legacy

from notagen.model import CacheView
from notagen.tokenizer import Patchilizer


def _batch(abc_text, n=2, lengths=(40, 27)):
    tok = Patchilizer(patch_length=TINY.patch_length)
    seqs = [tok.encode_train(abc_text)[:L] for L in lengths[:n]]
    width = max(map(len, seqs))
    patches = torch.zeros(n, width * 16, dtype=torch.long)
    masks = torch.zeros(n, width, dtype=torch.long)
    for i, s in enumerate(seqs):
        patches[i, : len(s) * 16] = torch.tensor(s).flatten()
        masks[i, : len(s)] = 1
    return patches, masks


def test_state_dict_keys_match(tiny_pair):
    _, old, new = tiny_pair
    old_keys = {k for k in old.state_dict() if not k.endswith((".attn.bias", ".attn.masked_bias"))}
    assert old_keys == set(new.state_dict())


def test_training_loss_matches(tiny_pair, abc_text):
    _, old, new = tiny_pair
    patches, masks = _batch(abc_text)
    with torch.no_grad():
        a = old(patches.clone(), masks.clone()).loss
        b = new(patches.clone(), masks.clone())
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_training_grads_match(tiny_pair, abc_text):
    _, old, new = tiny_pair
    patches, masks = _batch(abc_text)
    old.zero_grad(), new.zero_grad()
    old(patches.clone(), masks.clone()).loss.backward()
    new(patches.clone(), masks.clone()).backward()
    og = dict(old.named_parameters())
    for name, p in new.named_parameters():
        torch.testing.assert_close(p.grad, og[name].grad, rtol=1e-4, atol=1e-5, msg=name)
    old.zero_grad(), new.zero_grad()


def test_rl_sequence_logps_match(tiny_pair, abc_text):
    rl = load_legacy("RL", **legacy_overrides(TINY))
    _, old, new = tiny_pair
    from legacy import legacy_model

    old_rl = legacy_model(rl, TINY).eval()
    old_rl.load_state_dict(old.state_dict())
    patches, masks = _batch(abc_text, n=1)
    with torch.no_grad():
        torch.testing.assert_close(
            old_rl(patches.clone(), masks.clone()), new.sequence_logps(patches, masks), rtol=1e-5, atol=1e-4
        )


def test_cached_patch_decoder_matches_full(tiny_pair, abc_text):
    _, old, new = tiny_pair
    ids = torch.tensor(Patchilizer(patch_length=TINY.patch_length).encode_train(abc_text)[:30])[None]
    with torch.no_grad():
        full_old = old.patch_level_decoder(ids.reshape(1, -1))["last_hidden_state"]
        full_new = new.patch_level_decoder(ids)
        torch.testing.assert_close(full_old, full_new, rtol=1e-4, atol=1e-4)

        # prefill 10 in row 1 of a 2-row cache, then step one patch at a time
        cache = new.new_patch_cache(2)
        out = new.patch_level_decoder(ids[:, :10], view=CacheView(cache, torch.tensor([0]), 10, rows=torch.tensor([1])))
        steps = [out]
        for t in range(10, 30):
            x = torch.zeros(2, 1, 16, dtype=torch.long)
            x[1, 0] = ids[0, t]
            o = new.patch_level_decoder(x, view=CacheView(cache, torch.tensor([0, t]), t + 1))
            steps.append(o[1:2])
        torch.testing.assert_close(torch.cat(steps, 1), full_new, rtol=1e-4, atol=1e-4)


def test_char_step_matches_legacy_generate_probs(tiny_pair):
    _, old, new = tiny_pair
    torch.manual_seed(1)
    enc = torch.randn(TINY.hidden_size)
    tokens = torch.tensor([1, 91, 86, 58, 49, 93])  # BOS + "[V:1]"
    with torch.no_grad():
        p_old = old.char_level_decoder.generate(enc, tokens)
        wte = new.char_level_decoder.base.transformer.wte
        x = torch.cat([enc.view(1, 1, -1), wte(tokens[1:])[None]], 1)
        cache = new.new_char_cache(1)
        logits = new.char_level_decoder.step(x, CacheView(cache, torch.tensor([0]), x.shape[1]))
        torch.testing.assert_close(p_old, logits[0, -1].softmax(-1), rtol=1e-5, atol=1e-6)


def test_safetensors_roundtrip(tmp_path, tiny_pair):
    from notagen.checkpoint import load_model, save_model

    _, _, new = tiny_pair
    save_model(new, tmp_path / "m")
    again = load_model(tmp_path / "m")
    assert again.cfg == new.cfg
    for (k, a), (_, b) in zip(new.state_dict().items(), again.state_dict().items()):
        assert torch.equal(a, b), k
    assert again.char_level_decoder.base.lm_head.weight is again.char_level_decoder.base.transformer.wte.weight


def test_legacy_pth_roundtrip(tmp_path, tiny_pair):
    from notagen.checkpoint import load_model

    _, old, new = tiny_pair
    name = "weights_notagen_t_p_size_16_p_length_48_p_layers_2_c_layers_2_h_size_128_lr_1e-05_batch_1.pth"
    torch.save({"model": old.state_dict(), "epoch": 3, "min_eval_loss": 1.5}, tmp_path / name)
    m = load_model(tmp_path / name)
    assert m.cfg.patch_length == 48 and m.cfg.hidden_size == 128
    for k, v in new.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
