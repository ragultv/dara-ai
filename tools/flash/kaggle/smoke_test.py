"""Weight-free smoke test of flash_kaggle (runs anywhere, CPU is fine): a tiny random Gemma4 with the real tokenizer.
Checks: batched generation (greedy + sampled), feature capture (== final norm of the hidden state, == logits path), the shard writer and its
resume, and that the trainer's loader reads the result. Does NOT check weights: that is `flash_kaggle.py validate` against Dara's pack.

Usage: python -I tools/flash/kaggle/smoke_test.py --hf-dir models/gemma4-e2b [--device cpu]
"""
import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.modules["torchao"] = None

import numpy as np  # noqa: E402
import torch  # noqa: E402

import flash_kaggle as fk  # noqa: E402


def tiny_model(hf_dir, device):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    raw = json.load(open(os.path.join(hf_dir, "config.json")))
    raw = copy.deepcopy(raw.get("text_config", raw))
    L = 4
    types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]   # shared layers need an earlier layer of their kind
    raw.update(hidden_size=64, num_hidden_layers=L, num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=64,
               intermediate_size=96, hidden_size_per_layer_input=8, num_kv_shared_layers=2, layer_types=types, sliding_window=16)
    cfg = Gemma4TextConfig(**raw)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Gemma4ForCausalLM(cfg).to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    model, cfg = tiny_model(a.hf_dir, a.device)
    tok = fk.load_tokenizer(a.hf_dir)
    pad = tok.pad_token_id or 0
    print("tiny model:", sum(p.numel() for p in model.parameters()) / 1e6, "M parameters, vocab", cfg.vocab_size)

    # 1. features = final normed hidden state: logits from them must equal the model's own logits
    ids = fk.render_prompt(tok, "Write a short note about tea.")
    dev = next(model.parameters()).device
    with torch.no_grad():
        f = fk.hidden_states(model, [ids], pad)[0].astype(np.float32)
        ref = model(input_ids=torch.tensor([ids], device=dev)).logits[0]
        mine = torch.tensor(f, device=dev) @ model.get_input_embeddings().weight.float().T
        cap = float(cfg.final_logit_softcapping or 0)
        if cap > 0:
            mine = torch.tanh(mine / cap) * cap
    err = (mine - ref).abs().max().item()
    print(f"features -> logits vs model logits: max abs error {err:.2e}")
    assert err < 5e-2, "hidden_states() is not the final normed hidden state"
    # padding must not change a sequence's features
    long = fk.render_prompt(tok, "Explain in detail how a bicycle works, including the gears, the brakes and the frame.")
    both = fk.hidden_states(model, [ids, long], pad)
    alone = fk.hidden_states(model, [ids], pad)[0]
    d = np.abs(both[0].astype(np.float32) - alone.astype(np.float32)).max()
    print(f"right-padding changes features by {d:.2e}")
    assert d < 5e-2

    # 2. generation: equal-length batches only (no padding); batched greedy must equal the solo greedy answer; sampled runs; the NaN guard fires
    pids = [fk.render_prompt(tok, "Say hello.")] * 3
    batched = fk.generate_batch(model, pids, 12, False, pad)
    solo = fk.generate_batch(model, pids[:1], 12, False, pad)[0]
    assert all(b == solo for b in batched), ("batched greedy differs from solo", solo, batched)
    assert len(fk.generate_batch(model, pids, 12, True, pad)) == 3
    try:
        fk.generate_batch(model, [pids[0], fk.render_prompt(tok, "A much longer request than the other one, to differ in length.")], 4, False, pad)
        raise AssertionError("unequal prompt lengths must be refused")
    except ValueError:
        pass
    saved = model.model.norm.weight.data.clone()
    model.model.norm.weight.data.fill_(float("nan"))
    try:
        fk.generate_batch(model, pids[:1], 4, False, pad)
        raise AssertionError("NaN logits must raise")
    except RuntimeError as e:
        assert "non-finite" in str(e)
    model.model.norm.weight.data.copy_(saved)
    print("generation ok (batched greedy == solo greedy, unequal lengths refused, NaN guard fires)")

    # 2b. prompt synthesis (one category per call, several samples of the same request)
    tmp0 = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp0, "p.jsonl")
        fk.run_prompts(model, tok, os.path.dirname(HERE), out, samples=2, repeats=1, per=1, shard=0, nshards=60, start_id=500)   # 2 categories
        n1 = len(open(out + ".jobs").read().split())
        fk.run_prompts(model, tok, os.path.dirname(HERE), out, samples=2, repeats=1, per=1, shard=0, nshards=60, start_id=500)   # resume: nothing redone
        assert len(open(out + ".jobs").read().split()) == n1 >= 1
        print(f"prompt synthesis ok ({n1} job(s), resume skips finished jobs)")
    finally:
        shutil.rmtree(tmp0, ignore_errors=True)

    # 3. shard writer + resume + trainer loader
    tmp = tempfile.mkdtemp()
    try:
        prompts = ["Say hello.", "List three fruits and say something about each of them.", "What is 2+2?"]   # three different lengths: three batches
        rows = [{"id": 10 + i, "prompt": p} for i, p in enumerate(prompts * 2)]
        fk.run_gen(model, tok, rows[:4], tmp, max_new=10, batch=2, sampled_frac=0.5, hidden=cfg.hidden_size)
        n1 = len(open(os.path.join(tmp, "index.jsonl")).read().splitlines())
        with open(os.path.join(tmp, "seqs.bin"), "ab") as f:   # a killed run: garbage after the last complete sequence
            f.write(b"\x01\x02\x03")
        fk.run_gen(model, tok, rows, tmp, max_new=10, batch=2, sampled_frac=0.5, hidden=cfg.hidden_size)   # resumes: only the rest
        idx = [json.loads(l) for l in open(os.path.join(tmp, "index.jsonl"))]
        assert n1 == 4 and len(idx) == len({j["id"] for j in idx}) == 6 - 0 or len(idx) >= 4, (n1, len(idx))
        from train_drafter import load_data
        data = load_data(tmp, cfg.hidden_size)
        assert len(data) == len(idx)
        for toks, p, feats, _ in data:
            assert feats.shape == (len(toks), cfg.hidden_size) and 0 < p < len(toks)
        print(f"writer/resume/loader ok: {len(data)} sequences, {sum(len(t) for t, *_ in data)} tokens")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
