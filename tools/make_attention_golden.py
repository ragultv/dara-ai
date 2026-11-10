"""Golden data for a full-attention layer: the real transformers Qwen3_5Attention with layer-3 weights from the pinned
checkpoint, on random input at positions 0..T-1 with a causal mask. Intermediates follow the reference's order:
  q_proj -> split [query | gate] per head -> q_norm / k_norm -> RoPE -> GQA attention (softmax fp32) -> * sigmoid(gate)
  -> o_proj.
The script replicates the module's forward step by step (so the intermediates exist) and asserts that it equals the
module's own output.
Output: build/golden/attention_layer3.bin
Usage : python -I tools/make_attention_golden.py [--tokens 70]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

sys.modules["torchao"] = None  # incompatible with torch 2.5.1; not needed here
sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402

REPO = "Qwen/Qwen3.5-4B"
REV = json.load(open("manifests/qwen3.5-4b.json"))["source"]["revision"]
P = "model.language_model.layers.3.self_attn."


def hf_get(path, lo, hi):
    return gp.get(f"https://huggingface.co/{REPO}/resolve/{REV}/{path}", lo, hi)


def fetch(index, hdrs, name):
    shard = index["weight_map"][name]
    if shard not in hdrs:
        n = struct.unpack("<Q", hf_get(shard, 0, 7))[0]
        hdrs[shard] = (n, json.loads(hf_get(shard, 8, 8 + n - 1)))
    n, h = hdrs[shard]
    t = h[name]
    lo, hi = t["data_offsets"]
    raw = hf_get(shard, 8 + n + lo, 8 + n + hi - 1)
    assert t["dtype"] == "BF16", t
    return np.frombuffer(raw, dtype="<u2").reshape(t["shape"]).copy()


def write(path, tensors):
    with open(path, "wb") as f:
        f.write(b"DGLD1\0\0\0")
        f.write(struct.pack("<I", len(tensors)))
        for name, (arr, code) in tensors.items():
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<II", code, arr.ndim))
            f.write(struct.pack(f"<{arr.ndim}Q", *arr.shape))
            f.write(np.ascontiguousarray(arr).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=70)
    a = ap.parse_args()

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5Attention, Qwen3_5TextRotaryEmbedding,
                                                              apply_rotary_pos_emb, eager_attention_forward)

    cfg = Qwen3_5TextConfig(**json.loads(hf_get("config.json", 0, 1 << 20))["text_config"])
    mod = Qwen3_5Attention(cfg, 3).float().eval()
    index = json.loads(hf_get("model.safetensors.index.json", 0, 10 << 20))
    hdrs = {}
    names = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"]
    raw = {n: fetch(index, hdrs, P + n) for n in names}
    for n in names:
        print(f"  fetched {n} {raw[n].shape}")
    with torch.no_grad():
        for n in names:
            val = torch.from_numpy((raw[n].astype(np.uint32) << 16).view(np.float32).copy())
            p = mod.get_parameter(n)
            assert tuple(p.shape) == tuple(val.shape), (n, p.shape, val.shape)
            p.copy_(val)

    T = a.tokens
    g = torch.Generator().manual_seed(99)
    x = torch.randn(1, T, cfg.hidden_size, generator=g, dtype=torch.float32)
    pos = torch.arange(T)[None]
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    with torch.no_grad():
        cos, sin = rot(x, pos)
        mask = torch.full((T, T), float("-inf")).triu(1)[None, None]  # additive causal mask
        ref_out, _ = mod(x, position_embeddings=(cos, sin), attention_mask=mask)

        # ---- step-by-step replication of Qwen3_5Attention.forward ------------------------------
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, mod.head_dim)
        q_raw = mod.q_proj(x)                                           # [1, T, 16*512]
        query_states, gate = torch.chunk(q_raw.view(*input_shape, -1, mod.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)                           # [1, T, 4096]
        q_normed = mod.q_norm(query_states.view(hidden_shape))          # [1, T, 16, 256]
        k_normed = mod.k_norm(mod.k_proj(x).view(hidden_shape))         # [1, T, 4, 256]
        v = mod.v_proj(x).view(hidden_shape)                            # [1, T, 4, 256]
        q_rot, k_rot = apply_rotary_pos_emb(q_normed.transpose(1, 2), k_normed.transpose(1, 2), cos, sin)
        attn, _ = eager_attention_forward(mod, q_rot, k_rot, v.transpose(1, 2), mask, scaling=mod.scaling, dropout=0.0)
        attn = attn.reshape(*input_shape, -1).contiguous()              # [1, T, 4096], before the gate
        gated = attn * torch.sigmoid(gate)
        mine = mod.o_proj(gated)
    err = (mine - ref_out).abs().max().item() / ref_out.abs().max().item()
    print(f"replication vs module forward: max rel diff {err:.2e}")
    assert err < 1e-4, "step-by-step replication does not match the module"

    def f32(t, perm=None):
        t = t.squeeze(0)
        if perm is not None:
            t = t.permute(*perm)
        return np.ascontiguousarray(t.float().numpy()), 0

    tensors = {
        "w.q_proj": (raw["q_proj.weight"], 1), "w.k_proj": (raw["k_proj.weight"], 1), "w.v_proj": (raw["v_proj.weight"], 1),
        "w.o_proj": (raw["o_proj.weight"], 1), "w.q_norm": (raw["q_norm.weight"], 1), "w.k_norm": (raw["k_norm.weight"], 1),
        "x": f32(x), "q_raw": f32(q_raw), "gate": f32(gate), "q_normed": f32(q_normed), "k_normed": f32(k_normed),
        "v": f32(v),
        "q_rot": f32(q_rot, (1, 0, 2)), "k_rot": f32(k_rot, (1, 0, 2)),   # [T, heads, 256]
        "attn": f32(attn), "gated": f32(gated), "out": f32(ref_out),
        "cos": f32(cos), "sin": f32(sin),
    }
    os.makedirs("build/golden", exist_ok=True)
    write("build/golden/attention_layer3.bin", tensors)
    print(f"wrote build/golden/attention_layer3.bin: T={T}, out |max|={ref_out.abs().max():.3f}")


if __name__ == "__main__":
    main()
