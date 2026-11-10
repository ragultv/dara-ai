"""Golden data for RMSNorm and RoPE from the real transformers modules and the pinned checkpoint's weights.

RMSNorm: Qwen3_5RMSNorm with layer-0 input_layernorm (hidden 2560) and layer-3 q_norm (head_dim 256) weights, on
         random, large, tiny (eps-dominated), zero, constant, mixed-magnitude and alternating-sign inputs. Also the
         intermediate rsqrt statistic.
RoPE   : Qwen3_5TextRotaryEmbedding + apply_rotary_pos_emb at positions from 0 up to the native 262143, with the
         inv_freq table, cos/sin, and rotated q (16 heads) / k (4 heads) of head_dim 256.
Output: build/golden/ops.bin (container as in make_deltanet_golden.py)
Usage : python -I tools/make_ops_golden.py
"""
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
    return np.frombuffer(raw, dtype="<u2").copy()


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
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding,
                                                              apply_rotary_pos_emb)

    cfg_json = json.loads(hf_get("config.json", 0, 1 << 20))
    cfg = Qwen3_5TextConfig(**cfg_json["text_config"])
    index = json.loads(hf_get("model.safetensors.index.json", 0, 10 << 20))
    hdrs = {}
    out = {}
    g = torch.Generator().manual_seed(7)

    # ---- RMSNorm -----------------------------------------------------------------------------
    for tag, wname, dim in (("hidden", "model.language_model.layers.0.input_layernorm.weight", cfg.hidden_size),
                            ("head", "model.language_model.layers.3.self_attn.q_norm.weight", cfg.head_dim)):
        raw = fetch(index, hdrs, wname)
        w = torch.from_numpy((raw.astype(np.uint32) << 16).view(np.float32).copy())
        norm = Qwen3_5RMSNorm(dim, eps=cfg.rms_norm_eps).float()
        with torch.no_grad():
            norm.weight.copy_(w)
        cases = [
            torch.randn(dim, generator=g),                         # ordinary
            torch.randn(dim, generator=g) * 1e4,                   # large
            torch.randn(dim, generator=g) * 1e-6,                  # tiny: variance ~1e-12 << eps, eps dominates
            torch.zeros(dim),                                      # zero
            torch.full((dim,), 3.0),                               # constant
            torch.cat([torch.tensor([1e4]), torch.randn(dim - 1, generator=g) * 1e-3]),  # one huge among tiny
            torch.where(torch.arange(dim) % 2 == 0, 1.0, -1.0) * torch.rand(dim, generator=g),  # alternating sign
            torch.randn(dim, generator=g) * torch.exp(torch.randn(dim, generator=g) * 4),  # wide dynamic range
        ]
        x = torch.stack(cases)
        with torch.no_grad():
            y = norm(x)
        inv_rms = torch.rsqrt(x.pow(2).mean(-1) + cfg.rms_norm_eps)
        out[f"norm_{tag}.w"] = (raw, 1)  # checkpoint weight (BF16 raw); the (1+w) form lives in the module
        out[f"norm_{tag}.x"] = (x.numpy(), 0)
        out[f"norm_{tag}.y"] = (y.numpy(), 0)
        out[f"norm_{tag}.inv_rms"] = (inv_rms.numpy(), 0)

    # ---- RoPE --------------------------------------------------------------------------------
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    positions = [0, 1, 2, 3, 63, 64, 255, 1000, 4096, 32767, 100000, 131071, 131072, 200000, 262143]
    T = len(positions)
    q = torch.randn(1, cfg.num_attention_heads, T, cfg.head_dim, generator=g)
    k = torch.randn(1, cfg.num_key_value_heads, T, cfg.head_dim, generator=g)
    pos = torch.tensor([positions])
    with torch.no_grad():
        cos, sin = rot(q, pos)  # [1, T, rotary_dim]
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
    out["rope.positions"] = (np.array(positions, dtype=np.float32), 0)  # exact: all positions < 2^24
    out["rope.inv_freq"] = (rot.inv_freq.numpy().astype(np.float32), 0)
    out["rope.cos"] = (cos[0].numpy(), 0)   # [T, 64] (duplicated halves)
    out["rope.sin"] = (sin[0].numpy(), 0)
    out["rope.q"] = (q[0].permute(1, 0, 2).contiguous().numpy(), 0)    # [T, 16, 256]
    out["rope.k"] = (k[0].permute(1, 0, 2).contiguous().numpy(), 0)    # [T, 4, 256]
    out["rope.q_rot"] = (qr[0].permute(1, 0, 2).contiguous().numpy(), 0)
    out["rope.k_rot"] = (kr[0].permute(1, 0, 2).contiguous().numpy(), 0)
    os.makedirs("build/golden", exist_ok=True)
    write("build/golden/ops.bin", out)
    print(f"wrote build/golden/ops.bin: rotary_dim={cos.shape[-1]}, positions up to {max(positions)}, "
          f"inv_freq[0..2]={rot.inv_freq[:3].tolist()}")


if __name__ == "__main__":
    main()
