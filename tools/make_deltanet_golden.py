"""Golden data for the DeltaNet layer: the real transformers Qwen3_5GatedDeltaNet with layer-0 weights from the
pinned checkpoint, run on random input. Records intermediates so the C++ implementation can be checked stage by
stage (not only the final output).

Output: build/golden/deltanet_layer0.bin  (container format below; ~100 MB because weights are kept as BF16)
Container: "DGLD1\\0\\0\\0", u32 count, then per tensor: u32 name_len, name, u32 dtype (0=f32, 1=bf16 raw u16),
           u32 ndim, u64 dims[ndim], raw data.
Usage: python -I tools/make_deltanet_golden.py [--tokens 70]
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

# The installed torchao is incompatible with torch 2.5.1 (torch.int1 missing) and transformers imports it
# eagerly if present. This script needs no quantization support, so hide it.
sys.modules["torchao"] = None

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402  (retrying range-request helper)

REPO = "Qwen/Qwen3.5-4B"
REV = json.load(open("manifests/qwen3.5-4b.json"))["source"]["revision"]
L = "model.language_model.layers.0.linear_attn."


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
    if t["dtype"] == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(t["shape"]).copy(), 0
    assert t["dtype"] == "BF16", t
    return np.frombuffer(raw, dtype="<u2").reshape(t["shape"]).copy(), 1


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


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
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg_json = json.loads(hf_get("config.json", 0, 1 << 20))
    cfg = Qwen3_5TextConfig(**cfg_json["text_config"])
    mod = Qwen3_5GatedDeltaNet(cfg, 0).float().eval()

    index = json.loads(hf_get("model.safetensors.index.json", 0, 10 << 20))
    hdrs = {}
    names = ["in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight", "conv1d.weight",
             "dt_bias", "A_log", "norm.weight", "out_proj.weight"]
    raw = {n: fetch(index, hdrs, L + n) for n in names}
    for n, (arr, code) in raw.items():
        print(f"  fetched {n} {arr.shape}")
    with torch.no_grad():
        for n, (arr, code) in raw.items():
            val = torch.from_numpy(bf16_to_f32(arr) if code == 1 else arr).float()
            p = mod.get_parameter(n)
            assert tuple(p.shape) == tuple(val.shape), (n, p.shape, val.shape)
            p.copy_(val)

    # Capture the arguments and results of the delta-rule call.
    cap = {}
    orig = mod.chunk_gated_delta_rule

    def wrapped(query, key, value, g, beta, **kw):
        kw["output_final_state"] = True
        cap["q"], cap["k"], cap["v"], cap["g"], cap["beta"] = (t.detach().clone() for t in (query, key, value, g, beta))
        out, state = orig(query, key, value, g=g, beta=beta, **kw)
        cap["core"], cap["state"] = out.detach().clone(), state.detach().clone()
        return out, state

    mod.chunk_gated_delta_rule = wrapped
    mod.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("normed", o.detach().clone()))

    g = torch.Generator().manual_seed(1234)
    T = a.tokens
    x = torch.randn(1, T, cfg.hidden_size, generator=g, dtype=torch.float32)
    with torch.no_grad():
        out = mod(x)
        qkv_pre = mod.in_proj_qkv(x)  # pre-conv inputs: the conv state after T tokens is its last 3 steps

    def f32(t):
        return np.ascontiguousarray(t.squeeze(0).float().numpy()), 0

    tensors = {
        # weights as stored (BF16 raw), plus the two F32 ones
        "w.in_qkv": raw["in_proj_qkv.weight"], "w.in_z": raw["in_proj_z.weight"], "w.in_b": raw["in_proj_b.weight"],
        "w.in_a": raw["in_proj_a.weight"], "w.conv": raw["conv1d.weight"], "w.dt_bias": raw["dt_bias"],
        "w.A_log": raw["A_log"], "w.norm": raw["norm.weight"], "w.out": raw["out_proj.weight"],
        # input, intermediates, output (all [T, ...])
        "x": f32(x), "q_conv": f32(cap["q"]), "k_conv": f32(cap["k"]), "v_conv": f32(cap["v"]),
        "g": f32(cap["g"]), "beta": f32(cap["beta"]), "core": f32(cap["core"]),
        "normed": (np.ascontiguousarray(cap["normed"].reshape(T, -1).numpy()), 0),
        "out": f32(out), "final_state": (cap["state"].squeeze(0).numpy().copy(), 0),
        "conv_state": (np.ascontiguousarray(qkv_pre.squeeze(0)[-(cfg.linear_conv_kernel_dim - 1):].T.numpy()), 0),
    }
    os.makedirs("build/golden", exist_ok=True)
    write("build/golden/deltanet_layer0.bin", tensors)
    print(f"wrote build/golden/deltanet_layer0.bin: T={T}, out |max|={out.abs().max():.3f}, "
          f"state |max|={cap['state'].abs().max():.3f}")


if __name__ == "__main__":
    main()
