"""Quantization error of real GGUF tensors against the original BF16 checkpoint (same tensor, same rows).

For each case, fetches the tensor's rows from the pinned BF16 safetensors by HTTP range request and compares
them with the dequantized GGUF rows in tests/fixtures (decoded by the independent gguf-py implementation).

FINDING: the GGUF converter de-interleaves the DeltaNet value heads. The checkpoint orders the 32 V heads as
(key_group k, r) -> 2k + r; GGUF orders them (r, key_group k) -> 16r + k. So GGUF V-head j is checkpoint
head hf_head(j) = 2j for j < 16 and 2(j-16)+1 for j >= 16. Tensors with a per-V-head axis (here out_proj
columns in blocks of 128 and in_proj_a rows) must be un-permuted before comparing with the checkpoint, and
any reference (HF golden tensors) used against a GGUF-based backend needs the same mapping.
"""
import json
import struct
import sys
import urllib.request

import numpy as np

REPO, REV = "Qwen/Qwen3.5-4B", json.load(open("manifests/qwen3.5-4b.json"))["source"]["revision"]
V_HEADS = 32


def hf_head(j):
    return 2 * j if j < V_HEADS // 2 else 2 * (j - V_HEADS // 2) + 1


# (label, fixture, block bytes, HF tensor, gguf rows, cols, v-head permutation: None | "col-blocks-128" | "rows")
CASES = [
    ("Q4_K blk.3.ffn_gate", "q4k_real.bin", 144, "model.language_model.layers.3.mlp.gate_proj.weight", 4, 2560, None),
    ("Q6_K blk.3.ffn_down", "q6k_down.bin", 210, "model.language_model.layers.3.mlp.down_proj.weight", 2, 9216, None),
    ("Q6_K token_embd (output head)", "q6k_embd.bin", 210, "model.language_model.embed_tokens.weight", 4, 2560, None),
    ("Q5_K blk.0.attn_qkv", "q5k_qkv.bin", 176, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 4, 2560, None),
    ("Q5_K blk.0.ssm_out", "q5k_ssmout.bin", 176, "model.language_model.layers.0.linear_attn.out_proj.weight", 2, 4096, "col-blocks-128"),
    ("Q8_0 blk.0.ssm_alpha", "q80_alpha.bin", 34, "model.language_model.layers.0.linear_attn.in_proj_a.weight", 8, 2560, "rows"),
]


def get(path, lo, hi):
    req = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REV}/{path}",
                                 headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    index = json.loads(get("model.safetensors.index.json", 0, 10 << 20))
    rng = np.random.default_rng(1)
    for label, fixture, bsz, tensor, rows, cols, perm in CASES:
        shard = index["weight_map"][tensor]
        n = struct.unpack("<Q", get(shard, 0, 7))[0]
        t = json.loads(get(shard, 8, 8 + n - 1))[tensor]
        assert t["dtype"] == "BF16" and t["shape"][1] == cols, t
        fetch_rows = t["shape"][0] if perm == "rows" else rows  # a row permutation needs the whole tensor
        lo = 8 + n + t["data_offsets"][0]
        raw = np.frombuffer(get(shard, lo, lo + fetch_rows * cols * 2 - 1), dtype="<u2")
        bf16 = (raw.astype(np.uint32) << 16).view(np.float32).reshape(fetch_rows, cols).astype(np.float64)
        if perm == "rows":
            bf16 = bf16[[hf_head(j) for j in range(rows)]]
        elif perm == "col-blocks-128":
            bf16 = np.concatenate([bf16[:, hf_head(j) * 128:(hf_head(j) + 1) * 128] for j in range(cols // 128)], axis=1)

        f = open(f"tests/fixtures/{fixture}", "rb").read()
        nb = struct.unpack_from("<I", f, 0)[0]
        exp = np.frombuffer(f, dtype="<f4", offset=4 + nb * bsz).astype(np.float64).reshape(rows, cols)

        err = exp - bf16
        rmse = np.sqrt((err ** 2).sum() / (bf16 ** 2).sum())
        rel = []
        for _ in range(200):
            x = rng.standard_normal(cols)
            rel.append(np.abs(exp @ x - bf16 @ x) / (np.abs(bf16) @ np.abs(x)))
        rel = np.array(rel)
        corr = np.corrcoef(bf16.ravel(), exp.ravel())[0, 1]
        note = f"  [V-head un-permute: {perm}]" if perm else ""
        print(f"{label:30s} weights rel RMSE {rmse:.4f} | GEMV err/sum|w||x| median {np.median(rel):.4f} "
              f"max {rel.max():.4f} | corr {corr:.5f}{note}")
        assert corr > 0.99, f"{label}: tensors do not line up"


if __name__ == "__main__":
    sys.exit(main())
