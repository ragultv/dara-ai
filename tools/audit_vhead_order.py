"""Audit how the GGUF stores every DeltaNet tensor relative to the checkpoint (layer 0 and layer 3 norms).

For each tensor that can carry a per-head axis, find the actual head mapping GGUF-head j <- checkpoint-head i
by correlating real rows (dequantized with the independent gguf-py), instead of assuming one. Also detects value
transforms: A_log -> -exp(A_log), and RMSNorm weight w -> w + 1.

Usage: python -I tools/audit_vhead_order.py
"""
import json
import os
import struct
import sys
import urllib.request

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402

GG_REPO, GG_REV, GG_FILE = "unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "Qwen3.5-4B-Q4_K_M.gguf"
HF_REPO = "Qwen/Qwen3.5-4B"
HF_REV = json.load(open("manifests/qwen3.5-4b.json"))["source"]["revision"]
QT = {"F32": None, "Q4_K": T.Q4_K, "Q5_K": T.Q5_K, "Q6_K": T.Q6_K, "Q8_0": T.Q8_0}
GG_URL = f"https://huggingface.co/{GG_REPO}/resolve/{GG_REV}/{GG_FILE}"
L = "model.language_model.layers."


def hf_get(path, lo, hi):
    return gp.get(f"https://huggingface.co/{HF_REPO}/resolve/{HF_REV}/{path}", lo, hi)


class Hf:
    def __init__(self):
        self.index = json.loads(hf_get("model.safetensors.index.json", 0, 10 << 20))
        self.hdr = {}

    def tensor_all(self, name):
        """Whole (small) tensor in one request."""
        shard = self.index["weight_map"][name]
        if shard not in self.hdr:
            n = struct.unpack("<Q", hf_get(shard, 0, 7))[0]
            self.hdr[shard] = (n, json.loads(hf_get(shard, 8, 8 + n - 1)))
        n, h = self.hdr[shard]
        t = h[name]
        lo, hi = t["data_offsets"]
        raw = hf_get(shard, 8 + n + lo, 8 + n + hi - 1)
        if t["dtype"] == "F32":
            return np.frombuffer(raw, dtype="<f4").astype(np.float64)
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)

    def tensor_rows(self, name, rows):
        shard = self.index["weight_map"][name]
        if shard not in self.hdr:
            n = struct.unpack("<Q", hf_get(shard, 0, 7))[0]
            self.hdr[shard] = (n, json.loads(hf_get(shard, 8, 8 + n - 1)))
        n, h = self.hdr[shard]
        t = h[name]
        shp = t["shape"]
        cols = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        esz = 4 if t["dtype"] == "F32" else 2
        out = []
        for r in rows:
            lo = 8 + n + t["data_offsets"][0] + r * cols * esz
            raw = hf_get(shard, lo, lo + cols * esz - 1)
            if t["dtype"] == "F32":
                out.append(np.frombuffer(raw, dtype="<f4").astype(np.float64))
            else:
                out.append((np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64))
        return np.array(out), shp


def load_gguf_header():
    size = 8 << 20
    while True:
        data = gp.get(GG_URL, 0, size - 1)
        try:
            _, _, tensors, data_off = gp.parse(data)
            return {t["name"]: t for t in tensors}, data_off
        except EOFError:
            size *= 2


def gg_rows(tensors, data_off, name, rows):
    t = tensors[name]
    ty = t["type"]
    cols = t["dims"][0]
    if ty == "F32":
        elems = int(np.prod(t["dims"]))
        raw = gp.get(GG_URL, data_off + t["offset"], data_off + t["offset"] + elems * 4 - 1)
        a = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        return a.reshape(-1, cols) if len(t["dims"]) > 1 else a.reshape(1, -1)
    be, bb = gp.BLOCK[ty]
    rb = cols // be * bb
    out = []
    for r in rows:
        lo = data_off + t["offset"] + r * rb
        blk = np.frombuffer(gp.get(GG_URL, lo, lo + rb - 1), dtype=np.uint8).reshape(1, rb).copy()
        out.append(dequantize(blk, QT[ty]).astype(np.float64).reshape(-1))
    return np.array(out)


def match(G, H):
    """For each GGUF probe row, best-correlated checkpoint probe row."""
    gm = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-30)
    hm = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-30)
    c = gm @ hm.T / G.shape[1]
    return c.argmax(1), c.max(1)


def hf_head_expected(j, n):
    return 2 * j if j < n // 2 else 2 * (j - n // 2) + 1


def report(label, mapping, corr, n):
    exp = [hf_head_expected(j, n) for j in range(n)]
    ident = list(range(n))
    kind = "GGUF-deinterleaved (16r+k)" if list(mapping) == exp else "identity" if list(mapping) == ident else "OTHER"
    print(f"  {label:34s} heads={n:2d} min corr {corr.min():.3f}  mapping: {kind}")
    if kind == "OTHER":
        print("     ", list(map(int, mapping)))
    return kind


def main():
    hf = Hf()
    tensors, data_off = load_gguf_header()
    print(f"GGUF {GG_FILE} @ {GG_REV[:10]}, HF {HF_REPO} @ {HF_REV[:10]}")

    # --- row-axis tensors: probe the first row of every 128-row head (and a second row to check intra-head order)
    print("\nPer-head row axes (layer 0):")
    sections = [  # (label, gguf name, hf name, row offset of the section, heads)
        ("attn_qkv  q section (16 heads)", "blk.0.attn_qkv.weight", L + "0.linear_attn.in_proj_qkv.weight", 0, 16),
        ("attn_qkv  k section (16 heads)", "blk.0.attn_qkv.weight", L + "0.linear_attn.in_proj_qkv.weight", 2048, 16),
        ("attn_qkv  v section (32 heads)", "blk.0.attn_qkv.weight", L + "0.linear_attn.in_proj_qkv.weight", 4096, 32),
        ("attn_gate (z, 32 heads)", "blk.0.attn_gate.weight", L + "0.linear_attn.in_proj_z.weight", 0, 32),
    ]
    results = {}
    for label, gname, hname, base, n in sections:
        for within in (0, 7):
            rows = [base + 128 * i + within for i in range(n)]
            G = gg_rows(tensors, data_off, gname, rows)
            H, _ = hf.tensor_rows(hname, rows)
            m, c = match(G, H)
            k = report(f"{label} row+{within}", m, c, n)
        results[label] = k

    # --- small row tensors: one row per head
    for label, gname, hname in [("ssm_beta  (in_proj_b)", "blk.0.ssm_beta.weight", L + "0.linear_attn.in_proj_b.weight"),
                                ("ssm_alpha (in_proj_a)", "blk.0.ssm_alpha.weight", L + "0.linear_attn.in_proj_a.weight")]:
        rows = list(range(32))
        G = gg_rows(tensors, data_off, gname, rows)
        H, _ = hf.tensor_rows(hname, rows)
        m, c = match(G, H)
        results[label] = report(label, m, c, 32)

    # --- vectors with a head axis
    print("\nPer-head vectors and value transforms (layer 0):")
    A_log = hf.tensor_all(L + "0.linear_attn.A_log")
    ssm_a = gg_rows(tensors, data_off, "blk.0.ssm_a", [0])[0]
    for name, want in [("ssm_a == -exp(A_log)[head map]", -np.exp(A_log)), ("ssm_a == A_log[head map]", A_log)]:
        perm = [hf_head_expected(j, 32) for j in range(32)]
        for pname, p in (("deinterleaved", perm), ("identity", list(range(32)))):
            err = np.abs(ssm_a - want[p]).max()
            if err < 1e-5 * (np.abs(want).max() + 1):
                print(f"  {name:34s} matches under {pname} order (max err {err:.2e})")
    dt = gg_rows(tensors, data_off, "blk.0.ssm_dt.bias", [0])[0]
    dt_hf = hf.tensor_all(L + "0.linear_attn.dt_bias")
    for pname, p in (("deinterleaved", [hf_head_expected(j, 32) for j in range(32)]), ("identity", list(range(32)))):
        if np.abs(dt - dt_hf[p]).max() < 1e-5 * (np.abs(dt_hf).max() + 1):
            print(f"  {'ssm_dt.bias == dt_bias':34s} matches under {pname} order")

    # --- conv1d: GGUF [4, 8192] row-major = [channel][tap]
    conv = gg_rows(tensors, data_off, "blk.0.ssm_conv1d.weight", [0])
    conv = conv.reshape(8192, 4) if conv.size == 8192 * 4 else conv
    conv_hf = hf.tensor_all(L + "0.linear_attn.conv1d.weight").reshape(8192, 4)
    for label, lo, n in (("conv1d q channels", 0, 16), ("conv1d k channels", 2048, 16), ("conv1d v channels", 4096, 32)):
        G = conv[[lo + 128 * i for i in range(n)]]
        H = conv_hf[[lo + 128 * i for i in range(n)]]
        d = np.abs(G[:, None, :] - H[None, :, :]).sum(2)
        results[label] = report(label, d.argmin(1), 1 - d.min(1) / (np.abs(H).sum(1).mean() + 1e-30), n)
    same = np.abs(conv - conv_hf).max() < 1e-6
    print(f"  conv1d identical element-for-element (no permutation at all): {same}")

    # --- norms
    print("\nNorm weights:")
    for gname, hname, tag in [("blk.0.attn_norm.weight", L + "0.input_layernorm.weight", "input_layernorm"),
                              ("blk.0.post_attention_norm.weight", L + "0.post_attention_layernorm.weight", "post_attention_layernorm"),
                              ("blk.3.attn_q_norm.weight", L + "3.self_attn.q_norm.weight", "q_norm"),
                              ("blk.3.attn_k_norm.weight", L + "3.self_attn.k_norm.weight", "k_norm"),
                              ("blk.0.ssm_norm.weight", L + "0.linear_attn.norm.weight", "gated norm (linear_attn.norm)")]:
        g = gg_rows(tensors, data_off, gname, [0])[0]
        h = hf.tensor_all(hname)
        ident = np.abs(g - h).max()
        plus1 = np.abs(g - (h + 1.0)).max()
        verdict = "identical" if ident < 1e-6 else "GGUF = HF + 1 (the +1 is folded in)" if plus1 < 1e-6 else f"differs (id {ident:.3g}, +1 {plus1:.3g})"
        print(f"  {tag:30s} {verdict}")

    json.dump(results, open("manifests/vhead_audit.json", "w"), indent=1)
    print("\nwrote manifests/vhead_audit.json")


if __name__ == "__main__":
    main()
