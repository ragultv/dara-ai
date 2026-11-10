"""Golden data for one full decoder layer of any Qwen3.5-family tier: the real transformers Qwen3_5DecoderLayer with the
layer's weights from the pinned checkpoint, on random input. Records each stage of the layer (input norm, mixer output,
first residual, post-mixer norm, MLP output, layer output) and, for DeltaNet layers, the final recurrent / conv state.

Usage: python -I tools/make_layer_golden.py --repo Qwen/Qwen3.5-4B --layer 0 --tokens 24 --out build/golden/layer_4b_l0.bin
All weights of the layer are stored as BF16 raw under "w.<name relative to the layer>"; dimensions are derived from
tensor shapes by the consumer, so the same file format serves every tier.
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
    ap.add_argument("--repo", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    api = json.loads(gp.get(f"https://huggingface.co/api/models/{a.repo}", 0, 1 << 22))
    rev = api["sha"]
    hf_get = lambda path, lo, hi: gp.get(f"https://huggingface.co/{a.repo}/resolve/{rev}/{path}", lo, hi)
    print(f"{a.repo} @ {rev[:10]}, layer {a.layer}")

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

    cfg = Qwen3_5TextConfig(**json.loads(hf_get("config.json", 0, 1 << 20))["text_config"])
    layer = Qwen3_5DecoderLayer(cfg, a.layer).float().eval()
    kind = layer.layer_type
    print(f"  layer type: {kind}, hidden {cfg.hidden_size}, intermediate {cfg.intermediate_size}")

    index = json.loads(hf_get("model.safetensors.index.json", 0, 10 << 20))
    prefix = f"model.language_model.layers.{a.layer}."
    names = sorted(n for n in index["weight_map"] if n.startswith(prefix))
    hdrs = {}
    raw = {}
    for n in names:
        shard = index["weight_map"][n]
        if shard not in hdrs:
            ln = struct.unpack("<Q", hf_get(shard, 0, 7))[0]
            hdrs[shard] = (ln, json.loads(hf_get(shard, 8, 8 + ln - 1)))
        ln, h = hdrs[shard]
        t = h[n]
        lo, hi = t["data_offsets"]
        buf = hf_get(shard, 8 + ln + lo, 8 + ln + hi - 1)
        rel = n[len(prefix):]
        if t["dtype"] == "BF16":
            raw[rel] = (np.frombuffer(buf, dtype="<u2").reshape(t["shape"]).copy(), 1)
        elif t["dtype"] == "F32":
            raw[rel] = (np.frombuffer(buf, dtype="<f4").reshape(t["shape"]).copy(), 0)
        else:
            raise SystemExit(f"unsupported dtype {t['dtype']} for {n}")
        print(f"  fetched {rel} {tuple(t['shape'])} {t['dtype']}")

    with torch.no_grad():
        for rel, (arr, code) in raw.items():
            val = torch.from_numpy((arr.astype(np.uint32) << 16).view(np.float32).copy() if code == 1 else arr.copy()).float()
            p = layer.get_parameter(rel)
            assert tuple(p.shape) == tuple(val.shape), (rel, p.shape, val.shape)
            p.copy_(val)

    cap = {}
    layer.input_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("normed1", o.detach().clone()))
    layer.post_attention_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("normed2", o.detach().clone()))
    layer.mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp", o.detach().clone()))
    mixer = layer.linear_attn if kind == "linear_attention" else layer.self_attn
    mixer.register_forward_hook(lambda m, i, o: cap.__setitem__("mixer", (o[0] if isinstance(o, tuple) else o).detach().clone()))
    if kind == "linear_attention":
        orig = mixer.chunk_gated_delta_rule

        def wrapped(query, key, value, g, beta, **kw):
            kw["output_final_state"] = True
            out, state = orig(query, key, value, g=g, beta=beta, **kw)
            cap["state"] = state.detach().clone()
            return out, state

        mixer.chunk_gated_delta_rule = wrapped

    T = a.tokens
    gen = torch.Generator().manual_seed(2024 + a.layer)
    x = torch.randn(1, T, cfg.hidden_size, generator=gen, dtype=torch.float32)
    pos = torch.arange(T)[None]
    with torch.no_grad():
        cos, sin = Qwen3_5TextRotaryEmbedding(cfg)(x, pos)
        mask = None if kind == "linear_attention" else torch.full((T, T), float("-inf")).triu(1)[None, None]
        out = layer(x, position_embeddings=(cos, sin), attention_mask=mask, position_ids=pos)
        resid1 = x + cap["mixer"]
        if kind == "linear_attention":
            qkv_pre = mixer.in_proj_qkv(cap["normed1"])

    f32 = lambda t: (np.ascontiguousarray(t.squeeze(0).float().numpy()), 0)
    tensors = {f"w.{k}": v for k, v in raw.items()}
    tensors.update({"x": f32(x), "normed1": f32(cap["normed1"]), "mixer": f32(cap["mixer"]), "resid1": f32(resid1),
                    "normed2": f32(cap["normed2"]), "mlp": f32(cap["mlp"]), "out": f32(out)})
    if kind == "linear_attention":
        tensors["final_state"] = (cap["state"].squeeze(0).numpy().copy(), 0)
        tensors["conv_state"] = (np.ascontiguousarray(qkv_pre.squeeze(0)[-(cfg.linear_conv_kernel_dim - 1):].T.numpy()), 0)
    tensors["meta.kind"] = (np.array([0 if kind == "linear_attention" else 1], dtype=np.float32), 0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    write(a.out, tensors)
    print(f"wrote {a.out}: T={T}, out |max|={out.abs().max():.3f}, rel diff residual identity "
          f"{((resid1 + cap['mlp']) - out).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
