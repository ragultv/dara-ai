"""Reference forward pass for a Qwen3.5-family model, layer by layer (fp32), using the real transformers modules.

The model is embed -> N decoder layers -> final RMSNorm -> tied output head. This runs exactly that, but loads ONE layer's
weights at a time, so a checkpoint larger than RAM can be executed. It computes logits at EVERY position of the given token
sequence (teacher forcing), so a single pass verifies all of Dara's greedy steps and all of its logits.

Two weight sources, both through the same HF modules:
  Reference A (default)  : the BF16 checkpoint (safetensors). The numerical reference.
  Reference B (--gguf F) : the SAME quantized weights Dara runs, dequantized with the independent gguf-py decoder and mapped
                           back to checkpoint layout (V-head order un-permuted, folded +1 of the four RMSNorms removed,
                           ssm_a -> A_log). It isolates Dara's runtime error from quantization error: Dara-on-GGUF should
                           match Reference B closely, while Reference A vs Reference B shows the quantization error.

Usage: python -I tools/ref_forward.py --model models/qwen3.5-4b --tokens build/run.tokens.txt --feed 20 --layer-tokens 12 --out build/ref
       python -I tools/ref_forward.py --model models/qwen3.5-4b --gguf models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf ... --out build/refB
Outputs: <out>.logits.bin (float32 [feed][vocab]), <out>.layers.bin (float32 [layers][layer-tokens][hidden])
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.modules["torchao"] = None  # incompatible with torch 2.5.1; not needed here


class Bf16Source:
    """Reference A: tensors straight from the safetensors checkpoint."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        self.dir = model_dir
        self.index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}
        self._open = safe_open

    def _h(self, name):
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = self._open(os.path.join(self.dir, shard), framework="pt")
        return self.handles[shard]

    def layer_params(self, i, kind, names):
        prefix = f"model.language_model.layers.{i}."
        return {n: self._h(prefix + n).get_tensor(prefix + n).float() for n in names}

    def embed_rows(self, ids):
        name = "model.language_model.embed_tokens.weight"
        sl = self._h(name).get_slice(name)
        return torch.stack([sl[i : i + 1][0].float() for i in ids])

    def head_chunks(self, step):
        name = "model.language_model.embed_tokens.weight"
        sl = self._h(name).get_slice(name)
        n = sl.get_shape()[0]
        for r in range(0, n, step):
            yield r, sl[r : min(r + step, n)].float()

    def final_norm_weight(self):
        name = "model.language_model.norm.weight"
        return self._h(name).get_tensor(name).float()


class GgufSource:
    """Reference B: the quantized GGUF, dequantized by gguf-py and mapped back to checkpoint layout."""

    def __init__(self, path, cfg):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.deq = dequantize
        self.cfg = cfg
        self.kh, self.vh = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        self.ratio = self.vh // self.kh
        # GGUF V head j holds checkpoint head h where j = k_heads*(h % ratio) + h // ratio
        self.gguf_of_hf = np.array([self.kh * (h % self.ratio) + h // self.ratio for h in range(self.vh)])

    def _f(self, name):
        t = self.t[name]
        return torch.from_numpy(self.deq(t.data, t.tensor_type).astype(np.float32).reshape(-1, t.shape[0]) if t.data.ndim > 1 or t.tensor_type != 0
                                else np.asarray(t.data, dtype=np.float32).copy())

    def _vec(self, name):
        return torch.from_numpy(np.asarray(self.t[name].data, dtype=np.float32).reshape(-1).copy())

    def _unperm_rows(self, w, base, head_rows):
        """w: GGUF tensor [rows, cols]; rows base.. are V heads (head_rows each) in GGUF order -> checkpoint order."""
        w = w.clone()
        v = w[base : base + self.vh * head_rows].reshape(self.vh, head_rows, -1)
        w[base : base + self.vh * head_rows] = v[self.gguf_of_hf].reshape(self.vh * head_rows, -1)
        return w

    def layer_params(self, i, kind, names):
        b = f"blk.{i}."
        p = {}
        p["input_layernorm.weight"] = self._vec(b + "attn_norm.weight") - 1.0
        p["post_attention_layernorm.weight"] = self._vec(b + "post_attention_norm.weight") - 1.0
        p["mlp.gate_proj.weight"] = self._f(b + "ffn_gate.weight")
        p["mlp.up_proj.weight"] = self._f(b + "ffn_up.weight")
        p["mlp.down_proj.weight"] = self._f(b + "ffn_down.weight")
        if kind == "linear_attention":
            kd = self.kh * 128
            p["linear_attn.in_proj_qkv.weight"] = self._unperm_rows(self._f(b + "attn_qkv.weight"), 2 * kd, 128)
            p["linear_attn.in_proj_z.weight"] = self._unperm_rows(self._f(b + "attn_gate.weight"), 0, 128)
            p["linear_attn.in_proj_b.weight"] = self._unperm_rows(self._f(b + "ssm_beta.weight"), 0, 1)
            p["linear_attn.in_proj_a.weight"] = self._unperm_rows(self._f(b + "ssm_alpha.weight"), 0, 1)
            conv = self._vec(b + "ssm_conv1d.weight").reshape(-1, 4)  # [channel][tap]
            conv = conv.clone()
            cv = conv[2 * kd :].reshape(self.vh, 128, 4)
            conv[2 * kd :] = cv[self.gguf_of_hf].reshape(self.vh * 128, 4)
            p["linear_attn.conv1d.weight"] = conv.reshape(-1, 1, 4)
            p["linear_attn.A_log"] = torch.log(-self._vec(b + "ssm_a"))[self.gguf_of_hf]
            p["linear_attn.dt_bias"] = self._vec(b + "ssm_dt.bias")[self.gguf_of_hf]
            p["linear_attn.norm.weight"] = self._vec(b + "ssm_norm.weight")
            out = self._f(b + "ssm_out.weight")  # [hidden, value_dim], columns in GGUF head order
            p["linear_attn.out_proj.weight"] = out.reshape(out.shape[0], self.vh, 128)[:, self.gguf_of_hf].reshape(out.shape[0], -1)
        else:
            p["self_attn.q_proj.weight"] = self._f(b + "attn_q.weight")
            p["self_attn.k_proj.weight"] = self._f(b + "attn_k.weight")
            p["self_attn.v_proj.weight"] = self._f(b + "attn_v.weight")
            p["self_attn.o_proj.weight"] = self._f(b + "attn_output.weight")
            p["self_attn.q_norm.weight"] = self._vec(b + "attn_q_norm.weight") - 1.0
            p["self_attn.k_norm.weight"] = self._vec(b + "attn_k_norm.weight") - 1.0
        return {n: p[n] for n in names}

    def embed_rows(self, ids):
        t = self.t["token_embd.weight"]
        return torch.from_numpy(np.stack([self.deq(t.data[i : i + 1], t.tensor_type).astype(np.float32).reshape(-1) for i in ids]))

    def head_chunks(self, step):
        t = self.t["output.weight" if "output.weight" in self.t else "token_embd.weight"]  # untied models have their own head
        n = t.data.shape[0]
        for r in range(0, n, step):
            yield r, torch.from_numpy(self.deq(t.data[r : min(r + step, n)], t.tensor_type).astype(np.float32).reshape(min(step, n - r), -1))

    def final_norm_weight(self):
        return self._vec("output_norm.weight") - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--feed", type=int, required=True)
    ap.add_argument("--layer-tokens", type=int, required=True, help="dump per-layer outputs for the first N positions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (Qwen3_5DecoderLayer, Qwen3_5RMSNorm,
                                                              Qwen3_5TextRotaryEmbedding)

    cfg = Qwen3_5TextConfig(**json.load(open(os.path.join(a.model, "config.json")))["text_config"])
    ids = [int(t) for t in open(a.tokens).read().split()][: a.feed]
    K = len(ids)
    src = GgufSource(a.gguf, cfg) if a.gguf else Bf16Source(a.model)
    print(f"reference weights: {'GGUF dequantized (Reference B)' if a.gguf else 'BF16 checkpoint (Reference A)'}, {K} tokens")

    t0 = time.time()
    hidden = src.embed_rows(ids).unsqueeze(0)  # [1, K, H]
    pos = torch.arange(K)[None]
    cos_sin = Qwen3_5TextRotaryEmbedding(cfg)(hidden, pos)
    mask = torch.full((K, K), float("-inf")).triu(1)[None, None]
    layer_dump = []
    for i in range(cfg.num_hidden_layers):
        layer = Qwen3_5DecoderLayer(cfg, i).float().eval()
        names = [n for n, _ in layer.named_parameters()]
        params = src.layer_params(i, layer.layer_type, names)
        with torch.no_grad():
            for n, p in layer.named_parameters():
                assert tuple(p.shape) == tuple(params[n].shape), (i, n, tuple(p.shape), tuple(params[n].shape))
                p.copy_(params[n])
            hidden = layer(hidden, position_embeddings=cos_sin, attention_mask=None if layer.layer_type == "linear_attention" else mask, position_ids=pos)
        layer_dump.append(hidden[0, : a.layer_tokens].numpy().copy())
        del layer, params
        print(f"  layer {i:2d} done ({time.time() - t0:.0f}s)", flush=True)

    norm = Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).float()
    with torch.no_grad():
        norm.weight.copy_(src.final_norm_weight())
        h = norm(hidden)[0]  # [K, H]
        logits = torch.empty(K, cfg.vocab_size)
        for r, rows in src.head_chunks(16384):
            logits[:, r : r + rows.shape[0]] = h @ rows.T  # tied head: logits = h @ E^T
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logits.numpy().astype(np.float32).tofile(a.out + ".logits.bin")
    np.stack(layer_dump).astype(np.float32).tofile(a.out + ".layers.bin")
    print(f"wrote {a.out}.logits.bin [{K} x {cfg.vocab_size}] and {a.out}.layers.bin in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
