"""Reference B for the Nano tier (Gemma 4 E2B): layer-by-layer fp32 forward over a token sequence, using the REAL transformers
Gemma 4 modules, with weights taken from the SAME quantized GGUF Dara runs (dequantized by the independent gguf-py decoder).

Memory: the fp32 model does not fit in RAM, so ONE decoder layer is built, loaded, run and freed at a time. The per-layer-input
tables (per_layer_token_embd, 1.6 GB quantized) are never expanded: only the rows of the fed tokens are dequantized. The tied
head streams token_embd in row chunks. Teacher forcing: logits at EVERY position (after final norm, tied head, soft-cap).

Usage: python -I tools/ref_forward_gemma.py --model models/gemma4-e2b --gguf models/gemma4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf \
           --tokens build/nano_prompt.tokens.txt --feed 64 --layer-tokens 8 --out build/refB_nano
Outputs: <out>.logits.bin (float32 [feed][vocab]), <out>.layers.bin (float32 [layers][layer-tokens][hidden]),
         tools/compare_logits.py works unchanged (--vocab 262144 --hidden 1536 --layers 35).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.modules["torchao"] = None  # incompatible with torch 2.5.1; not needed here


class GgufGemma:
    def __init__(self, path):
        from gguf import GGUFReader
        from gguf.quants import dequantize
        self.reader = GGUFReader(path)
        self.t = {t.name: t for t in self.reader.tensors}
        self.deq = dequantize

    def mat(self, name):
        """A matrix as float32 [out][in]. GGUF shape lists are [in, out]; gguf-py data is already [out][in]."""
        t = self.t[name]
        if t.tensor_type.name == "F32":
            a = np.asarray(t.data, dtype=np.float32)
        else:
            a = self.deq(t.data, t.tensor_type).astype(np.float32)
        return torch.from_numpy(a.reshape(-1, int(t.shape[0])).copy())

    def vec(self, name):
        return torch.from_numpy(np.asarray(self.t[name].data, dtype=np.float32).reshape(-1).copy())

    def rows(self, name, ids):
        t = self.t[name]
        return torch.from_numpy(np.stack([self.deq(t.data[i:i + 1], t.tensor_type).astype(np.float32).reshape(-1) for i in ids]))

    def head_chunks(self, step):
        t = self.t["output.weight" if "output.weight" in self.t else "token_embd.weight"]
        n = t.data.shape[0]
        for r in range(0, n, step):
            m = min(step, n - r)
            yield r, torch.from_numpy(self.deq(t.data[r:r + m], t.tensor_type).astype(np.float32).reshape(m, -1))

    def layer_params(self, i, layer, shared):
        b = f"blk.{i}."
        p = {
            "input_layernorm.weight": self.vec(b + "attn_norm.weight"),                       # plain weight (HF Gemma4RMSNorm multiplies by it directly)
            "post_attention_layernorm.weight": self.vec(b + "post_attention_norm.weight"),
            "pre_feedforward_layernorm.weight": self.vec(b + "ffn_norm.weight"),
            "post_feedforward_layernorm.weight": self.vec(b + "post_ffw_norm.weight"),
            "mlp.gate_proj.weight": self.mat(b + "ffn_gate.weight"),
            "mlp.up_proj.weight": self.mat(b + "ffn_up.weight"),
            "mlp.down_proj.weight": self.mat(b + "ffn_down.weight"),
            "self_attn.q_proj.weight": self.mat(b + "attn_q.weight"),
            "self_attn.o_proj.weight": self.mat(b + "attn_output.weight"),
            "self_attn.q_norm.weight": self.vec(b + "attn_q_norm.weight"),
            "per_layer_input_gate.weight": self.mat(b + "inp_gate.weight"),
            "per_layer_projection.weight": self.mat(b + "proj.weight"),
            "post_per_layer_input_norm.weight": self.vec(b + "post_norm.weight"),
        }
        if not shared:  # shared layers carry unused k/v tensors in the GGUF; HF has no such parameters there
            p["self_attn.k_proj.weight"] = self.mat(b + "attn_k.weight")
            p["self_attn.v_proj.weight"] = self.mat(b + "attn_v.weight")
            p["self_attn.k_norm.weight"] = self.vec(b + "attn_k_norm.weight")
        return p, self.vec(b + "layer_output_scale.weight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--feed", type=int, required=True)
    ap.add_argument("--layer-tokens", type=int, required=True, help="dump per-layer outputs for the first N positions")
    ap.add_argument("--logits-last", type=int, default=0, help="only compute logits for the last N positions (0 = all)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (Gemma4RMSNorm, Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding)

    raw = json.load(open(os.path.join(a.model, "config.json")))
    cfg = Gemma4TextConfig(**raw.get("text_config", raw))
    cfg._attn_implementation = "eager"
    ids = [int(t) for t in open(a.tokens).read().split()][: a.feed]
    K = len(ids)
    H, L, W = cfg.hidden_size, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input
    src = GgufGemma(a.gguf)
    print(f"Reference B (GGUF dequantized, real HF Gemma4 modules), {K} tokens, {L} layers")
    t0 = time.time()

    with torch.no_grad():
        # ---- embeddings and per-layer inputs (replicates Gemma4TextModel.get_per_layer_inputs / project_per_layer_inputs) ----
        embeds = src.rows("token_embd.weight", ids) * torch.tensor(H ** 0.5, dtype=torch.float32)          # [K, H]
        tok_pl = (src.rows("per_layer_token_embd.weight", ids) * torch.tensor(W ** 0.5, dtype=torch.float32)).reshape(K, L, W)
        proj = embeds @ src.mat("per_layer_model_proj.weight").T                                            # [K, L*W]
        proj = (proj * (H ** -0.5)).reshape(K, L, W)
        pl_norm = Gemma4RMSNorm(W, eps=cfg.rms_norm_eps).float()
        pl_norm.weight.copy_(src.vec("per_layer_proj_norm.weight"))
        per_layer_inputs = (pl_norm(proj) + tok_pl) * (2.0 ** -0.5)                                         # [K, L, W]

        hidden = embeds.unsqueeze(0)
        pos = torch.arange(K)[None]
        rotary = Gemma4TextRotaryEmbedding(cfg)
        pe = {lt: rotary(hidden, pos, lt) for lt in set(cfg.layer_types)}

        # rope_freqs.weight vs the HF-computed inv_freq of the full-attention (proportional) layers
        rf = src.vec("rope_freqs.weight")
        hf_inv = getattr(rotary, "full_attention_inv_freq")
        base = cfg.rope_parameters["full_attention"]["rope_theta"]
        hd = cfg.global_head_dim
        plain = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        implied = plain / rf
        n_rot = int(cfg.rope_parameters["full_attention"]["partial_rotary_factor"] * hd // 2)
        print(f"rope_freqs.weight: len {rf.numel()}, first {rf[:3].tolist()}, last {rf[-3:].tolist()}; HF inv_freq len {hf_inv.numel()}, rotated pairs {n_rot}")
        print(f"  max |implied(plain/rope_freqs) - HF inv_freq| over rotated pairs: {(implied[:n_rot] - hf_inv[:n_rot]).abs().max().item():.3e} "
              f"(rel {((implied[:n_rot] - hf_inv[:n_rot]).abs() / hf_inv[:n_rot]).max().item():.3e}); non-rotated pairs: HF inv_freq max {hf_inv[n_rot:].abs().max().item():.1e}, "
              f"rope_freqs min {rf[n_rot:].min().item():.3e}")

        causal = torch.full((K, K), float("-inf")).triu(1)
        sliding = causal.clone()
        for q in range(K):
            for k in range(0, max(0, q - cfg.sliding_window + 1)):
                sliding[q, k] = float("-inf")
        masks = {"full_attention": causal[None, None], "sliding_attention": sliding[None, None]}

        shared_kv = {}
        layer_dump = []
        first_shared = L - cfg.num_kv_shared_layers
        for i in range(L):
            layer = Gemma4TextDecoderLayer(cfg, i).float().eval()
            params, scalar = src.layer_params(i, layer, i >= first_shared)
            names = dict(layer.named_parameters())
            assert set(names) == set(params), (i, sorted(set(names) ^ set(params)))
            for n, p in names.items():
                assert tuple(p.shape) == tuple(params[n].shape), (i, n, tuple(p.shape), tuple(params[n].shape))
                p.copy_(params[n])
            layer.layer_scalar.copy_(scalar)
            lt = cfg.layer_types[i]
            hidden = layer(hidden, per_layer_inputs[:, i, :].unsqueeze(0), shared_kv_states=shared_kv, position_embeddings=pe[lt],
                           attention_mask=masks[lt], position_ids=pos, past_key_values=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            assert torch.isfinite(hidden).all(), f"non-finite output at layer {i}"
            layer_dump.append(hidden[0, : a.layer_tokens].numpy().copy())
            del layer, params
            print(f"  layer {i:2d} ({lt[:4]}) done ({time.time() - t0:.0f}s)", flush=True)

        norm = Gemma4RMSNorm(H, eps=cfg.rms_norm_eps).float()
        norm.weight.copy_(src.vec("output_norm.weight"))
        h = norm(hidden)[0]
        if a.logits_last:
            h = h[-a.logits_last:]
        logits = torch.empty(h.shape[0], cfg.vocab_size)
        for r, rows in src.head_chunks(16384):
            logits[:, r:r + rows.shape[0]] = h @ rows.T
        cap = cfg.final_logit_softcapping
        logits = torch.tanh(logits / cap) * cap

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logits.numpy().astype(np.float32).tofile(a.out + ".logits.bin")
    np.stack(layer_dump).astype(np.float32).tofile(a.out + ".layers.bin")
    print(f"wrote {a.out}.logits.bin [{logits.shape[0]} x {cfg.vocab_size}] and {a.out}.layers.bin in {time.time() - t0:.0f}s")
    last = int(logits[-1].argmax())
    print("argmax of last position:", last)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        print("decoded:", repr(tok.decode([last])), "| top5:", [tok.decode([int(x)]) for x in logits[-1].topk(5).indices])
    except Exception as e:  # noqa: BLE001
        print("(tokenizer decode skipped:", type(e).__name__, e, ")")


if __name__ == "__main__":
    main()
