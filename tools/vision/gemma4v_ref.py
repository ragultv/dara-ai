"""Reference image features for Dara's Gemma 4 (Nano) vision path: the official transformers Gemma 4 vision modules (image processor,
vision tower, multimodal embedder) with the weights read from the llama.cpp-format mmproj GGUF. Writes the preprocessed patches and the
final soft tokens (what goes into the text model) so the C++ path can be compared number by number.

Usage: python -I tools/vision/gemma4v_ref.py --mmproj models/gemma4-e2b-gguf/mmproj-F16.gguf --config models/gemma4-e2b/config.json
       --image build/vision/shapes.png --out build/vision/ref [--patch-order cyx|yxc]
Outputs: <out>.patches.bin (float32 [n][768], pixel values in [0,1] as the processor gives them), <out>.pos.bin (int32 [n][2] x,y),
         <out>.feats.bin (float32 [n/9][1536]), <out>.json (sizes).
"""
import argparse, json, struct, sys

import numpy as np
import torch
from PIL import Image

# The installed torchao may not match the installed torch; transformers imports it eagerly when present. This script needs none of it.
import importlib.util as _ilu
_find_spec = _ilu.find_spec
_ilu.find_spec = lambda name, *a, **k: None if name == "torchao" or name.startswith("torchao.") else _find_spec(name, *a, **k)
sys.modules["torchao"] = None

from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig, Gemma4TextConfig
from transformers.models.gemma4.image_processing_pil_gemma4 import Gemma4ImageProcessorPil
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel, Gemma4MultimodalEmbedder

GGML = {0: (np.float32, 4), 1: (np.float16, 2)}


def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    struct.unpack("<I", f.read(4))
    nt, nk = struct.unpack("<QQ", f.read(16))
    def rs():
        n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        if t == 8: rs()
        elif t == 9:
            at, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
            if at == 8:
                for _ in range(n): rs()
            else: f.read(sz[at] * n)
        else: f.read(sz[t])
    align = 32
    for _ in range(nk):
        k = rs(); t, = struct.unpack("<I", f.read(4)); skip(t)
    infos = []
    for _ in range(nt):
        n = rs(); nd, = struct.unpack("<I", f.read(4)); dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        t, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8)); infos.append((n, dims, t, off))
    base = (f.tell() + align - 1) // align * align
    out = {}
    for n, dims, t, off in infos:
        if not n.startswith(("v.", "mm.input")): continue
        dt, es = GGML[t]
        cnt = int(np.prod(dims))
        f.seek(base + off)
        a = np.frombuffer(f.read(cnt * es), dtype=dt).astype(np.float32)
        out[n] = a.reshape(tuple(reversed(dims)))   # numpy order: last GGUF dim first
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patch-order", default="cyx", help="layout of v.patch_embd in the GGUF: cyx = conv [out][c][y][x]")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    vc = Gemma4VisionConfig(**cfg["vision_config"])
    vc._attn_implementation = "eager"
    tc = Gemma4TextConfig(**cfg["text_config"])
    W = read_gguf(a.mmproj)

    vm = Gemma4VisionModel(vc).float().eval()
    sd = {}
    pe = W["v.patch_embd.weight"]                         # numpy [768][3][16][16] for "cyx"
    if a.patch_order == "cyx": lin = pe.transpose(0, 2, 3, 1).reshape(pe.shape[0], -1)   # -> [out][y][x][c], the processor's patch order
    else: lin = pe.reshape(pe.shape[0], -1)
    sd["patch_embedder.input_proj.weight"] = lin
    sd["patch_embedder.position_embedding_table"] = W["v.position_embd.weight"]   # [2][10240][768]
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_out": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    norms = {"ln1": "input_layernorm", "attn_post_norm": "post_attention_layernorm", "ln2": "pre_feedforward_layernorm",
             "ffn_post_norm": "post_feedforward_layernorm", "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm"}
    for i in range(vc.num_hidden_layers):
        p = f"v.blk.{i}."
        for g, h in names.items():
            sd[f"encoder.layers.{i}.{h}.linear.weight"] = W[p + g + ".weight"]
            for b in ("input_min", "input_max", "output_min", "output_max"):
                sd[f"encoder.layers.{i}.{h}.{b}"] = W[p + g + "." + b].reshape(())
        for g, h in norms.items():
            sd[f"encoder.layers.{i}.{h}.weight"] = W[p + g + ".weight"]
    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary" not in m and "inv_freq" not in m]
    if missing or unexpected: print("missing", missing, "unexpected", unexpected); sys.exit(1)

    emb = Gemma4MultimodalEmbedder(vc, tc).float().eval()
    emb.load_state_dict({"embedding_projection.weight": torch.from_numpy(np.ascontiguousarray(W["mm.input_projection.weight"]))})

    proc = Gemma4ImageProcessorPil()
    img = Image.open(a.image).convert("RGB")
    bf = proc(images=[img], return_tensors="pt")
    pv, pos = bf["pixel_values"].float(), bf["image_position_ids"]
    n_soft = int(bf["num_soft_tokens_per_image"][0])
    with torch.no_grad():
        h = vm(pixel_values=pv, pixel_position_ids=pos).last_hidden_state
        feats = emb(h)
    real = (pos[0] != -1).all(-1)
    patches = pv[0][real].numpy().astype(np.float32)
    pos_real = pos[0][real].numpy().astype(np.int32)
    patches.tofile(a.out + ".patches.bin"); pos_real.tofile(a.out + ".pos.bin")
    f = feats.reshape(-1, feats.shape[-1]).numpy().astype(np.float32)
    f.tofile(a.out + ".feats.bin")
    json.dump({"patches": int(patches.shape[0]), "soft_tokens": n_soft, "feat_rows": int(f.shape[0]), "dim": int(f.shape[1]),
               "grid_w": int(pos_real[:, 0].max() + 1), "grid_h": int(pos_real[:, 1].max() + 1)}, open(a.out + ".json", "w"))
    print(f"patches {patches.shape[0]} (grid {pos_real[:,0].max()+1}x{pos_real[:,1].max()+1}), soft tokens {n_soft}, feats {f.shape}, "
          f"feat rms {float(np.sqrt((f**2).mean())):.4f}")


if __name__ == "__main__":
    main()
