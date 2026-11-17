"""Reference image features for Dara's Qwen3.5 (Core 4B / Pro 9B) vision path: the official transformers Qwen2-VL image processor and
Qwen3_5VisionModel with the weights read from the llama.cpp-format mmproj GGUF (clip / qwen3vl_merger).

Usage: python -I tools/vision/qwen35v_ref.py --mmproj models/qwen3.5-4b-gguf/mmproj-F16.gguf --config models/qwen3.5-4b/config.json
       --image build/vision/shapes.png --out build/vision/qref [--max-tokens 1024]
Outputs: <out>.patches.bin (float32 [n][3*2*16*16], normalized, temporal frames duplicated, merge-block order), <out>.feats.bin
         (float32 [n/4][out_hidden]), <out>.json (grid t/h/w).
"""
import argparse, json, struct, sys

import numpy as np
import torch
from PIL import Image

import importlib.util as _ilu   # the installed torchao may not match the installed torch; transformers imports it eagerly
_find_spec = _ilu.find_spec
_ilu.find_spec = lambda name, *a, **k: None if name == "torchao" or name.startswith("torchao.") else _find_spec(name, *a, **k)
sys.modules["torchao"] = None

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

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
    for _ in range(nk):
        rs(); t, = struct.unpack("<I", f.read(4)); skip(t)
    infos = []
    for _ in range(nt):
        n = rs(); nd, = struct.unpack("<I", f.read(4)); dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        t, = struct.unpack("<I", f.read(4)); off, = struct.unpack("<Q", f.read(8)); infos.append((n, dims, t, off))
    base = (f.tell() + 31) // 32 * 32
    out = {}
    for n, dims, t, off in infos:
        dt, es = GGML[t]
        f.seek(base + off)
        a = np.frombuffer(f.read(int(np.prod(dims)) * es), dtype=dt).astype(np.float32)
        out[n] = a.reshape(tuple(reversed(dims)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024, help="cap on merged image tokens (max_pixels = tokens * 32 * 32)")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    vc = Qwen3_5VisionConfig(**cfg["vision_config"])
    vc._attn_implementation = "eager"
    W = read_gguf(a.mmproj)
    vm = Qwen3_5VisionModel(vc).float().eval()
    sd = {"patch_embed.proj.weight": np.stack([W["v.patch_embd.weight"], W["v.patch_embd.weight.1"]], axis=2),   # [out][c][t][y][x]
          "patch_embed.proj.bias": W["v.patch_embd.bias"], "pos_embed.weight": W["v.position_embd.weight"],
          "merger.norm.weight": W["v.post_ln.weight"], "merger.norm.bias": W["v.post_ln.bias"],
          "merger.linear_fc1.weight": W["mm.0.weight"], "merger.linear_fc1.bias": W["mm.0.bias"],
          "merger.linear_fc2.weight": W["mm.2.weight"], "merger.linear_fc2.bias": W["mm.2.bias"]}
    names = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj", "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}
    for i in range(vc.depth):
        for g, h in names.items():
            for wb in ("weight", "bias"):
                sd[f"blocks.{i}.{h}.{wb}"] = W[f"v.blk.{i}.{g}.{wb}"]
    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in sd.items()}
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected: print("missing", missing, "unexpected", unexpected); sys.exit(1)

    max_pixels = a.max_tokens * 32 * 32
    proc = Qwen2VLImageProcessorPil(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
                                    size={"shortest_edge": 65536, "longest_edge": max_pixels})
    img = Image.open(a.image).convert("RGB")
    bf = proc(images=[img], return_tensors="pt")
    pv, thw = bf["pixel_values"].float(), bf["image_grid_thw"]
    with torch.no_grad():
        feats = vm(pv, grid_thw=thw).pooler_output
    pv.numpy().astype(np.float32).tofile(a.out + ".patches.bin")
    f = feats.numpy().astype(np.float32)
    f.tofile(a.out + ".feats.bin")
    t, h, w = [int(x) for x in thw[0]]
    json.dump({"grid_t": t, "grid_h": h, "grid_w": w, "patches": int(pv.shape[0]), "tokens": int(f.shape[0]), "dim": int(f.shape[1])}, open(a.out + ".json", "w"))
    print(f"grid t{t} h{h} w{w}: {pv.shape[0]} patches -> {f.shape[0]} tokens x {f.shape[1]}, feat rms {float(np.sqrt((f ** 2).mean())):.4f}")


if __name__ == "__main__":
    main()
