"""Export a trained Dara Flash drafter (tools/flash/train_drafter.py checkpoint) to the runtime format read by src/model/flash_drafter.cu.

Layout: b"DFLASH01", int32 hidden heads kv_heads head_dim rot inter K, float32 theta eps, int32 keep[K],
        Q8_0 matrices (GGUF block layout: fp16 scale + 32 int8) fc, q, k, v, o, gate, up, down, then float32 in_norm, q_norm, k_norm, post_norm.
Usage: python -I tools/flash/export_drafter.py --ckpt build/flash/drafter_4b.pt --out build/flash/drafter_4b.dflash
"""
import argparse, struct

import numpy as np
import torch


def q8_0(w):
    w = np.asarray(w, np.float32)
    rows, cols = w.shape
    assert cols % 32 == 0, w.shape
    b = w.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(-1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    df = d.astype(np.float32)
    q = np.where(df > 0, np.round(b / np.where(df > 0, df, 1)), 0).clip(-127, 127).astype(np.int8)
    out = np.empty((rows, cols // 32, 34), np.uint8)
    out[..., :2] = d.view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st, keep, c = ck["state"], np.asarray(ck["keep"], np.int32), ck["config"]
    g = lambda k: st[k].float().numpy()
    with open(a.out, "wb") as f:
        f.write(b"DFLASH01")
        f.write(struct.pack("<7i", c["hidden"], c["heads"], c["kv_heads"], c["head_dim"], c["rot"], c["inter"], len(keep)))
        f.write(struct.pack("<2f", c["theta"], a.eps))
        f.write(keep.astype("<i4").tobytes())
        for k in ["fc", "q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down"]:
            f.write(q8_0(g(k + ".weight")))
        for k in ["in_norm", "q_norm", "k_norm", "post_norm"]:
            f.write(g(k + ".w").astype("<f4").tobytes())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
