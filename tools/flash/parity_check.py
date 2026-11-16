"""Compares the C++ Dara Flash drafter (tests/flash_parity_test) with the PyTorch model on the same inputs, using the exported (Q8_0) weights.
Usage: python -I tools/flash/parity_check.py --ckpt d.pt --data build/flash/q4b --embd build/flash/q4b_embd.f16 --seq I --n N --cpp out.bin
"""
import argparse, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_drafter import Drafter, load_data  # noqa: E402
from export_drafter import q8_0  # noqa: E402


def deq(w):
    raw = np.frombuffer(q8_0(w), np.uint8).reshape(w.shape[0], w.shape[1] // 32, 34)
    d = raw[..., :2].copy().view(np.float16).astype(np.float32)
    q = raw[..., 2:].copy().view(np.int8).astype(np.float32)
    return (q * d).reshape(w.shape)


def main():
    ap = argparse.ArgumentParser()
    for k in ["--ckpt", "--data", "--embd", "--cpp"]: ap.add_argument(k, required=True)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    c, keep = ck["config"], np.asarray(ck["keep"])
    H = c["hidden"]
    m = Drafter(H, c["heads"], c["kv_heads"], c["head_dim"], c["rot"], c["inter"], c["theta"])
    st = {k: (torch.tensor(deq(v.float().numpy())) if v.dim() == 2 else v.float()) for k, v in ck["state"].items()}
    m.load_state_dict(st)
    m.eval()
    toks, p, feats, _ = load_data(a.data, H)[a.seq]
    n = min(a.n, len(toks) - 1)
    embd = np.memmap(a.embd, dtype=np.float16, mode="r").reshape(-1, H)
    f = torch.tensor(np.asarray(feats[:n], np.float32))[None]
    e = torch.tensor(np.asarray(embd[toks[1: n + 1]], np.float32))[None]
    with torch.no_grad():
        y = m(f, e, torch.arange(n))[0, -1].numpy()
    head = np.asarray(embd[keep], np.float32)
    tok_py = int(keep[np.argmax(head @ y)])
    raw = open(a.cpp, "rb").read()
    for i, name in enumerate(["one call", "split calls"]):
        o = i * (4 + 4 * H)
        tok_c = int(np.frombuffer(raw, np.int32, 1, o)[0])
        yc = np.frombuffer(raw, np.float32, H, o + 4)
        cos = float(yc @ y / (np.linalg.norm(yc) * np.linalg.norm(y) + 1e-12))
        rel = float(np.linalg.norm(yc - y) / (np.linalg.norm(y) + 1e-12))
        print(f"{name}: cosine {cos:.6f}, rel err {rel:.2e}, token cpp {tok_c} py {tok_py}")


if __name__ == "__main__":
    main()
