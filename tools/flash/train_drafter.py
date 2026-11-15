"""Train a Dara Flash drafter (EAGLE-style) on self-distillation data from tools/flash_data_qwen.

Drafter, per position t (causal over the sequence):
    x_t   = fc([f_t ; emb(tok_{t+1})])                    f_t: the target's final (normed) hidden state at t; tok_{t+1}: the token that followed
    y_t   = DecoderLayer(x_t)                              one Qwen3.5-style attention layer (gated attention, partial RoPE, q/k norm) + SwiGLU MLP
    f^_{t+1} = y_t   ~ f_{t+1}                             predicted next feature (regression)
    logits = head(f^_{t+1}) over a reduced vocabulary     head rows = the target's own output head (frozen) -> predicts tok_{t+2}
The layer has the target's attention shape (heads, kv heads, head_dim 256, rot_dim 64) so Dara's runtime can execute it with its existing kernels.

Usage: python -I tools/flash/train_drafter.py --data build/flash/q4b --embd build/flash/q4b_embd.f16 --out build/flash/drafter_4b.pt
       [--hidden 2560 --heads 16 --kv-heads 4 --head-dim 256 --rot 64 --inter 4096 --vocab-k 32768 --epochs 4]
"""
import argparse, json, math, os, random, struct, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_data(d, hidden):
    index = [json.loads(l) for l in open(os.path.join(d, "index.jsonl"), encoding="utf-8") if l.strip()]
    raw = open(os.path.join(d, "seqs.bin"), "rb").read()
    seqs, off = [], 0
    while off + 8 <= len(raw):
        n, p = struct.unpack_from("ii", raw, off); off += 8
        if off + 4 * n > len(raw): break          # a sequence still being written (data generation running)
        seqs.append((np.frombuffer(raw, np.int32, n, off).copy(), p)); off += 4 * n
    feats = np.memmap(os.path.join(d, "feats.bin"), dtype=np.float16, mode="r")
    out = []
    for (toks, p), ix in zip(seqs, index):
        n = len(toks)
        if ix["feat_offset"] + n * hidden > feats.shape[0]: break
        f = feats[ix["feat_offset"]: ix["feat_offset"] + n * hidden].reshape(n, hidden)
        out.append((toks, p, f))
    return out


class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(n)); self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.w


def rope(x, pos, rot, theta):   # NeoX-style partial rotation of the first `rot` dims; x [B, H, T, D]
    half = rot // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    ang = pos[:, None].float() * inv[None, :]                      # [T, half]
    c, s = ang.cos()[None, None], ang.sin()[None, None]
    a, b = x[..., :half], x[..., half:rot]
    return torch.cat([a * c - b * s, b * c + a * s, x[..., rot:]], dim=-1)


class Drafter(nn.Module):
    def __init__(self, hidden, heads, kv_heads, head_dim, rot, inter, theta):
        super().__init__()
        self.h, self.kv, self.d, self.rot, self.theta = heads, kv_heads, head_dim, rot, theta
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.in_norm = RMSNorm(hidden)
        self.q_proj = nn.Linear(hidden, heads * head_dim * 2, bias=False)    # [query | gate] per head, as Qwen3.5
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm, self.k_norm = RMSNorm(head_dim), RMSNorm(head_dim)
        self.post_norm = RMSNorm(hidden)
        self.gate = nn.Linear(hidden, inter, bias=False)
        self.up = nn.Linear(hidden, inter, bias=False)
        self.down = nn.Linear(inter, hidden, bias=False)

    def forward(self, f, e, pos):
        B, T, _ = f.shape
        x = self.fc(torch.cat([f, e], dim=-1))
        h = self.in_norm(x)
        qg = self.q_proj(h).view(B, T, self.h, 2 * self.d)
        q, gate = qg[..., : self.d], qg[..., self.d:]
        k = self.k_proj(h).view(B, T, self.kv, self.d)
        v = self.v_proj(h).view(B, T, self.kv, self.d)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = rope(q, pos, self.rot, self.theta), rope(k, pos, self.rot, self.theta)
        rep = self.h // self.kv
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, self.h * self.d)
        a = a * torch.sigmoid(gate.reshape(B, T, self.h * self.d))
        x = x + self.o_proj(a)
        m = self.post_norm(x)
        return x + self.down(F.silu(self.gate(m)) * self.up(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--embd", required=True, help="target token embedding / output head, fp16 [vocab][hidden] (tools/gguf_dump_tensor)")
    ap.add_argument("--head", default=None, help="separate output head fp16 (untied models); default: --embd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=2560)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--rot", type=int, default=64)
    ap.add_argument("--theta", type=float, default=1e7)
    ap.add_argument("--inter", type=int, default=4096)
    ap.add_argument("--vocab-k", type=int, default=32768)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--noise", type=float, default=0.2, help="uniform noise on input features (EAGLE): robustness to the drafter's own errors")
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seqs", type=int, default=0, help="use only the first N sequences (smoke tests)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    a = ap.parse_args()
    if os.name == "nt":   # keep the system out of idle standby while training (the dGPU may not come back from Modern Standby)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    torch.manual_seed(0); random.seed(0)
    dev = a.device
    H = a.hidden
    data = load_data(a.data, H)
    if a.max_seqs: data = data[: a.max_seqs]
    print(f"{len(data)} sequences, {sum(len(t) for t, _, _ in data)} tokens")
    vocab_total = os.path.getsize(a.embd) // (2 * H)
    embd = np.memmap(a.embd, dtype=np.float16, mode="r").reshape(vocab_total, H)
    headm = embd if a.head is None else np.memmap(a.head, dtype=np.float16, mode="r").reshape(-1, H)
    # reduced vocabulary: the most frequent tokens of the target's own answers (and prompts)
    counts = np.zeros(vocab_total, np.int64)
    for toks, p, _ in data: np.add.at(counts, toks, 1)
    keep = np.argsort(-counts)[: a.vocab_k]
    keep.sort()
    remap = np.full(vocab_total, -1, np.int64); remap[keep] = np.arange(len(keep))
    cover = counts[keep].sum() / counts.sum()
    print(f"reduced vocabulary {len(keep)} tokens covers {100 * cover:.2f}% of token occurrences")
    head_w = torch.tensor(np.asarray(headm[keep], np.float32), device=dev)          # [K, H] frozen
    emb_cache = {}

    model = Drafter(H, a.heads, a.kv_heads, a.head_dim, a.rot, a.inter, a.theta).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"drafter parameters: {nparam / 1e6:.1f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    random.shuffle(data)
    hold, train = data[: a.holdout], data[a.holdout:]
    steps = a.epochs * math.ceil(len(train) / a.batch)
    if a.max_steps: steps = min(steps, a.max_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    remap_t = torch.tensor(remap, device=dev)

    def batch_tensors(batch, noise):
        T = max(len(t) for t, _, _ in batch) - 1
        B = len(batch)
        f = torch.zeros(B, T, H, device=dev); e = torch.zeros(B, T, H, device=dev); ft = torch.zeros(B, T, H, device=dev)
        tgt = torch.full((B, T), -1, dtype=torch.long, device=dev); mask = torch.zeros(B, T, device=dev)
        for i, (toks, p, feats) in enumerate(batch):
            n = len(toks) - 1                              # positions t = 0 .. n-1 predict feature t+1
            fe = torch.tensor(np.asarray(feats, np.float32), device=dev)
            f[i, :n] = fe[:n]
            ft[i, :n] = fe[1: n + 1]
            e[i, :n] = torch.tensor(np.asarray(embd[toks[1: n + 1]], np.float32), device=dev)
            nxt = torch.tensor(toks[2: n + 1].astype(np.int64), device=dev)   # tok_{t+2} for t <= n-2
            tgt[i, : n - 1] = remap_t[nxt]
            mask[i, max(p - 1, 0): n] = 1.0                 # loss on the answer region (the context still includes the prompt)
        if noise > 0:
            f = f + (torch.rand_like(f) - 0.5) * 2 * noise
        return f, e, ft, tgt, mask, torch.arange(T, device=dev)

    def evaluate():
        model.eval()
        hit = tot = 0
        with torch.no_grad():
            for i in range(0, len(hold), a.batch):
                f, e, ft, tgt, mask, pos = batch_tensors(hold[i: i + a.batch], 0.0)
                y = model(f, e, pos)
                pred = (y @ head_w.T).argmax(-1)
                ok = (tgt >= 0) & (mask > 0)
                hit += ((pred == tgt) & ok).sum().item(); tot += ok.sum().item()
        model.train()
        return hit / max(tot, 1)

    print(f"held-out next-token accuracy before training: {evaluate():.3f}")
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        random.shuffle(train)
        for i in range(0, len(train), a.batch):
            if step >= steps: break
            f, e, ft, tgt, mask, pos = batch_tensors(train[i: i + a.batch], a.noise)
            y = model(f, e, pos)
            reg = (F.smooth_l1_loss(y, ft, reduction="none").mean(-1) * mask).sum() / mask.sum()
            logits = y @ head_w.T
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-1, reduction="none").view_as(mask)
            ok = (tgt >= 0).float() * mask
            ce = (ce * ok).sum() / ok.sum().clamp(min=1)
            loss = reg + 0.1 * ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                print(f"epoch {ep} step {step}/{steps}: loss {loss.item():.4f} (feature {reg.item():.4f}, token CE {ce.item():.3f}) {time.time() - t0:.0f}s", flush=True)
        print(f"epoch {ep}: held-out next-token accuracy {evaluate():.3f}", flush=True)
    torch.save({"state": model.state_dict(), "keep": keep, "config": vars(a)}, a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
