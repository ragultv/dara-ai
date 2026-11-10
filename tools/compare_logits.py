"""Compare a Dara run against the reference: per-layer states, logits at every position, top-k agreement, greedy tokens.

Usage: python -I tools/compare_logits.py --dara build/run --ref build/ref --prompt-len T0 --vocab V --hidden H --layers L
Exit code 0 only if every greedy token matches the reference argmax AND all logit errors are within --tol (relative to the
largest |logit| at that position).
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dara", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--prompt-len", type=int, required=True)
    ap.add_argument("--vocab", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--tol", type=float, default=5e-3)
    a = ap.parse_args()

    seq = [int(t) for t in open(a.dara + ".tokens.txt").read().split()]
    T0 = a.prompt_len
    dl = np.fromfile(a.dara + ".logits.bin", dtype=np.float32).reshape(-1, a.vocab)
    rl = np.fromfile(a.ref + ".logits.bin", dtype=np.float32).reshape(-1, a.vocab)
    P = dl.shape[0]
    assert rl.shape[0] == P, f"Dara has {P} positions, reference has {rl.shape[0]}"
    assert len(seq) == P + 1

    # ---- per-layer hidden states over the prompt (accumulated error across the stack) --------------
    dh = np.fromfile(a.dara + ".layers.bin", dtype=np.float32).reshape(a.layers, T0, a.hidden)
    rh = np.fromfile(a.ref + ".layers.bin", dtype=np.float32).reshape(a.layers, -1, a.hidden)[:, :T0]
    print("layer output vs reference (prompt positions): max|err| / max|ref|   and   mean|err| / mean|ref|")
    worst_layer = 0.0
    for l in range(a.layers):
        e = np.abs(dh[l] - rh[l])
        rel_max = e.max() / np.abs(rh[l]).max()
        rel_mean = e.mean() / np.abs(rh[l]).mean()
        worst_layer = max(worst_layer, rel_max)
        if l % 4 == 3 or l in (0, a.layers - 1) or rel_max > 1e-3:
            print(f"  layer {l:2d}: {rel_max:.3e}   {rel_mean:.3e}")
    print(f"  worst layer max-rel error: {worst_layer:.3e}")

    # ---- logits at every position ---------------------------------------------------------------
    max_abs, max_rel, top1_ok = 0.0, 0.0, 0
    top5 = top10 = 0.0
    mismatches = []
    for p in range(P):
        d, r = dl[p].astype(np.float64), rl[p].astype(np.float64)
        err = np.abs(d - r)
        max_abs = max(max_abs, err.max())
        max_rel = max(max_rel, err.max() / np.abs(r).max())
        td, tr = int(d.argmax()), int(r.argmax())
        top1_ok += td == tr
        if td != tr:
            srt = np.sort(r)[::-1]
            mismatches.append((p, td, tr, float(srt[0] - srt[1])))
        for k, acc in ((5, 0), (10, 0)):
            s = len(set(np.argpartition(-d, k)[:k]) & set(np.argpartition(-r, k)[:k])) / k
            if k == 5: top5 += s
            else: top10 += s
    print(f"\nlogits over {P} positions x {a.vocab} vocabulary:")
    print(f"  max |abs error| {max_abs:.4e} | max relative error (to the position's largest |logit|) {max_rel:.4e}")
    print(f"  top-1 agreement {top1_ok}/{P} | mean top-5 overlap {top5 / P:.3f} | mean top-10 overlap {top10 / P:.3f}")
    for p, td, tr, margin in mismatches:
        print(f"  top-1 mismatch at position {p}: Dara {td}, reference {tr} (reference top-1/top-2 margin {margin:.4f})")

    # ---- greedy tokens: g_k is the argmax of the logits at position T0-1+k-1 ------------------------
    gen = seq[T0:]
    ref_arg = [int(rl[T0 - 1 + k].argmax()) for k in range(len(gen))]
    greedy_ok = gen == ref_arg
    print(f"\ngreedy continuation ({len(gen)} tokens): Dara {gen}")
    print(f"                              reference argmax {ref_arg} -> {'MATCH' if greedy_ok else 'MISMATCH'}")

    ok = greedy_ok and max_rel <= a.tol
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} (tolerance {a.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
