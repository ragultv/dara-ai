"""KV-format quality: compare the last-N-position logits of a reference run (FP32 KV) and a test run (e.g. TurboQuant 4-bit KV).
Usage: python -I bench/kv_quality.py <ref prefix> <test prefix> <vocab> [N]    (reads <prefix>.logits.bin, float32 [N][vocab])
Reports, per position and in aggregate: top-1 agreement, max |logit error| relative to the position's largest |logit|, KL(ref || test) of the
softmax distributions, and the rank of the reference top-1 token in the test logits."""
import sys
import numpy as np

ref_p, test_p, V = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = np.fromfile(ref_p + ".logits.bin", dtype=np.float32).reshape(-1, V)
t = np.fromfile(test_p + ".logits.bin", dtype=np.float32).reshape(-1, V)
n = min(len(r), len(t), int(sys.argv[4]) if len(sys.argv) > 4 else 10**9)
r, t = r[:n].astype(np.float64), t[:n].astype(np.float64)

def logsoftmax(x):
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))

lr, lt = logsoftmax(r), logsoftmax(t)
kl = (np.exp(lr) * (lr - lt)).sum(axis=1)
rel = np.abs(r - t).max(axis=1) / np.abs(r).max(axis=1)
agree = (r.argmax(1) == t.argmax(1))
rank = np.array([(t[i] > t[i, r[i].argmax()]).sum() for i in range(n)])
print(f"positions compared: {n}")
print(f"top-1 agreement: {agree.sum()}/{n} ({100 * agree.mean():.1f}%)   reference top-1 is within the test top-3: {(rank < 3).sum()}/{n}")
print(f"KL(ref||test) per position: mean {kl.mean():.5f}  median {np.median(kl):.5f}  max {kl.max():.5f}  (nats)")
print(f"max |logit error| / max |logit|: mean {rel.mean():.4f}  max {rel.max():.4f}")
