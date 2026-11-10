"""Quality gate for Dara's lossy fast paths (int8 activations, TurboQuant 4-bit KV) against the FP32 reference path.

For each model and workload:
  1. reference run: FP32 activations, FP32 KV, greedy, N new tokens (its logits and tokens are the reference)
  2. teacher-forced test runs (--teacher <reference tokens>): every test step sees exactly the reference context, so one near-tie flip cannot derail the rest
     of the comparison. Variants: int8 + FP32 KV, FP32 + TQ4 KV, int8 + TQ4 KV.
Per variant: top-1 agreement with the reference's greedy token, KL(ref || test) mean / p99 / max over the generated positions, and how many disagreements are
near-ties (reference top-2 margin < 0.5 logits).

Usage: python -I bench/quality_gate.py [--models 4b nano] [--new 128]
"""
import argparse, json, os, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "cuda")
Q = os.path.join(ROOT, "build", "bench", "quality")
MODELS = {
    "4b": dict(exe="qwen_generate.exe", gguf=os.path.join(ROOT, "models", "qwen3.5-4b-gguf", "Qwen3.5-4B-Q4_K_M.gguf"), vocab=248320, extra=[],
               long=os.path.join(ROOT, "build", "bench", "prompt1500_qwen.tokens.txt")),
    "nano": dict(exe="nano_generate.exe", gguf=os.path.join(ROOT, "models", "gemma4-e2b-gguf", "gemma-4-E2B-it-Q4_K_M.gguf"), vocab=262144, extra=["--no-layers"],
                 long=os.path.join(ROOT, "build", "bench", "prompt1500_gemma.tokens.txt")),
}
WORKLOADS = ["english", "tamil", "code", "structured", "long1500"]   # also available: long8k (build/bench/quality/long8k.txt, ~8K tokens)
ALL_VARIANTS = {"int8_f32kv": ("1", "f32"), "fp32_q8kv": ("0", "q8"), "int8_q8kv": ("1", "q8"), "int8_k16v8kv": ("1", "k16v8"), "fp32_tq4kv": ("0", "tq4"), "int8_tq4kv": ("1", "tq4")}


def run(model, workload, tag, int8, kv_mode, n_new, teacher=None):
    m = MODELS[model]
    out = os.path.join(Q, f"{model}_{workload}_{tag}")
    src = ["--tokens", m["long"]] if workload == "long1500" else ["--prompt-file", os.path.join(Q, workload + ".txt")]
    cmd = [os.path.join(BIN, m["exe"]), "--gguf", m["gguf"], *m["extra"], *src, "--new", str(n_new), "--kv", "12288" if workload.startswith("long8k") else "2048", "--kv-mode", kv_mode,
           "--logits-last", "1", "--out", out]
    if teacher: cmd += ["--teacher", teacher]
    env = dict(os.environ, DARA_INT8=int8)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=3600)
    if p.returncode != 0:
        raise RuntimeError(f"{model} {workload} {tag} failed:\n{(p.stdout + p.stderr)[-1500:]}")
    return out


def logsoftmax(x):
    mx = x.max(axis=1, keepdims=True)
    return x - mx - np.log(np.exp(x - mx).sum(axis=1, keepdims=True))


def compare(ref, test, V):
    r = np.fromfile(ref + ".logits.bin", dtype=np.float32).reshape(-1, V).astype(np.float64)
    t = np.fromfile(test + ".logits.bin", dtype=np.float32).reshape(-1, V).astype(np.float64)
    n = min(len(r), len(t))
    r, t = r[:n], t[:n]
    lr, lt = logsoftmax(r), logsoftmax(t)
    kl = (np.exp(lr) * (lr - lt)).sum(axis=1)
    ra, ta = r.argmax(1), t.argmax(1)
    srt = np.sort(r, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    dis = ra != ta
    return dict(positions=int(n), top1=int((~dis).sum()), near_tie_flips=int((dis & (margin < 0.5)).sum()),
                kl_mean=float(kl.mean()), kl_p99=float(np.percentile(kl, 99)), kl_max=float(kl.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4b", "nano"])
    ap.add_argument("--workloads", nargs="+", default=WORKLOADS)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS))
    a = ap.parse_args()
    results = {}
    for model in a.models:
        V = MODELS[model]["vocab"]
        for w in a.workloads:
            ref = run(model, w, "ref", "0", "f32", a.new)
            for tag in a.variants:
                int8, kv = ALL_VARIANTS[tag]
                test = run(model, w, tag, int8, kv, a.new, teacher=ref + ".tokens.txt")
                res = compare(ref, test, V)
                results[f"{model}/{w}/{tag}"] = res
                print(f"{model:4s} {w:10s} {tag:11s} top-1 {res['top1']:3d}/{res['positions']} (near-tie flips {res['near_tie_flips']})  "
                      f"KL mean {res['kl_mean']:.5f} p99 {res['kl_p99']:.5f} max {res['kl_max']:.5f}", flush=True)
                os.remove(test + ".logits.bin")
    with open(os.path.join(Q, "results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
