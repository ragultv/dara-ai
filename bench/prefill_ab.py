"""Before / after benchmark of the long-context attention work, on real models: prefill speed and decode speed at several prompt lengths.

"before" switches the three new paths off with their environment variables (they are the only differences, so it is the same binary and the same model):
    DARA_GQA_ATTN=0            decode attention reads each K/V row once per query head (old) instead of once per group
    DARA_ATTN_GROUP_PREFILL=0  prefill attention runs one GEMM pair per query head (old) instead of one per K/V head
    DARA_ATTN_QK_INT8=0        prefill attention scores in FP32 (old) instead of int8 x int8 with dp4a
"after" = the defaults. Prompts are repository text (code and docs) wrapped in each model's chat template; greedy, 64 new tokens, median of --reps runs,
the two configurations alternated. Only the engine's own timings are used (prefill seconds, warm decode tokens/s).

Usage: python -I bench/prefill_ab.py [--models nano 4b] [--sizes 1 4 16] [--reps 2]     (sizes in thousands of prompt tokens)
"""
import argparse, glob, os, re, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402
import task_bench as tb  # noqa: E402

ROOT = benchenv.ROOT
OLD = {"DARA_GQA_ATTN": "0", "DARA_ATTN_GROUP_PREFILL": "0", "DARA_ATTN_QK_INT8": "0"}
CHARS_PER_TOKEN = {"nano": 3.4, "4b": 3.45}


def corpus():
    files = []
    for pat in ("docs/*.md", "src/**/*.cu", "src/**/*.cpp", "src/**/*.h", "tools/*.cu"):
        files += sorted(glob.glob(os.path.join(ROOT, pat), recursive=True))
    return "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


def run_one(a, model, pf, env_extra, out):
    cmd = [a.dara_exe, "--gguf", benchenv.gguf(a, model), "--bench", "--special", "--prompt-file", pf, "--new", "64", "--kv", str(a.kv), "--chunk", "512", "--out", out]
    if model == "nano":
        cmd += ["--no-layers"]
    else:
        cmd += ["--kv-mode", "q8"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=dict(os.environ, **env_extra), stdin=subprocess.DEVNULL, timeout=3600)
    o = p.stdout + p.stderr
    m = re.search(r"prefill (\d+) tokens: ([0-9.]+) s", o)
    d = re.search(r"warm decode: \d+ steps, [0-9.]+ s/token = ([0-9.]+) tok/s(?: \(wall ([0-9.]+) tok/s\))?", o)
    if p.returncode != 0 or not m or not d:
        return None
    return int(m.group(1)), float(m.group(2)), float(d.group(2) or d.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nano", "4b"])
    ap.add_argument("--sizes", nargs="+", type=float, default=[1, 4, 16])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kv", type=int, default=40000)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "prefill_ab"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    benchenv.check_paths(a, a.models, need_llama=False)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    os.makedirs(a.out, exist_ok=True)
    text = corpus()
    rows = []
    print(f"\n{'model':5s} {'prompt':>7s} | {'prefill tok/s: before':>21s} {'after':>8s} {'x':>6s} | {'decode tok/s: before':>20s} {'after':>8s} {'x':>6s}")
    for model in a.models:
        fam = benchenv.FAMILY[model]
        for size in a.sizes:
            chars = int(size * 1000 * CHARS_PER_TOKEN[model])
            pf = os.path.join(a.out, f"{model}_{int(size)}k.prompt.txt")
            with open(pf, "w", encoding="utf-8", newline="") as f:
                f.write(tb.chat_text(fam, text[:chars] + "\nSummarize."))
            res = {"before": [], "after": []}
            for rep in range(a.reps):
                for cfg in (("before", "after") if rep % 2 == 0 else ("after", "before")):
                    r = run_one(a, model, pf, OLD if cfg == "before" else {}, os.path.join(a.out, f"{model}_{int(size)}k_{cfg}"))
                    if r: res[cfg].append(r)
            if not res["before"] or not res["after"]:
                print(f"{model:5s} {size:6.0f}k | failed")
                continue
            n = res["before"][0][0]
            pb = n / statistics.median(r[1] for r in res["before"]); pa = n / statistics.median(r[1] for r in res["after"])
            db = statistics.median(r[2] for r in res["before"]); da = statistics.median(r[2] for r in res["after"])
            rows.append({"model": model, "tokens": n, "prefill_before": pb, "prefill_after": pa, "decode_before": db, "decode_after": da})
            print(f"{model:5s} {n:7d} | {pb:21.0f} {pa:8.0f} {pa / pb:5.2f}x | {db:20.1f} {da:8.1f} {da / db:5.2f}x", flush=True)
    import json
    json.dump({"machine": machine, "rows": rows}, open(os.path.join(a.out, "prefill_ab.json"), "w"), indent=1)
    # verdict for a GPU these paths were not measured on (the RTX check): nothing may get more than 3% slower
    slower = [r for r in rows if r["prefill_after"] < 0.97 * r["prefill_before"] or r["decode_after"] < 0.97 * r["decode_before"]]
    print(f"\nGPU: {machine.get('gpu')}")
    if not rows:
        print("VERDICT: no results")
    elif slower:
        print("VERDICT: SLOWER than before in " + ", ".join(f"{r['model']} {r['tokens']} tokens" for r in slower) +
              "  -> run with DARA_GQA_ATTN=0 / DARA_ATTN_GROUP_PREFILL=0 / DARA_ATTN_QK_INT8=0 to turn the new paths off, and send this output")
    else:
        print("VERDICT: not slower anywhere (every ratio >= 0.97)")


if __name__ == "__main__":
    main()
