"""Task benchmark (Husky-style): Dara vs llama.cpp on real prompts, per task category.

Each task in bench/tasks/tasks.json is wrapped in the model's chat template (identical text for both runtimes), then generated greedily up to --max-new
tokens, stopping at end-of-sequence. Reported per task: prompt tokens, generated tokens, prompt time (~ time to first token), decode tokens/s, and the
Dara / llama.cpp decode ratio; then the geometric mean per category. llama.cpp runs as llama-completion (plain decoding, no speculation); its timings
come from its own perf printout, Dara's from its 'prefill' / 'warm decode' lines. The two runtimes alternate task by task.

Usage: python -I bench/task_bench.py --models 4b nano [9b] [--max-new 256] [--dara-args "..."] [--dara-exe PATH] [--llama-dir DIR]
                                     [--models-dir DIR] [--llama-ngl 9b=24] [--out build/bench/tasks]
See docs/BENCHMARKING.md.
"""
import argparse, json, math, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = None   # parsed arguments (paths and llama.cpp layer counts: bench/benchenv.py)


def chat_text(family, user):
    if family == "qwen":   # Qwen3.5 chat template, thinking disabled (no BOS: the tokenizer adds none)
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<|turn>user\n{user}<turn|>\n<|turn>model\n"   # Gemma 4 (both runtimes prepend <bos>)


def run(cmd, env=None, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout + "\n" + p.stderr


def dara(model, prompt_file, max_new, out, extra_env, extra_args=()):
    cmd = [A.dara_exe, "--gguf", benchenv.gguf(A, model), "--bench", "--special", "--stop-at-eos", "--prompt-file", prompt_file, "--new", str(max_new),
           "--kv", "4096", "--chunk", "512", "--out", out, *extra_args]
    rc, o = run(cmd, env=dict(os.environ, **extra_env))
    if rc != 0:
        return {"error": o[-400:]}
    r = {}
    m1 = re.search(r"prompt text \(\d+ bytes\) -> (\d+) tokens", o)
    m2 = re.search(r"prefill (\d+) tokens: ([0-9.]+) s", o)
    m3 = re.search(r"warm decode: (\d+) steps, [0-9.]+ s/token = ([0-9.]+) tok/s(?: \(wall ([0-9.]+) tok/s\))?", o)
    m4 = re.search(r"decode wall clock: (\d+) tokens in [0-9.]+ s = ([0-9.]+) tok/s", o)
    if m2: r["prompt_tokens"], r["prompt_s"] = int(m2.group(1)), float(m2.group(2))
    elif m1: r["prompt_tokens"] = int(m1.group(1))
    if m4:   # wall clock over all generated tokens (also covers speculative decoding, where one step adds several tokens)
        r["gen_tokens"] = int(m4.group(1)) + 1
        r["decode_tps"] = float(m4.group(2))
    elif m3:
        r["gen_tokens"] = int(m3.group(1)) + 1
        r["decode_tps"] = float(m3.group(3) or m3.group(2))   # wall clock when reported
    try:
        r["text"] = open(out + ".text.txt", encoding="utf-8", errors="replace").read()
    except OSError:
        pass
    return r


def llama(model, prompt_file, max_new):
    cmd = [os.path.join(A.llama_dir, "llama-completion.exe"), "-m", benchenv.gguf(A, model), "-f", prompt_file, "-n", str(max_new), "--temp", "0",
           *benchenv.ngl_args(A, model), "-fa", "on", "--no-display-prompt",
           "-c", "4096", "--no-warmup", "-no-cnv"]   # -no-cnv: llama-completion would otherwise enter interactive chat mode for models with a template
    rc, o = run(cmd)
    if rc != 0:
        return {"error": o[-400:]}
    r = {}
    m1 = re.search(r"prompt eval time =\s*([0-9.]+) ms /\s*(\d+) tokens", o)
    m2 = re.search(r"\n[^\n]*?\beval time =\s*([0-9.]+) ms /\s*(\d+) runs", o)
    if m1: r["prompt_s"], r["prompt_tokens"] = float(m1.group(1)) / 1000, int(m1.group(2))
    if m2:
        ms, n = float(m2.group(1)), int(m2.group(2))
        r["gen_tokens"] = n + 1
        r["decode_tps"] = n / (ms / 1000) if ms > 0 else None
    return r


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4b", "nano"])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--dara-env", nargs="*", default=[], help="KEY=VALUE pairs for Dara runs")
    ap.add_argument("--dara-args", default="", help="extra Dara command-line arguments, e.g. \"--spec 7\"")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "tasks"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    global A
    A = a
    benchenv.check_paths(a, a.models)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    os.makedirs(a.out, exist_ok=True)
    extra_env = dict(kv.split("=", 1) for kv in a.dara_env)
    tasks = json.load(open(os.path.join(ROOT, "bench", "tasks", "tasks.json"), encoding="utf-8"))["tasks"]
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks]
    extra_args = a.dara_args.split()
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine, "max_new": a.max_new, "dara_env": extra_env, "dara_args": extra_args,
              "llama_ngl": {m: benchenv.ngl_for(a, m) for m in a.models}, "results": {}}
    for model in a.models:
        fam = benchenv.FAMILY[model]
        print(f"\n=== {model} ===")
        print(f"{'task':22s} {'cat':9s} {'prompt':>6s} | {'llama tok/s':>11s} {'dara tok/s':>10s} {'ratio':>6s} | {'llama ttft':>10s} {'dara ttft':>9s} | gen L/D")
        rows = []
        for i, t in enumerate(tasks):
            pf = os.path.join(a.out, f"{model}_{t['id']}.prompt.txt")
            with open(pf, "w", encoding="utf-8", newline="") as f:
                f.write(chat_text(fam, t["prompt"]))
            order = ["llama", "dara"] if i % 2 == 0 else ["dara", "llama"]
            res = {}
            for who in order:
                res[who] = llama(model, pf, a.max_new) if who == "llama" else dara(model, pf, a.max_new, os.path.join(a.out, f"{model}_{t['id']}"), extra_env, extra_args)
            L, D = res["llama"], res["dara"]
            ratio = D["decode_tps"] / L["decode_tps"] if D.get("decode_tps") and L.get("decode_tps") else None
            rows.append({"task": t["id"], "category": t["category"], "llama": {k: v for k, v in L.items() if k != "text"}, "dara": {k: v for k, v in D.items() if k != "text"},
                         "decode_ratio": ratio})
            f = lambda x, s="{:.1f}": (s.format(x) if isinstance(x, (int, float)) else "-")
            print(f"{t['id']:22s} {t['category']:9s} {f(L.get('prompt_tokens'), '{}'):>6s} | {f(L.get('decode_tps')):>11s} {f(D.get('decode_tps')):>10s} "
                  f"{f(ratio, '{:.2f}x'):>6s} | {f(L.get('prompt_s'), '{:.2f}s'):>10s} {f(D.get('prompt_s'), '{:.2f}s'):>9s} | {L.get('gen_tokens', '-')}/{D.get('gen_tokens', '-')}"
                  + (f"  ERR {(L.get('error') or D.get('error'))[:120]!r}" if L.get('error') or D.get('error') else ""), flush=True)
        for cat in ("edit", "transform", "generate"):
            g = gmean([r["decode_ratio"] for r in rows if r["category"] == cat])
            if g: print(f"  geometric-mean decode ratio, {cat:9s}: {g:.2f}x")
        g = gmean([r["decode_ratio"] for r in rows])
        if g: print(f"  geometric-mean decode ratio, all tasks : {g:.2f}x")
        report["results"][model] = rows
    with open(os.path.join(a.out, "task_bench.json"), "w", encoding="utf-8") as fo:
        json.dump(report, fo, indent=1)


if __name__ == "__main__":
    main()
