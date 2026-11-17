"""Quick plain-decode check (a few minutes): how close Dara and llama.cpp get to this GPU's memory-bandwidth ceiling.

1. dara.exe --probe measures the GPU's real streaming bandwidth (the ceiling for decode speed).
2. Plain decoding (no Flash, no speculation), 256 tokens, on two fixed prompts, 3 runs each (median), Dara vs llama.cpp.
3. Bytes read per token = the model weights in VRAM (from Dara's placement line); achieved GB/s = bytes x tokens/s, shown as a share of the probe.
4. One profiled Dara run per model (per-operation GPU time), saved for tuning.

Usage: python -I bench/decode_quick.py [--models 4b nano] [--dara-exe PATH] [--llama-dir DIR] [--models-dir DIR] [--out build/bench/decode_quick]
See docs/BENCHMARKING.md.
"""
import argparse, json, os, re, statistics, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402

ROOT = benchenv.ROOT
PROMPTS = {   # (task id in bench/tasks/tasks.json)
    "write_function": "generate",
    "function_edit": "edit",
}
A = None


def run(cmd, env=None, timeout=1800):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout, stdin=subprocess.DEVNULL, cwd=ROOT)
    return p.returncode, p.stdout + "\n" + p.stderr


def chat(model, user):
    if benchenv.FAMILY[model] == "qwen": return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return f"<|turn>user\n{user}<turn|>\n<|turn>model\n"


def dara(model, pf, out, extra_env=None, extra_args=()):
    cmd = [A.dara_exe, "--gguf", benchenv.gguf(A, model), "--bench", "--special", "--prompt-file", pf, "--new", "256", "--kv", "4096", "--out", out, *extra_args]
    if model == "nano": cmd += ["--no-layers"]
    rc, o = run(cmd, env=dict(os.environ, **(extra_env or {})))
    r = {"log": o}
    m = re.search(r"warm decode: (\d+) steps, [0-9.]+ s/token = ([0-9.]+) tok/s(?: \(wall ([0-9.]+) tok/s\))?", o)
    if m: r["tps"] = float(m.group(2))   # GPU-side steady state; wall clock below
    if m and m.group(3): r["wall_tps"] = float(m.group(3))
    mw = re.search(r"decode wall clock: \d+ tokens in [0-9.]+ s = ([0-9.]+) tok/s", o)
    if mw: r["wall_tps"] = float(mw.group(1))
    ml = re.search(r"VRAM-resident layers (\d+) \(([0-9.]+) MiB\)", o)
    mh = re.search(r"output head ([0-9.]+) MiB in VRAM", o)
    if ml: r["weight_mib"] = float(ml.group(2)) + (float(mh.group(1)) if mh else 0.0)
    if rc != 0: r["error"] = o[-400:]
    return r


def llama(model, pf):
    cmd = [os.path.join(A.llama_dir, "llama-completion.exe"), "-m", benchenv.gguf(A, model), "-f", pf, "-n", "256", "--temp", "0", *benchenv.ngl_args(A, model),
           "-fa", "on", "--no-display-prompt", "-c", "4096", "--no-warmup", "-no-cnv", "--ignore-eos"]
    rc, o = run(cmd)
    m = re.search(r"\n[^\n]*?\beval time =\s*([0-9.]+) ms /\s*(\d+) runs", o)
    if rc != 0 or not m: return {"error": o[-400:]}
    ms, n = float(m.group(1)), int(m.group(2))
    return {"tps": n / (ms / 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4b", "nano"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "decode_quick"))
    ap.add_argument("--flash-dir", default=None, help="folder with drafter_4b_v2.dflash (default: <dara.exe folder>\\flash, else build\\flash)")
    benchenv.add_args(ap)
    a = ap.parse_args()
    if not a.flash_dir:
        cand = os.path.join(os.path.dirname(os.path.abspath(a.dara_exe)), "flash")
        a.flash_dir = cand if os.path.isdir(cand) else os.path.join(ROOT, "build", "flash")
    global A
    A = a
    benchenv.check_paths(a, a.models)
    os.makedirs(a.out, exist_ok=True)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    rc, po = run([a.dara_exe, "--probe"])
    open(os.path.join(a.out, "probe.txt"), "w", encoding="utf-8").write(po)
    mp = re.search(r"PROBE gpu=\"([^\"]*)\" sm=(\d+) theoretical_gbs=([0-9.]+) stream_gbs=([0-9.]+) q4k_pattern_gbs=([0-9.]+)", po)
    probe = {"gpu": mp.group(1), "sm": mp.group(2), "theoretical_gbs": float(mp.group(3)), "stream_gbs": float(mp.group(4)),
             "q4k_pattern_gbs": float(mp.group(5))} if mp else {"error": po[-300:]}
    print(f"bandwidth probe: {probe}", flush=True)
    ceiling = probe.get("stream_gbs")
    tasks = {t["id"]: t for t in json.load(open(os.path.join(ROOT, "bench", "tasks", "tasks.json"), encoding="utf-8"))["tasks"]}
    res = {"machine": machine, "probe": probe, "results": {}}
    print(f"\n{'model':5s} {'prompt':15s} | {'llama tok/s':>11s} {'dara tok/s':>10s} {'ratio':>6s} | {'MiB/token':>9s} {'dara GB/s':>9s} {'% probe':>7s} {'llama % probe':>13s}")
    for model in a.models:
        rows = []
        for tid in PROMPTS:
            pf = os.path.join(a.out, f"{model}_{tid}.prompt.txt")
            open(pf, "w", encoding="utf-8", newline="").write(chat(model, tasks[tid]["prompt"]))
            D, L = [], []
            for rep in range(a.reps):
                L.append(llama(model, pf))
                D.append(dara(model, pf, os.path.join(a.out, f"{model}_{tid}")))
            dt = statistics.median([d["tps"] for d in D if "tps" in d]) if any("tps" in d for d in D) else None
            lt = statistics.median([l["tps"] for l in L if "tps" in l]) if any("tps" in l for l in L) else None
            mib = next((d["weight_mib"] for d in D if "weight_mib" in d), None)
            gbs = mib * 1048576 * dt / 1e9 if (mib and dt) else None
            lgbs = mib * 1048576 * lt / 1e9 if (mib and lt) else None
            row = {"prompt": tid, "dara_tps": dt, "llama_tps": lt, "ratio": dt / lt if (dt and lt) else None, "weight_mib": mib, "dara_gbs": gbs, "llama_gbs": lgbs,
                   "dara_pct_probe": 100 * gbs / ceiling if (gbs and ceiling) else None, "llama_pct_probe": 100 * lgbs / ceiling if (lgbs and ceiling) else None,
                   "errors": [x["error"][-200:] for x in D + L if "error" in x]}
            rows.append(row)
            f = lambda x, s="{:.1f}": s.format(x) if isinstance(x, (int, float)) else "-"
            print(f"{model:5s} {tid:15s} | {f(lt):>11s} {f(dt):>10s} {f(row['ratio'], '{:.2f}x'):>6s} | {f(mib, '{:.0f}'):>9s} {f(gbs):>9s} {f(row['dara_pct_probe'], '{:.0f}%'):>7s} "
                  f"{f(row['llama_pct_probe'], '{:.0f}%'):>13s}" + (f"  ERR {row['errors'][0]!r}" if row["errors"] else ""), flush=True)
        # one profiled run: per-operation GPU time of the decode step
        pf = os.path.join(a.out, f"{model}_write_function.prompt.txt")
        prof = dara(model, pf, os.path.join(a.out, f"{model}_profile"), extra_env={"DARA_LAYER_PROF": "1"}, extra_args=(["--profile"] if model == "nano" else []))
        open(os.path.join(a.out, f"{model}_profile.txt"), "w", encoding="utf-8").write(prof["log"])
        res["results"][model] = rows
        # A/B of the planar Q6_K output head (4B / 9B): plain decode above uses the default (planar); here Flash with planar on and off
        drafter = os.path.join(a.flash_dir, "drafter_4b_v2.dflash")
        if model == "4b" and os.path.exists(drafter):
            pf = os.path.join(a.out, "4b_function_edit.prompt.txt")
            ab = {}
            for planar in ("1", "0"):
                runs = [dara(model, pf, os.path.join(a.out, f"4b_flash_planar{planar}"), extra_env={"DARA_Q6_PLANAR": planar},
                             extra_args=("--flash", drafter, "--spec", "4", "--flash-min-p", "0.6")) for _ in range(a.reps)]
                vals = [r["wall_tps"] for r in runs if "wall_tps" in r]
                pl = [dara(model, os.path.join(a.out, "4b_write_function.prompt.txt"), os.path.join(a.out, f"4b_plain_planar{planar}"),
                           extra_env={"DARA_Q6_PLANAR": planar}) for _ in range(a.reps)]
                pv = [r["tps"] for r in pl if "tps" in r]
                ab[f"planar={planar}"] = {"flash_tps": statistics.median(vals) if vals else None, "plain_tps": statistics.median(pv) if pv else None}
            res["planar_head_ab"] = ab
            print(f"4b output head A/B (planar Q6_K on / off): {ab}", flush=True)
    json.dump(res, open(os.path.join(a.out, "decode_quick.json"), "w"), indent=1)
    print(f"\nresults: {a.out}  (send this folder back: decode_quick.json, probe.txt, *_profile.txt)")


if __name__ == "__main__":
    main()
