"""Controlled benchmark: Dara vs llama.cpp on the SAME GGUF files, on this machine.

Lives in bench/ (outside the hashed release tree: it measures releases, it is not part of one).

What is compared (definitions chosen so both sides do the same work):
  prefill  : llama-bench "pp512" (512-token prompt, tokens/s)  vs  Dara prefill of a 512-token prompt, last-position logits only
  decode   : llama-bench "tg128" (128 generated tokens from an empty context, tokens/s)  vs  Dara greedy decode of 128 tokens after a short prompt
Dara decode is reported two ways: ENGINE time (the forward passes only: what llama-bench's tg measures, since it does no sampling) and
WALL time (includes Dara's host-side argmax over the vocabulary on every token).

Controls: the GPU must be idle before every run (no other process may hold VRAM); the two runtimes are alternated and the order is shuffled
per repetition so thermal / power drift hits both equally; a global warm-up precedes measurement; GPU clocks, temperature, power and
throttle reasons are recorded around every run; llama.cpp is given its best configuration from a small sweep (flash attention on/off, and
for the 9B several GPU-layer counts), and every configuration tried is reported, not only the winner.

Known, reported differences (not hidden): llama.cpp uses an F16 KV cache, Dara FP32; the llama.cpp build is the user's prebuilt b12189 CUDA
build, which prints that it lacks tensor-core / Turing-specific tuning flags; llama-bench decodes fixed tokens while Dara samples greedily.

Usage: python -I bench/bench_vs_llamacpp.py --models nano 4b 9b --reps 5 --out build/bench
"""
import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = r"D:\oprel\bin\llama_cpp\b12189\cuda\llama-bench.exe"
DARA_BIN = os.path.join(ROOT, "build", "cuda")
DARA_OPTS = {"int8": "0", "kv": "f32", "chunk": 0}   # set from the command line; recorded in the report
MODELS = {
    "nano": dict(gguf=os.path.join(ROOT, "models", "gemma4-e2b-gguf", "gemma-4-E2B-it-Q4_K_M.gguf"), exe="nano_generate.exe", family="gemma",
                 ngl=[99], label="Nano (Gemma 4 E2B Q4_K_M)"),
    "4b": dict(gguf=os.path.join(ROOT, "models", "qwen3.5-4b-gguf", "Qwen3.5-4B-Q4_K_M.gguf"), exe="qwen_generate.exe", family="qwen",
               ngl=[99], label="Qwen3.5-4B Q4_K_M"),
    "9b": dict(gguf=os.path.join(ROOT, "models", "qwen3.5-9b-gguf", "Qwen3.5-9B-Q4_K_M.gguf"), exe="qwen_generate.exe", family="qwen",
               ngl=[22, 24, 20], label="Qwen3.5-9B Q4_K_M"),
}


def sh(cmd, timeout=3600):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def gpu_state():
    rc, out, _ = sh(["nvidia-smi", "--query-gpu=memory.used,memory.free,clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
                     "--format=csv,noheader,nounits"], 30)
    f = [x.strip() for x in out.strip().split(",")]
    keys = ["mem_used_mib", "mem_free_mib", "sm_mhz", "sm_max_mhz", "temp_c", "power_w", "throttle"]
    return dict(zip(keys, f)) if len(f) == len(keys) else {"raw": out.strip()}


def on_battery():
    """True when a laptop runs on battery (Windows Win32_Battery.BatteryStatus 1 = discharging): GPU and CPU are power-capped and timings are invalid."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-WmiObject Win32_Battery).BatteryStatus"], capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0].strip() == "1" if out else False
    except Exception:  # noqa: BLE001
        return False


def wait_idle(max_used_mib=250, timeout=120):
    t0 = time.time()
    while True:
        st = gpu_state()
        try:
            if float(st["mem_used_mib"]) <= max_used_mib:
                return st
        except Exception:  # noqa: BLE001
            pass
        if time.time() - t0 > timeout:
            raise SystemExit(f"GPU is not idle ({st}); close the program using it (e.g. a llama.cpp / oprel backend) and rerun")
        time.sleep(3)


def llama_bench(gguf, ngl, fa, pp=512, tg=128, reps=1):
    cmd = [LLAMA, "-m", gguf, "-p", str(pp), "-n", str(tg), "-ngl", str(ngl), "-fa", str(fa), "-r", str(reps), "-o", "json"]
    rc, out, err = sh(cmd, 1800)
    if rc != 0:
        return None, (err or out)[-400:]
    try:
        data = json.loads(out[out.index("["):])
    except Exception as e:  # noqa: BLE001
        return None, f"unparseable llama-bench output: {e}: {out[-300:]}"
    res = {"build": data[0].get("build_number"), "commit": data[0].get("build_commit")}
    for row in data:
        if row.get("n_prompt", 0) > 0 and row.get("n_gen", 0) == 0:
            res["pp_ts"] = row["avg_ts"]
        if row.get("n_gen", 0) > 0 and row.get("n_prompt", 0) == 0:
            res["tg_ts"] = row["avg_ts"]
    return res, None


def dara_prefill_tokens(family, n=512):
    p = os.path.join(ROOT, "build", "bench", f"prompt512_{family}.tokens.txt")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}: create it first (a {n}-token prompt for the {family} tokenizer)")
    return p


def parse_dara(out):
    d = {}
    m = re.search(r"prefill (\d+) tokens: ([0-9.]+) s", out)
    if m:
        d["prefill_tokens"], d["prefill_s"] = int(m.group(1)), float(m.group(2))
        d["prefill_ts"] = d["prefill_tokens"] / d["prefill_s"]
    m = re.search(r"warm decode: (\d+) steps, [0-9.]+ s/token = ([0-9.]+) tok/s", out)
    if m:
        d["decode_engine_ts"] = float(m.group(2))
    m = re.search(r"decode wall clock: \d+ tokens in [0-9.]+ s = ([0-9.]+) tok/s", out) or re.search(r"\(wall ([0-9.]+) tok/s\)", out)
    if m:
        d["decode_wall_ts"] = float(m.group(1))
    m = re.search(r"cold load: ([0-9.]+) s \| placement: ([^\n]*)", out)
    if m:
        d["cold_load_s"], d["placement"] = float(m.group(1)), m.group(2)[:120]
    return d


def dara_run(model, kind):
    info = MODELS[model]
    exe = os.path.join(DARA_BIN, info["exe"])
    if info["family"] == "qwen":
        common = ["--gguf", info["gguf"], "--bench", "--kv", "1024", "--out", os.path.join(ROOT, "build", "bench", "last")]
        if kind == "prefill":
            cmd = [exe] + common + ["--tokens", dara_prefill_tokens("qwen"), "--new", "1"]
        else:
            cmd = [exe] + common + ["--prompt", "The capital of France is", "--new", "129"]
    else:
        common = ["--gguf", info["gguf"], "--no-layers", "--kv", "1024", "--out", os.path.join(ROOT, "build", "bench", "last")]
        if kind == "prefill":
            cmd = [exe] + common + ["--tokens", dara_prefill_tokens("gemma"), "--new", "1", "--logits-last", "1"]
        else:
            cmd = [exe] + common + ["--bench", "--prompt", "The capital of France is", "--new", "129"]   # --bench: greedy on the GPU, no logits to the host
    cmd += ["--kv-mode", DARA_OPTS["kv"]] + (["--chunk", str(DARA_OPTS["chunk"])] if DARA_OPTS["chunk"] else [])
    env = dict(os.environ, DARA_INT8=DARA_OPTS["int8"])
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800, env=env)
    rc, out, err = p.returncode, p.stdout, p.stderr
    if rc != 0:
        return None, (err or out)[-400:]
    return parse_dara(out), None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": statistics.mean(xs), "sd": statistics.pstdev(xs) if len(xs) > 1 else 0.0, "n": len(xs), "min": min(xs), "max": max(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nano", "4b"], choices=list(MODELS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench"))
    ap.add_argument("--dara-int8", choices=["0", "1"], default="0", help="DARA_INT8: int8 activations (lossy fast mode, quality-gated)")
    ap.add_argument("--dara-kv", choices=["f32", "q8", "tq4"], default="f32")
    ap.add_argument("--dara-chunk", type=int, default=0, help="prefill chunk size (0 = Dara default)")
    a = ap.parse_args()
    DARA_OPTS.update(int8=a.dara_int8, kv=a.dara_kv, chunk=a.dara_chunk)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(20261112)
    report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu_before": gpu_state(), "dara_options": dict(DARA_OPTS), "models": {}}
    print(f"Dara options: {DARA_OPTS}")

    if on_battery():
        raise SystemExit("the machine is running on battery: GPU/CPU are power-capped and the comparison would be invalid. Connect AC power and rerun.")
    print("global warm-up (GPU clocks up, files cached)...")
    wait_idle()
    for m in a.models:
        llama_bench(MODELS[m]["gguf"], MODELS[m]["ngl"][0], 0, pp=64, tg=32, reps=1)
        dara_run(m, "prefill")   # Dara warm-up too (file cache, clocks); not measured

    for m in a.models:
        info = MODELS[m]
        print(f"\n=== {info['label']} ===")
        entry = {"label": info["label"], "llama_configs": {}, "dara": {"runs": []}}
        # 1. llama.cpp configuration sweep (one short run each), then the best config gets the full alternated repetitions
        best = None
        for ngl in info["ngl"]:
            for fa in (0, 1):
                wait_idle()
                res, err = llama_bench(info["gguf"], ngl, fa, reps=2)
                key = f"ngl={ngl},fa={fa}"
                entry["llama_configs"][key] = res if res else {"failed": err}
                print(f"  llama.cpp {key}: " + (f"pp512 {res.get('pp_ts', 0):.1f}  tg128 {res.get('tg_ts', 0):.2f}" if res else f"FAILED ({(err or '')[:80]})"))
                if res and "tg_ts" in res and (best is None or res["tg_ts"] > best[2]["tg_ts"]):
                    best = (ngl, fa, res)
        if best is None:
            print("  llama.cpp: no configuration ran; skipping this model")
            report["models"][m] = entry
            continue
        entry["llama_best"] = {"ngl": best[0], "fa": best[1]}
        print(f"  best llama.cpp configuration by decode speed: ngl={best[0]} fa={best[1]}")

        # 2. alternated repetitions
        lp, lt, dp, de, dw = [], [], [], [], []
        for rep in range(a.reps):
            order = ["llama", "dara"]
            rng.shuffle(order)
            for who in order:
                st = wait_idle()
                t0 = time.time()
                if who == "llama":
                    res, err = llama_bench(info["gguf"], best[0], best[1], reps=1)
                    if res:
                        lp.append(res.get("pp_ts")); lt.append(res.get("tg_ts"))
                    else:
                        print("  llama.cpp run failed:", err)
                else:
                    pre, e1 = dara_run(m, "prefill")
                    dec, e2 = dara_run(m, "decode")
                    if pre:
                        dp.append(pre.get("prefill_ts"))
                    if dec:
                        de.append(dec.get("decode_engine_ts")); dw.append(dec.get("decode_wall_ts"))
                        entry["dara"]["placement"] = dec.get("placement")
                    if not (pre and dec):
                        print("  Dara run failed:", e1 or e2)
                    entry["dara"]["runs"].append({"prefill": pre, "decode": dec})
                entry.setdefault("gpu", []).append({"rep": rep, "who": who, "before": st, "after": gpu_state(), "seconds": round(time.time() - t0, 1)})
            print(f"  rep {rep + 1}/{a.reps}: llama tg {lt[-1] if lt else None}  dara engine {de[-1] if de else None} wall {dw[-1] if dw else None}")
        entry["summary"] = {"llama_pp512": stats(lp), "llama_tg128": stats(lt), "dara_prefill512": stats(dp), "dara_decode_engine": stats(de), "dara_decode_wall": stats(dw)}
        report["models"][m] = entry

    report["gpu_after"] = gpu_state()
    with open(os.path.join(a.out, "bench_results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    # markdown summary
    lines = ["| model | metric | llama.cpp | Dara | Dara / llama.cpp |", "|---|---|---|---|---|"]
    for m, e in report["models"].items():
        s = e.get("summary")
        if not s:
            continue
        def row(metric, lk, dk):
            L, D = s.get(lk), s.get(dk)
            if not L or not D:
                return
            lines.append(f"| {e['label']} | {metric} | {L['mean']:.1f} +- {L['sd']:.1f} | {D['mean']:.1f} +- {D['sd']:.1f} | {D['mean'] / L['mean']:.2f}x |")
        row("prefill 512 tok/s", "llama_pp512", "dara_prefill512")
        row("decode 128 tok/s (Dara engine time)", "llama_tg128", "dara_decode_engine")
        row("decode 128 tok/s (Dara wall, incl. argmax)", "llama_tg128", "dara_decode_wall")
    with open(os.path.join(a.out, "bench_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
