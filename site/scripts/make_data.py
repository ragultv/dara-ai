"""Builds site/src/data/benchmarks.json from the benchmark result files, so every number on the site comes from a recorded run (nothing is typed in by hand).

Reads (newest run of each, whatever exists):
  build/bench/husky_final/<run>/husky_bench.json   16-task matrix: llama.cpp vs Dara plain vs Dara + Flash, cold prefill      (bench/husky_bench.py)
  build/bench/context_gqa/context_sweep.json        decode / prefill / first token against context length, current engine    (bench/context_sweep.py)
  build/bench/context/context_sweep.json            the same, before the grouped-query attention work
  build/bench/prefill_ab/prefill_ab.json            same binary, new attention paths off vs on                                (bench/prefill_ab.py)
  RTX-3050-4GB-bench/                               the second machine (plain and Flash task runs, prefill_ab), pushed by whoever ran docs/RTX_LAPTOP_CHECK.md
Usage (from the repo root): python -I site/scripts/make_data.py
"""
import glob, json, os, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "site", "src", "data", "benchmarks.json")


def newest(pattern):
    files = sorted(glob.glob(os.path.join(ROOT, pattern)), key=os.path.getmtime)
    return files[-1] if files else None


def load(path):
    return json.load(open(path, encoding="utf-8")) if path else None


def r1(x, n=1):
    return None if x is None else round(float(x), n)


def husky(model, rep):
    m = rep["models"].get(model)
    if not m:
        return None
    flash_key = next((k for k in m["rows"][0]["cfg"] if k.startswith("flash_")), None)
    rows = []
    for r in m["rows"]:
        c = r["cfg"]
        rows.append({
            "task": r["task"], "category": r["category"], "promptTokens": r["prompt_tokens"],
            "llama": r1(c["llama"]["decode_tps"]), "plain": r1(c["plain"]["decode_tps"]),
            "flash": r1(c[flash_key]["decode_tps"]) if flash_key else None,
            "ttftLlama": r1(c["llama"]["ttft_s"], 2), "ttftDara": r1(c["plain"]["ttft_s"], 2),
            "sameTextAsPlain": r["canary"].get(flash_key) if flash_key else None,
        })
    return {"flashDrafter": flash_key, "rows": rows, "coldPrefill": {k: r1(v, 0) for k, v in m["prefill_tps"].items()}}


def sweep(path):
    d = load(path)
    if not d:
        return None
    out = []
    for row in d["results"]:
        p, f = row.get("plain", {}), row.get("flash", {})
        out.append({"tokens": f.get("prompt_tokens") or p.get("prompt_tokens") or 0, "plainDecode": r1(p.get("decode_tps")), "flashDecode": r1(f.get("decode_tps")),
                    "coldFirstTokenSeconds": r1((f.get("ttft_ms") or 0) / 1000, 1) if f.get("cold") else None, "prefillTps": r1(f.get("prefill_tps"), 0)})
    return out


def rtx3050():
    """Second machine: bench/task_bench.py runs (plain, Flash on Nano, Flash on Core 4B) and prefill_ab.py, as pushed in RTX-3050-4GB-bench/."""
    base = os.path.join(ROOT, "RTX-3050-4GB-bench")
    plain = load(os.path.join(base, "tasks_plain", "task_bench.json"))
    if not plain:
        return None
    out = {"machine": {k: plain["machine"].get(k) for k in ("gpu", "vram", "driver", "cpu", "power")} | {"ramGb": plain["machine"].get("ram_gb")}, "nano": None, "core4b": None, "attentionAb": None}
    for model, key, flash_file in (("nano", "nano", "tasks_flash_nano"), ("4b", "core4b", "tasks_flash_4b")):
        fl = load(os.path.join(base, flash_file, "task_bench.json"))
        flash = {r["task"]: r for r in (fl or {}).get("results", {}).get(model, [])}
        rows = []
        for r in plain["results"].get(model, []):
            f = flash.get(r["task"])
            rows.append({"task": r["task"], "category": r["category"], "llama": r1(r["llama"]["decode_tps"]), "plain": r1(r["dara"]["decode_tps"]),
                         "flash": r1(f["dara"]["decode_tps"]) if f else None})
        out[key] = {"rows": rows}
    out["attentionAb"] = (load(os.path.join(base, "prefill_ab.json")) or {}).get("rows")
    return out


def main():
    hp = newest("build/bench/husky_final/*/husky_bench.json") or newest("build/bench/husky/*/husky_bench.json")
    rep = load(hp)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    data = {
        "generated": time.strftime("%Y-%m-%d"), "commit": commit, "source": os.path.relpath(hp, ROOT) if hp else None,
        "reps": rep.get("reps") if rep else None,
        "nano": husky("nano", rep) if rep else None,
        "core4b": husky("4b", rep) if rep else None,
        "contextAfter": sweep(os.path.join(ROOT, "build/bench/context_gqa/context_sweep.json")),
        "contextBefore": sweep(os.path.join(ROOT, "build/bench/context/context_sweep.json")),
        "attentionAb": (load(os.path.join(ROOT, "build/bench/prefill_ab/prefill_ab.json")) or {}).get("rows"),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(data, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", os.path.relpath(OUT, ROOT), "| matrix:", data["source"], "| nano" if data["nano"] else "", "| core4b" if data["core4b"] else "(core4b not available yet)")


if __name__ == "__main__":
    main()
