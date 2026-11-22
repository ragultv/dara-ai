"""Husky-style benchmark (what husky.underdog.ai measures), Dara vs llama.cpp, controlled.

Husky's method: 16 prompts from the app's own tasks, greedy decoding, MEDIAN of 3 runs per engine, engines alternated, an idle machine, a canary
that the faster engine's output equals the baseline's. This script does the same for:
  llama      llama.cpp plain decoding (llama-completion)
  plain      Dara, no speculation
  flash_v2 / flash_v3 / ...   Dara with a Flash drafter (prompt lookup + the trained drafter), one entry per drafter given
and reports, per task: decode tokens/s, time to first token (prompt processing), the ratio to llama.cpp; geometric means per category;
cold prefill tokens/s on a ~950-token prompt; and the canary (does each Dara mode write the same text as Dara plain?).

Usage: python -I bench/husky_bench.py --models nano 4b [--reps 3] [--max-new 256] [--allow-battery]
           [--drafters nano:v2=build/flash/drafter_nano_v2.dflash nano:v3=build/flash/drafter_nano_v3.dflash 4b:v2=build/flash/drafter_4b_v2.dflash]
"""
import argparse, json, math, os, statistics, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402
import task_bench as tb  # noqa: E402

ROOT = benchenv.ROOT
FLASH_ARGS = ["--spec", "4", "--flash-min-p", "0.6"]
DEFAULT_DRAFTERS = ["nano:v2=build/flash/drafter_nano_v2.dflash", "nano:v3=build/flash/drafter_nano_v3.dflash", "4b:v2=build/flash/drafter_4b_v2.dflash"]


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def gmean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def f(x, s="{:.1f}"):
    return s.format(x) if isinstance(x, (int, float)) else "-"


def run_cfg(model, name, extra, pf, max_new, out):
    if name == "llama":
        return tb.llama(model, pf, max_new)
    return tb.dara(model, pf, max_new, out, {}, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nano", "4b"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--drafters", nargs="*", default=DEFAULT_DRAFTERS, help="model:label=path pairs")
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task ids (smoke test)")
    ap.add_argument("--allow-battery", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "husky"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    tb.A = a
    benchenv.check_paths(a, a.models)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    if str(machine.get("power", "")).startswith("BATTERY") and not a.allow_battery:
        raise SystemExit("laptop is on battery: timings would be 2-5x off. Plug in, or pass --allow-battery.")
    out_dir = os.path.join(a.out, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    tasks = json.load(open(os.path.join(ROOT, "bench", "tasks", "tasks.json"), encoding="utf-8"))["tasks"]
    all_tasks = tasks
    if a.tasks:
        tasks = [t for t in tasks if t["id"] in a.tasks]
    report ={"started": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine, "reps": a.reps, "max_new": a.max_new, "models": {}}
    md = [f"# Husky-style benchmark ({report['started']})", "", f"Machine: {json.dumps(machine)}", f"Median of {a.reps} runs, engines alternated, greedy, max {a.max_new} new tokens.", ""]

    for model in a.models:
        fam = benchenv.FAMILY[model]
        cfgs = [("llama", None), ("plain", [])]
        for d in a.drafters:
            key, path = d.split("=", 1)
            m, label = key.split(":")
            if m == model:
                p = path if os.path.isabs(path) else os.path.join(ROOT, path)
                if os.path.exists(p):
                    cfgs.append((f"flash_{label}", [*FLASH_ARGS, "--flash", p]))
                else:
                    print(f"skip drafter {label}: {p} not found")
        names = [c[0] for c in cfgs]
        extra = dict(cfgs)
        print(f"\n=== {model}: {', '.join(names)} ===", flush=True)
        samples = {t["id"]: {n: [] for n in names} for t in tasks}
        texts = {t["id"]: {} for t in tasks}
        for rep in range(a.reps):
            for i, t in enumerate(tasks):
                pf = os.path.join(out_dir, f"{model}_{t['id']}.prompt.txt")
                if rep == 0:
                    with open(pf, "w", encoding="utf-8", newline="") as fh:
                        fh.write(tb.chat_text(fam, t["prompt"]))
                k = (rep + i) % len(names)
                for n in names[k:] + names[:k]:   # rotate the order every task and rep
                    r = run_cfg(model, n, extra[n], pf, a.max_new, os.path.join(out_dir, f"{model}_{t['id']}_{n}"))
                    samples[t["id"]][n].append(r)
                    if rep == 0 and n != "llama" and "text" in r:
                        texts[t["id"]][n] = r["text"]
            print(f"  rep {rep + 1}/{a.reps} done", flush=True)
        rows = []
        for t in tasks:
            row = {"task": t["id"], "category": t["category"], "prompt_tokens": None, "cfg": {}}
            for n in names:
                rs = samples[t["id"]][n]
                row["cfg"][n] = {"decode_tps": med([r.get("decode_tps") for r in rs]), "ttft_s": med([r.get("prompt_s") for r in rs]),
                                 "gen_tokens": med([r.get("gen_tokens") for r in rs]), "errors": [r["error"][-120:] for r in rs if "error" in r]}
                row["prompt_tokens"] = row["prompt_tokens"] or next((r.get("prompt_tokens") for r in rs if r.get("prompt_tokens")), None)
            base = row["cfg"]["llama"]["decode_tps"]
            for n in names:
                c = row["cfg"][n]
                c["ratio_vs_llama"] = c["decode_tps"] / base if (c["decode_tps"] and base) else None
            row["canary"] = {n: (texts[t["id"]].get(n) == texts[t["id"]].get("plain")) for n in names if n not in ("llama", "plain")}
            rows.append(row)
        # ---- table
        head = f"{'task':20s} {'cat':9s} {'in':>4s} | " + " ".join(f"{n:>10s}" for n in names) + " | " + " ".join(f"{n[:9]+' x':>10s}" for n in names[1:]) + " | ttft llama/plain"
        lines = [head]
        for r in rows:
            lines.append(f"{r['task']:20s} {r['category']:9s} {f(r['prompt_tokens'], '{}'):>4s} | " + " ".join(f"{f(r['cfg'][n]['decode_tps']):>10s}" for n in names) + " | "
                        + " ".join(f"{f(r['cfg'][n]['ratio_vs_llama'], '{:.2f}x'):>10s}" for n in names[1:]) + f" | {f(r['cfg']['llama']['ttft_s'], '{:.2f}')}/{f(r['cfg']['plain']['ttft_s'], '{:.2f}')}")
        summ = []
        for cat in ("edit", "transform", "generate", None):
            sel = [r for r in rows if cat is None or r["category"] == cat]
            summ.append(f"{(cat or 'ALL'):9s} " + "  ".join(f"{n}: {f(gmean([r['cfg'][n]['ratio_vs_llama'] for r in sel]), '{:.2f}x')}" for n in names[1:]))
        canary = ["canary (same text as Dara plain, rep 1): " + "  ".join(f"{n}: {sum(r['canary'][n] for r in rows)}/{len(rows)}" for n in names if n not in ("llama", "plain"))]
        for r in rows:
            bad = [n for n, ok in r["canary"].items() if not ok]
            if bad:
                canary.append(f"   differs from plain: {r['task']} ({', '.join(bad)})")
        # ---- cold prefill
        big = os.path.join(out_dir, f"{model}_prefill.prompt.txt")
        with open(big, "w", encoding="utf-8", newline="") as fh:
            fh.write(tb.chat_text(fam, "\n\n".join(t["prompt"] for t in all_tasks[:3])))
        pre = {n: [] for n in ("llama", "plain")}
        ptok = None
        for rep in range(max(3, a.reps)):
            for n in (["llama", "plain"] if rep % 2 == 0 else ["plain", "llama"]):
                r = run_cfg(model, n, [], big, 2, os.path.join(out_dir, f"{model}_prefill_{n}"))
                if r.get("prompt_s") and r.get("prompt_tokens"):
                    pre[n].append(r["prompt_tokens"] / r["prompt_s"]); ptok = r["prompt_tokens"]
        pf_l, pf_d = med(pre["llama"]), med(pre["plain"])
        prefill = f"cold prefill, {ptok} tokens: llama.cpp {f(pf_l, '{:.0f}')} tok/s, Dara {f(pf_d, '{:.0f}')} tok/s ({f(pf_d / pf_l if pf_d and pf_l else None, '{:.2f}x')})"
        for ln in lines + summ + canary + [prefill]:
            print(ln)
        md += [f"## {model}", "", "```", *lines, "", "geometric-mean decode ratio vs llama.cpp:", *summ, "", *canary, "", prefill, "```", ""]
        report["models"][model] = {"rows": rows, "prefill_tps": {"llama": pf_l, "dara": pf_d, "tokens": ptok}}
        json.dump(report, open(os.path.join(out_dir, "husky_bench.json"), "w"), indent=1)
        open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8").write("\n".join(md))
    print(f"\nresults: {out_dir}")


if __name__ == "__main__":
    main()
