"""Decode speed and time to first token against context length (Nano by default), Flash on vs off, through `dara serve`.

For each context length the prompt is real text from this repository (code and docs, so lookup cannot just copy a repeated pattern) cut to the length,
a unique first line (so no earlier request's prompt cache helps), and an instruction to write an essay. Each level sends two cold requests:
Flash on (the given drafter) and Flash off (plain). Reported per request, from the server's own timings: prompt tokens, time to first token,
prefill tok/s, decode tok/s, and the speculative decoder's state (mode, steps, accepted/drafted, and the note when it stands itself down).

Usage: python -I bench/context_sweep.py [--model nano] [--drafter build/flash/drafter_nano_v3.dflash]
           [--levels 0 4096 8192 16384 32768 65536 81920 102400 128000 131000] [--max-new 256] [--cold-plain-max N]
"""
import argparse, glob, json, os, statistics, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402

ROOT = benchenv.ROOT


def http(url, path, body=None, timeout=3600):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def corpus():
    files = []
    for pat in ("docs/*.md", "src/**/*.cu", "src/**/*.cpp", "src/**/*.h", "tools/*.cu", "bench/*.py", "tests/*.py"):
        files += sorted(glob.glob(os.path.join(ROOT, pat), recursive=True))
    parts, seen = [], set()
    for f in files:
        if f in seen or "context_sweep" in f:
            continue
        seen.add(f)
        try:
            parts.append(f"// ===== {os.path.relpath(f, ROOT)} =====\n" + open(f, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nano")
    ap.add_argument("--drafter", default="build/flash/drafter_nano_v3.dflash")
    ap.add_argument("--levels", nargs="+", type=int, default=[0, 4096, 8192, 16384, 32768, 65536, 81920, 102400, 128000, 131000])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--cold-plain-max", type=int, default=10**9, help="above this prompt length the plain request reuses the Flash request's prompt cache (decode speed only)")
    ap.add_argument("--port", type=int, default=18091)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "context"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    benchenv.check_paths(a, [a.model], need_llama=False)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    os.makedirs(a.out, exist_ok=True)
    drafter = a.drafter if os.path.isabs(a.drafter) else os.path.join(ROOT, a.drafter)
    url = f"http://127.0.0.1:{a.port}"
    log = open(os.path.join(a.out, "server.log"), "w")
    proc = subprocess.Popen([a.dara_exe, "serve", "--model", benchenv.gguf(a, a.model), "--port", str(a.port), "--no-store", "--ctx", str(a.ctx), "--flash", drafter],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    results = []
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise SystemExit("server exited early: see " + os.path.join(a.out, "server.log"))
            try:
                h = http(url, "/health", timeout=3)
                if h.get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        cfg = http(url, "/dara/load").get("config")
        print("server config:", json.dumps(cfg), flush=True)
        text = corpus()
        tok_total = http(url, "/dara/tokenize", {"text": text})
        n_tok = tok_total.get("tokens") or tok_total.get("count") or tok_total.get("n_tokens") or list(tok_total.values())[0]
        ratio = len(text) / float(n_tok)
        print(f"corpus: {len(text)} chars, {n_tok} tokens ({ratio:.2f} chars/token)", flush=True)
        instruction = "\n\n---\nWrite a detailed essay of about 300 words about the software described above: what it does and how it is built."
        def count(content):
            r = http(url, "/dara/tokenize", {"messages": [{"role": "user", "content": content}]})
            return r.get("tokens") or r.get("count") or r.get("n_tokens") or list(r.values())[0]

        def chars_for(L):
            """The text length whose whole prompt (chat template + nonce + text + instruction) is as close to L tokens as possible, from below."""
            if L <= 0:
                return 0
            lo, hi = 0, min(len(text), int(L * ratio * 1.4))
            while hi - lo > 400:
                mid = (lo + hi) // 2
                if count("[run flash-000000-0000000000.000]\n" + text[:mid] + instruction) <= L: lo = mid
                else: hi = mid
            return lo

        for L in a.levels:
            body_chars = chars_for(L)
            if L > 0 and body_chars >= len(text) - 400:
                print(f"level {L}: corpus too short"); continue
            max_new = max(16, min(a.max_new, a.ctx - L - 24))
            row = {"level": L}
            plain_first_prompt = None
            for mode in ("flash", "plain"):
                nonce = f"[run {mode}-{L}-{time.time():.3f}]\n"
                if mode == "plain" and L > a.cold_plain_max and plain_first_prompt is not None:
                    content = plain_first_prompt   # same prompt as the Flash request: its prefix is cached, so only decode speed is meaningful
                    cold = False
                else:
                    content = nonce + (text[:body_chars] if L > 0 else "") + instruction
                    cold = True
                    if mode == "flash":
                        plain_first_prompt = content
                tstart = time.time()
                try:
                    r = http(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": max_new, "temperature": 0, "flash": mode == "flash"})
                except Exception as e:
                    row[mode] = {"error": str(e)[:200]}
                    print(f"level {L:>6} {mode:5s} ERROR {e}", flush=True)
                    continue
                t, u, sp = r.get("timings", {}), r.get("usage", {}), r.get("speculative", {})
                row[mode] = {"prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "completion_tokens": u.get("completion_tokens"),
                             "ttft_ms": t.get("ttft_ms"), "prefill_tps": t.get("prefill_tokens_per_second"), "decode_tps": t.get("decode_tokens_per_second"),
                             "cold": cold, "spec": sp, "wall_s": round(time.time() - tstart, 1)}
                d = row[mode]
                print(f"level {L:>6} {mode:5s} prompt {d['prompt_tokens']} cached {d['cached_tokens']} +{d['completion_tokens']} | ttft {d['ttft_ms']} ms | prefill {d['prefill_tps']} tok/s | "
                      f"decode {d['decode_tps']} tok/s | spec {sp.get('mode')} steps {sp.get('verify_steps')} acc {sp.get('accepted')}/{sp.get('drafted')} {sp.get('note') or ''}", flush=True)
            results.append(row)
            json.dump({"machine": machine, "config": cfg, "results": results}, open(os.path.join(a.out, "context_sweep.json"), "w"), indent=1)
    finally:
        proc.terminate()
        try: proc.wait(timeout=30)
        except Exception: proc.kill()
        log.close()
    print("saved", os.path.join(a.out, "context_sweep.json"))


if __name__ == "__main__":
    main()
