"""Multi-conversation prompt cache: switching between long conversations on `dara serve`, cache on vs off.

Three different ~1500-token conversations A, B, C are interleaved: A1 B1 C1 A2 B2 C2 A3 (each Ni = the conversation so far + a short follow-up).
With the cache OFF (DARA_PROMPT_CACHE_MB=0) every switch prefills the whole conversation again; with it ON the parked K/V of the conversation is copied
back and only the new message is prefilled. For every request this prints prompt tokens, tokens reused from a cache, time to first token, and checks
that greedy answers are IDENTICAL with the cache on and off (the cache must never change a word).

Usage: python -I bench/multi_conv_test.py [--model nano] [--max-new 24]
"""
import argparse, glob, json, os, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402

ROOT = benchenv.ROOT


def http(url, path, body=None, timeout=900):
    req = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def corpus():
    files = []
    for pat in ("docs/*.md", "src/**/*.cu", "src/**/*.cpp", "src/**/*.h", "tools/*.cu"):
        files += sorted(glob.glob(os.path.join(ROOT, pat), recursive=True))
    return "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)


def run_scenario(a, cache_mb, label, order=None):
    port = 18095
    env = dict(os.environ, DARA_PROMPT_CACHE_MB=str(cache_mb))
    log = open(os.path.join(a.out, f"server_{label}.log"), "w")
    proc = subprocess.Popen([a.dara_exe, "serve", "--model", benchenv.gguf(a, a.model), "--port", str(port), "--no-store", "--flash", "off", "--ctx", "16384"],
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
    url = f"http://127.0.0.1:{port}"
    rows, texts = [], []
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise SystemExit("server exited early")
            try:
                if http(url, "/health", timeout=3).get("status") == "ok": break
            except Exception:
                pass
            time.sleep(1)
        text = corpus()
        starts = {"A": 0, "B": 5000, "C": 10000}
        chars = a.chars
        conv = {k: [{"role": "user", "content": f"[document {k}]\n" + text[s:s + chars] + "\n\nWhat is this document about? Answer in one sentence."}] for k, s in starts.items()}
        order = order or ["A", "B", "C", "A", "B", "C", "A"]
        seen = {k: 0 for k in starts}
        # warm-up request (kernel load etc.), not recorded
        http(url, "/v1/chat/completions", {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0})
        for k in order:
            seen[k] += 1
            if seen[k] > 1:
                conv[k].append({"role": "user", "content": f"Follow-up {seen[k]}: name one function mentioned in the document."})
            r = http(url, "/v1/chat/completions", {"messages": conv[k], "max_tokens": a.max_new, "temperature": 0})
            ans = r["choices"][0]["message"]["content"]
            conv[k].append({"role": "assistant", "content": ans})
            u, t = r["usage"], r["timings"]
            rows.append((f"{k}{seen[k]}", u["prompt_tokens"], u["prompt_tokens_details"]["cached_tokens"], t["ttft_ms"]))
            texts.append(ans)
    finally:
        proc.terminate()
        try: proc.wait(timeout=30)
        except Exception: proc.kill()
        log.close()
        time.sleep(3)
    return rows, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nano")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--chars", type=int, default=4300, help="characters of the first message of each conversation (~1500 tokens)")
    ap.add_argument("--consecutive", action="store_true", help="also run A A A back to back with the cache off and compare")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "multi_conv"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    benchenv.check_paths(a, [a.model], need_llama=False)
    os.makedirs(a.out, exist_ok=True)
    benchenv.print_machine(benchenv.machine_info())
    off, off_t = run_scenario(a, 0, "off")
    on, on_t = run_scenario(a, 2048, "on")
    if a.consecutive:   # the same conversation back to back, cache off: the live-checkpoint path, with no parking involved
        cons, cons_t = run_scenario(a, 0, "consecutive", ["A", "A", "A"])
        for i, (j, name) in enumerate(((0, "A1"), (3, "A2"), (6, "A3"))):
            print(f"{name}: consecutive (live checkpoint) == cache ON (parked and restored): {cons_t[i] == on_t[j]};  == cache OFF (from scratch): {cons_t[i] == off_t[j]}")
    print(f"\n{'request':8s} {'prompt':>7s} | {'cache OFF: reused':>17s} {'ttft ms':>9s} | {'cache ON: reused':>16s} {'ttft ms':>9s} | same text")
    for (n, p, co, to), (n2, p2, cn, tn), x, y in zip(off, on, off_t, on_t):
        print(f"{n:8s} {p:7d} | {co:17d} {to:9.0f} | {cn:16d} {tn:9.0f} | {'yes' if x == y else 'NO  <-- differs'}")
    for (n, *_), x, y in zip(off, off_t, on_t):
        if x != y:
            k = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            print(f"\n{n} differs at character {k}:\n   OFF: {x!r}\n   ON : {y!r}")
    same = sum(x == y for x, y in zip(off_t, on_t))
    print(f"\nanswers identical with the cache on and off: {same}/{len(off_t)}")


if __name__ == "__main__":
    main()
