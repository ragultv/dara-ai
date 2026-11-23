"""Time to first token on a CONTINUED chat (Husky's prefix-cache number), Dara server vs llama-server.

Scenario per run: a system prompt + a long first message (~950 tokens, three real task prompts) -> the model answers (cold turn), then the same
conversation with the answer and ONE short new message is sent (continued turn). A server that keeps the conversation's K/V (prompt cache) only
has to process the new message, so its continued-turn first token should be fast. Each run uses a unique first line so no earlier run's cache helps.
Measured: wall time from sending the request to the first streamed content chunk. Median of --reps runs, the first run is a discarded warm-up.

Usage: python -I bench/chat_ttft_bench.py --models nano 4b [--reps 5]
"""
import argparse, json, os, statistics, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchenv  # noqa: E402

ROOT = benchenv.ROOT


def wait_ready(url, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            r = urllib.request.urlopen(url + "/health", timeout=3)
            body = r.read().decode()
            if r.status == 200 and ('"status":"ok"' in body.replace(" ", "") or "ok" in body.lower()):
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def stream_ttft(url, messages, max_tokens, extra=None):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True, **(extra or {})}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    first, text = None, ""
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            ch = (d.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            piece = delta.get("content") or ""
            if first is None and (piece or delta.get("reasoning_content")):   # some servers stream the model's thought first
                first = time.time() - t0
            text += piece
    if first is None:
        print("   (no content chunk streamed; reply text: %r)" % text[:80], flush=True)
    return first, text


def scenario(url, tasks, nonce, extra):
    long_msg = f"[ref {nonce}]\n" + "\n\n".join(t["prompt"] for t in tasks[:3]) + "\n\nSummarize the above in two sentences."
    conv = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": long_msg}]
    t1, a1 = stream_ttft(url, conv, 48, extra)
    conv += [{"role": "assistant", "content": a1}, {"role": "user", "content": "Thanks. Now answer with just the word OK."}]
    t2, _ = stream_ttft(url, conv, 8, extra)
    return t1, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nano", "4b"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "bench", "chat_ttft"))
    benchenv.add_args(ap)
    a = ap.parse_args()
    benchenv.check_paths(a, a.models)
    machine = benchenv.machine_info()
    benchenv.print_machine(machine)
    tasks = json.load(open(os.path.join(ROOT, "bench", "tasks", "tasks.json"), encoding="utf-8"))["tasks"]
    os.makedirs(a.out, exist_ok=True)
    res = {"machine": machine, "models": {}}
    lines = []
    for model in a.models:
        gguf = benchenv.gguf(a, model)
        res["models"][model] = {}
        for engine in ("llama", "dara"):
            if engine == "llama":
                port = 18089
                cmd = [os.path.join(a.llama_dir, "llama-server.exe"), "-m", gguf, "-c", "4096", *(["-ngl", benchenv.ngl_for(a, model)] if benchenv.ngl_for(a, model) != "auto" else []),
                       "-fa", "on", "--port", str(port), "--no-warmup"]
                extra = {"cache_prompt": True}
            else:
                port = 18090
                cmd = [a.dara_exe, "serve", "--model", gguf, "--port", str(port), "--no-store", "--flash", "off", "--ctx", "4096"]
                extra = {}
            url = f"http://127.0.0.1:{port}"
            log = open(os.path.join(a.out, f"{model}_{engine}.server.log"), "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            try:
                if not wait_ready(url, proc):
                    print(f"{model}/{engine}: server did not become ready (see {a.out})")
                    res["models"][model][engine] = {"error": "not ready"}
                    continue
                cold, cont = [], []
                for i in range(a.reps + 1):
                    t1, t2 = scenario(url, tasks, f"{engine}-{model}-{i}-{time.time()}", extra)
                    if i == 0:
                        continue   # warm-up
                    if t1: cold.append(t1)
                    if t2: cont.append(t2)
                res["models"][model][engine] = {"cold_ttft_s": statistics.median(cold) if cold else None, "continued_ttft_s": statistics.median(cont) if cont else None,
                                                "cold_all": cold, "continued_all": cont}
                r = res["models"][model][engine]
                ms = lambda v: f"{v * 1000:7.0f} ms" if v else "      n/a"
                lines.append(f"{model:5s} {engine:6s} cold first token {ms(r['cold_ttft_s'])} | continued chat first token {ms(r['continued_ttft_s'])}")
                print(lines[-1], flush=True)
            finally:
                proc.terminate()
                try: proc.wait(timeout=30)
                except Exception: proc.kill()
                log.close()
                time.sleep(3)   # let the GPU memory go
    json.dump(res, open(os.path.join(a.out, "chat_ttft.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
