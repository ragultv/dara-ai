"""K/V format x context benchmark through `dara serve`, per model tier (and Flash on / off where a drafter exists).

Phase A (context setting): for each context size, start the server with --ctx N and record whether it fits, the placement (layers on the GPU,
output head location), and decode speed with an empty context (256 tokens, greedy).
Phase B (context fill): at the largest context that fits, grow ONE conversation through the depths (the prompt cache means only the new part is
prefilled each step). At each depth: prefill speed of the new part, then a question at the end: time to first token, decode speed over ~200
tokens, and a needle check (a code given in the system message must be recalled).

Usage: python -I bench/kv_matrix.py [--models nano 4b 9b] [--formats q8 k8v4 q4 tq4] [--flash off on] [--depths 4096 8192 ...] [--phase A B]
Results: build/bench/kv_matrix/<model>_<format>_<flash>.json (appended as it runs) and a summary table at the end.
"""
import argparse, glob, json, os, re, subprocess, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "bench", "kv_matrix")
IDS = {"nano": "dara-nano", "4b": "dara-core-4b", "9b": "dara-pro-9b"}
HAS_DRAFTER = {"nano", "4b"}
NATIVE = {"nano": 131072, "4b": 262144, "9b": 262144}
CODE = "4-7-1-9-3-8"
SYSTEM = f"You are a careful assistant. Remember this exactly: the vault code is {CODE}. You will be asked for it later."
QUESTION = ("What is the vault code given at the very beginning of this conversation? Reply with the code first, then write a 150-word "
            "overview of what the document parts above are about.")
PROSE = "Explain how a hash map works, in detail."


def filler_chunks(chars=7000):
    """Long, varied, deterministic text: the repository's own docs and sources."""
    files = sorted(glob.glob(os.path.join(ROOT, "docs", "*.md")) + glob.glob(os.path.join(ROOT, "src", "**", "*.*"), recursive=True) +
                   glob.glob(os.path.join(ROOT, "bench", "*.py")) + glob.glob(os.path.join(ROOT, "tools", "*.cu")))
    text = ""
    for f in files:
        try:
            text += f"\n\n# file: {os.path.relpath(f, ROOT)}\n" + open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    text = text.replace("<|", "< |").replace("|>", "| >")   # never feed template-like markers
    return [text[i:i + chars] for i in range(0, len(text), chars)]


class Server:
    def __init__(self, a, model, fmt, flash, ctx):
        self.port = a.port
        args = [a.exe, "serve", "--model", IDS[model], "--models-dir", a.models_dir, "--port", str(a.port), "--ctx", str(ctx), "--kv-format", fmt,
                "--flash", flash, "--max-tokens", "256"]
        if a.reserve: args += ["--vram-reserve", str(a.reserve)]
        self.p = subprocess.Popen(args, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace")
        self.info, self.ok, self.log = "", False, []
        while True:
            line = self.p.stderr.readline()
            if not line: break
            self.log.append(line.rstrip())
            if "layers on GPU" in line: self.info = line.strip()
            if "listening" in line: self.ok = True; break
        if not self.ok: self.stop()

    def post(self, path, body):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=7200))

    def get(self, path):
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=60))

    def stop(self):
        try:
            self.p.terminate(); self.p.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.p.kill()


def placement(info):
    m = re.search(r"layers on GPU (\d+)/(\d+)", info)
    return {"layers_gpu": int(m.group(1)) if m else None, "layers": int(m.group(2)) if m else None, "head": "cpu" if "head in RAM" in info else "gpu",
            "flash_active": "flash off" not in info}


def save(path, rec):
    os.makedirs(OUT, exist_ok=True)
    json.dump(rec, open(path, "w"), indent=1)


def phase_a(a, model, fmt, flash, rec, path):
    rec.setdefault("settings", {})
    for d in a.depths:
        if d > NATIVE[model]: break
        s = Server(a, model, fmt, flash, d)
        if not s.ok:
            err = next((l for l in reversed(s.log) if "ERROR" in l), s.log[-1] if s.log else "no output")
            rec["settings"][str(d)] = {"fits": False, "error": err[:300]}
            print(f"  A ctx {d:6d}: does not fit ({err[:120]})", flush=True)
            save(path, rec)
            break
        try:
            st = s.get("/dara/stats")
            r = s.post("/v1/chat/completions", {"messages": [{"role": "user", "content": PROSE}], "max_tokens": 256, "temperature": 0,
                                                "flash": flash == "on"})
            row = {"fits": True, **placement(s.info), "vram_used_mb": st["gpu"]["vram_used_mb"], "private_ram_mb": st["process"]["private_working_set_mb"],
                   "decode_tps": r["timings"]["decode_tokens_per_second"], "spec": r["speculative"]["mode"], "tokens_per_pass": r["speculative"]["tokens_per_pass"]}
            rec["settings"][str(d)] = row
            print(f"  A ctx {d:6d}: {row['layers_gpu']}/{row['layers']} layers, head {row['head']}, flash {'on' if row['flash_active'] else 'off'}, "
                  f"VRAM {row['vram_used_mb']:.0f} MB -> {row['decode_tps']:.2f} tok/s (empty context)", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["settings"][str(d)] = {"fits": True, "error": str(e)[:300]}
            print(f"  A ctx {d:6d}: request failed: {e}", flush=True)
        finally:
            s.stop()
        save(path, rec)


def phase_b(a, model, fmt, flash, rec, path):
    fitting = [int(k) for k, v in rec.get("settings", {}).items() if v.get("fits") and "error" not in v]
    if not fitting:
        print("  B: nothing fits", flush=True)
        return
    ctx = max(fitting)
    s = Server(a, model, fmt, flash, ctx)
    if not s.ok:
        print("  B: server did not start", flush=True)
        return
    rec["fill"] = {"ctx": ctx, **placement(s.info), "depths": {}}
    chunks = filler_chunks()
    msgs = [{"role": "system", "content": SYSTEM}]
    ci = 0
    try:
        for d in a.depths:
            if d > ctx - 600: break
            # grow the conversation to ~d tokens (prompt incl. the question), one document part per user turn
            while True:
                n = s.post("/dara/tokenize", {"messages": msgs + [{"role": "user", "content": QUESTION}]})["tokens"]
                if n >= d - 400 or ci >= len(chunks): break
                if msgs[-1]["role"] == "user": msgs.append({"role": "assistant", "content": "Noted."})
                msgs.append({"role": "user", "content": f"Here is part {ci + 1} of a document:\n\n{chunks[ci]}"})
                ci += 1
            # prime: prefill the new part and save it in the prompt cache
            t0 = time.time()
            p = s.post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1, "temperature": 0})
            prefill = p["timings"]
            # the question (does not overwrite the cached conversation)
            q = s.post("/v1/chat/completions", {"messages": msgs + [{"role": "assistant", "content": "Noted."}, {"role": "user", "content": QUESTION}],
                                                "max_tokens": 200, "temperature": 0, "flash": flash == "on", "cache_write": False})
            text = q["choices"][0]["message"]["content"]
            found = CODE in text or CODE.replace("-", "") in text.replace(" ", "").replace("-", "")
            row = {"prompt_tokens": q["usage"]["prompt_tokens"], "prefill_new_tokens": prefill["prefill_tokens"], "prefill_tps": prefill["prefill_tokens_per_second"],
                   "prime_s": round(time.time() - t0, 2), "ttft_ms": q["timings"]["ttft_ms"], "cached": q["usage"]["prompt_tokens_details"]["cached_tokens"],
                   "decode_tps": q["timings"]["decode_tokens_per_second"], "decoded": q["timings"]["decode_tokens"], "needle": found,
                   "spec": q["speculative"]["mode"], "tokens_per_pass": q["speculative"]["tokens_per_pass"], "answer_head": text[:80]}
            rec["fill"]["depths"][str(d)] = row
            print(f"  B depth {row['prompt_tokens']:6d}: prefill {row['prefill_tps']:7.1f} tok/s ({row['prefill_new_tokens']} new), ttft {row['ttft_ms']:7.0f} ms, "
                  f"decode {row['decode_tps']:6.2f} tok/s, needle {'OK' if found else 'MISSED'}", flush=True)
            save(path, rec)
    except Exception as e:  # noqa: BLE001
        rec["fill"]["error"] = str(e)[:300]
        print(f"  B failed: {e}", flush=True)
    finally:
        s.stop()
        save(path, rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["nano", "4b", "9b"])
    ap.add_argument("--formats", nargs="+", default=["q8", "k8v4", "q4", "tq4"])
    ap.add_argument("--flash", nargs="+", default=["off", "on"])
    ap.add_argument("--depths", nargs="+", type=int, default=[4096, 8192, 12288, 16384, 24576, 32768, 49152, 57344, 65536, 98304, 131072])
    ap.add_argument("--phase", nargs="+", default=["A", "B"])
    ap.add_argument("--exe", default=os.path.join(ROOT, "build", "cuda", "dara.exe"))
    ap.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--port", type=int, default=11440)
    ap.add_argument("--reserve", type=int, default=0, help="--vram-reserve MB (0: the default)")
    a = ap.parse_args()
    for model in a.models:
        for flash in a.flash:
            if flash == "on" and model not in HAS_DRAFTER: continue
            for fmt in a.formats:
                path = os.path.join(OUT, f"{model}_{fmt}_{flash}.json")
                rec = json.load(open(path)) if os.path.exists(path) else {"model": model, "format": fmt, "flash": flash}
                print(f"== {model} kv={fmt} flash={flash}", flush=True)
                if "A" in a.phase: phase_a(a, model, fmt, flash, rec, path)
                if "B" in a.phase: phase_b(a, model, fmt, flash, rec, path)


if __name__ == "__main__":
    main()
