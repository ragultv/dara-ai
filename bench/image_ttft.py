"""Time to first token with an image, Dara vs llama.cpp (llama-server, the way oprel-SDK runs local vision models).

llama.cpp: llama-server with the model + mmproj already loaded; time from the HTTP request (OpenAI chat, base64 image, streaming) to the
first streamed content token. Each configuration runs twice and the second (warm) run is reported.
Dara:      warm time to first token = image decode + resize/preprocess + vision encode + prompt prefill (everything after both models are
           loaded), from Dara's own timing lines.

Usage: python -I bench/image_ttft.py [--models 4b 9b nano] [--out build/bench/image_ttft.json]
"""
import argparse, base64, json, os, re, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = r"D:\oprel\bin\llama_cpp\b12189\cuda"
DARA = os.path.join(ROOT, "build", "cuda", "dara.exe")
M = {
    "4b": dict(gguf="models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf", mmproj="models/qwen3.5-4b-gguf/mmproj-F16.gguf", ngl=99),
    "9b": dict(gguf="models/qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf", mmproj="models/qwen3.5-9b-gguf/mmproj-F16.gguf", ngl=24),
    "nano": dict(gguf="models/gemma4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf", mmproj="models/gemma4-e2b-gguf/mmproj-F16.gguf", ngl=99),
}
QUESTION = "What text is written in the white box? Answer briefly."
IMAGES = {"full 1920x1080": "build/vision/photo1080.jpg", "oprel-preprocessed 853x480": "build/vision/photo1080_oprel.jpg"}
PORT = 8099


def p(rel): return os.path.join(ROOT, rel)


def llama_server_ttft(model):
    m = M[model]
    cmd = [os.path.join(LLAMA_DIR, "llama-server.exe"), "-m", p(m["gguf"]), "--mmproj", p(m["mmproj"]), "-ngl", str(m["ngl"]), "-fa", "on",
           "-c", "8192", "--port", str(PORT), "--no-webui"]
    log = open(p(f"build/bench/llama_server_{model}.log"), "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, cwd=LLAMA_DIR, stdout=log, stderr=subprocess.STDOUT)
    out = {}
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None: return {"error": f"llama-server exited with {srv.returncode} (see build/bench/llama_server_{model}.log)"}
            try:
                if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)).get("status") == "ok": break
            except Exception: pass
            if time.time() - t0 > 300: return {"error": "llama-server did not become ready"}
            time.sleep(1)
        for name, img in IMAGES.items():
            b64 = base64.b64encode(open(p(img), "rb").read()).decode()
            body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                                                               {"type": "text", "text": QUESTION}]}],
                    "max_tokens": 24, "temperature": 0, "stream": True, "cache_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            res = None
            for rep in range(2):
                req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
                t1 = time.time(); ttft = None; text = ""
                with urllib.request.urlopen(req, timeout=900) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:") or line.endswith("[DONE]"): continue
                        d = json.loads(line[5:])
                        delta = (d.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                        if delta and ttft is None: ttft = time.time() - t1
                        text += delta
                res = {"ttft_s": round(ttft, 3) if ttft else None, "answer": text.strip()[:80]}
            out[name] = res
            print(f"  llama.cpp {model} {name}: TTFT {res['ttft_s']} s | {res['answer']!r}", flush=True)
    finally:
        srv.terminate()
        try: srv.wait(30)
        except Exception: srv.kill()
    return out


def dara_ttft(model, img, max_tokens):
    m = M[model]
    cmd = [DARA, "--gguf", p(m["gguf"]), "--mmproj", p(m["mmproj"]), "--image", p(img), "--chat", QUESTION, "--stop-at-eos", "--new", "24",
           "--out", p("build/bench/_dara_img")]
    if model == "nano": cmd += ["--no-layers"]
    else: cmd += ["--bench", "--image-max-tokens", str(max_tokens)]
    o = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT).stdout
    mi = re.search(r"-> (\d+) tokens, preprocessed in ([0-9.]+) s, encoded in ([0-9.]+) s", o)
    mp = re.search(r"prefill (\d+) tokens: ([0-9.]+) s", o)
    ans = o.split("--- generated ---", 1)[1].split("--- end", 1)[0] if "--- generated ---" in o else ""
    ans = re.sub(r"<\|im_end\|>|<turn\|>|flash[^\n]*", "", ans).strip()
    if not (mi and mp): return {"error": o[-300:]}
    prep, enc, pre = float(mi.group(2)), float(mi.group(3)), float(mp.group(2))
    return {"image_tokens": int(mi.group(1)), "prompt_tokens": int(mp.group(1)), "preprocess_s": prep, "encode_s": enc, "prefill_s": pre,
            "ttft_s": round(prep + enc + pre, 3), "answer": ans[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4b", "9b", "nano"])
    ap.add_argument("--out", default=p("build/bench/image_ttft.json"))
    a = ap.parse_args()
    os.makedirs(p("build/bench"), exist_ok=True)
    res = {}
    for model in a.models:
        r = res[model] = {"dara": {}}
        for name, img in IMAGES.items():
            for cap in ([1024, 400] if model != "nano" and name.startswith("full") else [1024]):
                key = f"{name}" + (f" (cap {cap})" if model != "nano" else "")
                d = dara_ttft(model, img, cap); d2 = dara_ttft(model, img, cap)   # second run: warm file cache
                r["dara"][key] = d2 if "ttft_s" in d2 else d
                x = r["dara"][key]
                print(f"  Dara {model} {key}: TTFT {x.get('ttft_s')} s (prep {x.get('preprocess_s')}, encode {x.get('encode_s')}, prefill {x.get('prefill_s')}, "
                      f"{x.get('image_tokens')} image tokens) | {x.get('answer')!r} {x.get('error', '')}", flush=True)
        r["llama.cpp"] = llama_server_ttft(model)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
