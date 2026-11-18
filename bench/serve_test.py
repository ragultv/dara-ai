"""End-to-end check of `dara serve` (standard library only).

Start the server first, e.g.:
    build\\cuda\\dara.exe serve --model models\\qwen3.5-4b-gguf\\Qwen3.5-4B-Q4_K_M.gguf --mmproj auto
then:
    python -I bench\\serve_test.py [--url http://127.0.0.1:11435] [--image some.png] [--compact]

Checks: /health, /v1/models, non-streaming chat with usage + timings, streaming chat (SSE, final chunk carries usage), the prompt cache on a
follow-up turn (cached_tokens > 0), greedy speculative decoding output == plain greedy output, stop strings, /dara/tokenize, /dara/stats,
/metrics, optional image input, optional compaction (--compact sends a conversation larger than the context).
"""
import argparse
import base64
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:11435")
ap.add_argument("--image", default=None)
ap.add_argument("--compact", action="store_true")
a = ap.parse_args()
fails = 0


def req(path, body=None, stream=False):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(a.url + path, data=data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(r, timeout=900)
    if stream:
        return resp
    raw = resp.read().decode()
    return json.loads(raw) if raw.startswith("{") else raw


def check(name, ok, info=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))
    if not ok:
        fails += 1


def summary(r):
    t, u = r["timings"], r["usage"]
    return (f"prompt {u['prompt_tokens']} cached {u['prompt_tokens_details']['cached_tokens']} +{u['completion_tokens']} | prefill {t['prefill_tokens_per_second']} tok/s"
            f" | ttft {t['ttft_ms']} ms | decode {t['decode_tokens_per_second']} tok/s | spec {r['speculative']['mode']}")


h = req("/health")
check("health", h.get("status") == "ok", h.get("model"))
m = req("/v1/models")
loaded = [d for d in m["data"] if d.get("dara", {}).get("loaded")]
check("models (catalog + loaded model)", len(loaded) == 1 and loaded[0]["id"] == h["model"],
      ", ".join(f"{d['id']}:{d['dara']['status']}" for d in m["data"]))

conv = [{"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name three primary colors, one per line."}]
r1 = req("/v1/chat/completions", {"messages": conv, "max_tokens": 64, "temperature": 0})
txt1 = r1["choices"][0]["message"]["content"]
check("chat non-stream", len(txt1) > 0 and r1["usage"]["completion_tokens"] > 0 and r1["timings"]["decode_tokens_per_second"] >= 0, summary(r1))
print("   ->", txt1.replace("\n", " | ")[:200])

conv2 = conv + [{"role": "assistant", "content": txt1}, {"role": "user", "content": "Now name one secondary color."}]
r2 = req("/v1/chat/completions", {"messages": conv2, "max_tokens": 32, "temperature": 0})
check("prompt cache on follow-up turn", r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0, summary(r2))

r3 = req("/v1/chat/completions", {"messages": conv, "max_tokens": 64, "temperature": 0})
check("greedy is deterministic (cache + speculation exact)", r3["choices"][0]["message"]["content"] == txt1, summary(r3))

# streaming
resp = req("/v1/chat/completions", {"messages": conv, "max_tokens": 64, "temperature": 0.7, "seed": 7, "stream": True}, stream=True)
pieces, last, done, t0, first_at = [], None, False, time.time(), None
for line in resp:
    line = line.decode().strip()
    if not line.startswith("data: "):
        continue
    payload = line[6:]
    if payload == "[DONE]":
        done = True
        break
    j = json.loads(payload)
    if "error" in j:
        print("   stream error:", j["error"])
        break
    d = j["choices"][0].get("delta", {})
    if d.get("content"):
        first_at = first_at or time.time()
        pieces.append(d["content"])
    if j["choices"][0].get("finish_reason"):
        last = j
check("stream: chunks + [DONE] + usage on the final chunk", done and last is not None and "usage" in last and len(pieces) > 0,
      summary(last) if last else "no final chunk")

# stop string
r4 = req("/v1/chat/completions", {"messages": [{"role": "user", "content": "Count from 1 to 10 separated by spaces."}], "max_tokens": 64, "temperature": 0, "stop": ["5"]})
t4 = r4["choices"][0]["message"]["content"]
check("stop string", "5" not in t4 and r4["choices"][0]["finish_reason"] == "stop", repr(t4[:60]))

# completions
r5 = req("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0})
check("completions", len(r5["choices"][0]["text"]) > 0, repr(r5["choices"][0]["text"]))

tk = req("/dara/tokenize", {"messages": conv})
check("tokenize", tk["tokens"] > 10, str(tk))

if a.image:
    b64 = base64.b64encode(open(a.image, "rb").read()).decode()
    mime = "image/png" if a.image.lower().endswith(".png") else "image/jpeg"
    ri = req("/v1/chat/completions", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}, {"type": "text", "text": "Describe this image in one sentence."}]}],
        "max_tokens": 64, "temperature": 0})
    check("image input", ri["usage"]["prompt_tokens_details"]["image_tokens"] > 0, summary(ri) + f" | image {ri['timings']['image_ms']} ms")
    print("   ->", ri["choices"][0]["message"]["content"][:200])

if a.compact:
    ctx = req("/dara/stats")["context"]["capacity"]
    filler = "This is filler sentence number {} about the history of the project and its many details. "
    msgs = [{"role": "system", "content": "You are helpful. The secret word is PELICAN."}]
    i = 0
    while sum(len(x["content"]) for x in msgs) < ctx * 5:   # ~1.2+ tokens per 5 chars -> well over the context
        msgs.append({"role": "user", "content": "".join(filler.format(i * 40 + k) for k in range(40))})
        msgs.append({"role": "assistant", "content": "Noted."})
        i += 1
    msgs.append({"role": "user", "content": "What is the secret word? Answer with one word."})
    t0 = time.time()
    rc = req("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0})
    check("compaction (conversation larger than the context)", rc["context"]["compacted"], f"{json.dumps(rc['context'])} | compact {rc['timings']['compact_ms']} ms | {time.time() - t0:.1f} s")
    print("   ->", rc["choices"][0]["message"]["content"])
    rc2 = req("/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0})
    check("compaction summary reused", rc2["context"]["summary_cached"], summary(rc2))

s = req("/dara/stats")
need = ["process", "ram", "gpu", "disk", "model", "context", "engine", "last_request"]
check("stats", all(k in s for k in need), f"private {s['process']['private_working_set_mb']} MB, VRAM used {s['gpu']['vram_used_mb']} MB, "
      f"avg decode {s['engine']['avg_decode_tokens_per_second']} tok/s")
mt = req("/metrics")
check("metrics", "dara_requests_total" in mt)
print(json.dumps(s, indent=1)[:3000])
print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
sys.exit(1 if fails else 0)
