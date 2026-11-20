r"""End-to-end test of the app database endpoints of `dara serve` (docs/ui/08-database.md). No GPU or model needed.
Usage: python -I tests/store_api_test.py [path\to\dara.exe]    (default build\cuda\dara.exe, or DARA_EXE)"""
import base64, json, os, shutil, subprocess, sys, tempfile, time, urllib.error, urllib.request

EXE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DARA_EXE", os.path.join("build", "cuda", "dara.exe")))
PORT = 11441
KEY = "test-key-123"
U = f"http://127.0.0.1:{PORT}"
failures = []


def check(name, cond, detail=""):
    print(("ok    " if cond else "FAIL  ") + name + ("" if cond else f"   {detail}"))
    if not cond: failures.append(name)


def call(method, path, body=None, key=KEY, raw=False):
    data = None if body is None else json.dumps(body).encode()
    h = {"Content-Type": "application/json"}
    if key: h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(U + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=30) as f: b = f.read(); return f.status, (b if raw else json.loads(b or b"{}"))
    except urllib.error.HTTPError as e:
        b = e.read(); return e.code, (b if raw else json.loads(b or b"{}"))


def start(data_dir, env_key=True):
    env = dict(os.environ)
    if env_key: env["DARA_API_KEY"] = KEY   # also tests the environment variable
    p = subprocess.Popen([EXE, "serve", "--port", str(PORT), "--data-dir", data_dir, "--models-dir", os.path.join(data_dir, "models")],
                         stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace", env=env)
    for l in p.stderr:
        if "listening" in l: break
    import threading
    threading.Thread(target=lambda: [None for _ in p.stderr], daemon=True).start()
    return p


def stop(p):
    p.terminate(); p.wait(timeout=20)


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
data_dir = tempfile.mkdtemp(prefix="dara_store_test_")
p = start(data_dir)
try:
    # auth
    s, _ = call("GET", "/dara/settings", key=None); check("settings without key -> 401", s == 401, s)
    s, _ = call("GET", "/dara/settings", key="wrong"); check("settings with wrong key -> 401", s == 401, s)

    s, info = call("GET", "/dara/store")
    check("store info", s == 200 and info["schema_version"] == 1 and info["conversations"] == 0, info)
    check("database file exists", os.path.exists(os.path.join(data_dir, "dara.db")))

    # settings
    s, st = call("GET", "/dara/settings"); check("default settings", s == 200 and st["theme"] == "system" and st["chat.max_tokens"] == 4096, st)
    s, st = call("PATCH", "/dara/settings", {"theme": "dark", "chat.temperature": 0.2, "custom.list": [1, {"a": None}]})
    check("patch settings", s == 200 and st["theme"] == "dark" and st["custom.list"] == [1, {"a": None}], st)
    s, e = call("PATCH", "/dara/settings", {"Bad Key": 1}); check("bad setting key -> 400", s == 400, e)
    s, e = call("PATCH", "/dara/settings", []); check("non-object settings body -> 400", s == 400, e)
    s, _ = call("DELETE", "/dara/settings/custom.list"); check("delete setting", s == 200)
    s, _ = call("DELETE", "/dara/settings/custom.list"); check("delete missing setting -> 404", s == 404)

    # conversations and messages
    s, c = call("POST", "/dara/conversations", {"title": "Purple notes", "model": "dara-nano", "params": {"temperature": 0.5}})
    check("create conversation", s == 201 and len(c["id"]) == 36 and c["params"] == {"temperature": 0.5} and c["message_count"] == 0, c)
    cid = c["id"]
    s, e = call("POST", "/dara/conversations", {"id": cid}); check("duplicate id -> 409", s == 409, e)
    s, e = call("POST", "/dara/conversations", {"id": "nope"}); check("non-uuid id -> 400", s == 400, e)
    s, m1 = call("POST", f"/dara/conversations/{cid}/messages", {"role": "user", "content": "Tell me about purple elephants"})
    check("append user message", s == 201 and m1["position"] == 0 and m1["attachments"] == [], m1)
    s, m2 = call("POST", f"/dara/conversations/{cid}/messages", {"role": "assistant", "content": "", "model": "dara-nano"})
    check("append assistant message (position 1)", s == 201 and m2["position"] == 1, m2)
    s, e = call("POST", f"/dara/conversations/{cid}/messages", {"role": "robot", "content": "x"}); check("bad role -> 400", s == 400, e)
    s, e = call("POST", "/dara/conversations/00000000-0000-4000-8000-000000000000/messages", {"role": "user", "content": "x"}); check("message to missing conversation -> 404", s == 404, e)
    stats = {"usage": {"completion_tokens": 14}, "timings": {"decode_tokens_per_second": 56.7}}
    s, m2 = call("PATCH", f"/dara/messages/{m2['id']}", {"content": "Elephants are large and purple in this story.", "finish_reason": "stop", "stats": stats})
    check("finish a streamed reply (patch)", s == 200 and m2["finish_reason"] == "stop" and m2["stats"] == stats, m2)
    s, conv = call("GET", f"/dara/conversations/{cid}")
    check("get conversation with messages in order", s == 200 and [m["role"] for m in conv["messages"]] == ["user", "assistant"] and conv["message_count"] == 2, conv)
    s, conv = call("PATCH", f"/dara/conversations/{cid}", {"title": "Renamed", "pinned": True})
    check("patch conversation", s == 200 and conv["title"] == "Renamed" and conv["pinned"] is True, conv)
    s, lst = call("GET", "/dara/conversations?limit=10")
    check("list conversations", s == 200 and lst["total"] == 1 and lst["data"][0]["id"] == cid, lst)

    # search (FTS) including the sync after an update, and hostile input
    s, r = call("GET", "/dara/search?q=eleph")
    check("search finds by prefix", s == 200 and len(r["data"]) >= 1 and r["data"][0]["conversation_id"] == cid, r)
    s, r = call("GET", "/dara/search?q=" + urllib.request.quote('"; DROP TABLE messages; -- NEAR('))
    check("search with hostile input is safe", s == 200, r)
    s, r = call("GET", "/dara/search?q=zzzznotthere"); check("search no match", s == 200 and r["data"] == [], r)

    # attachments
    s, a = call("POST", f"/dara/messages/{m1['id']}/attachments", {"data": "data:image/png;base64," + base64.b64encode(PNG).decode()})
    check("attach png", s == 201 and a["mime"] == "image/png" and a["bytes"] == len(PNG), a)
    s, body = call("GET", f"/dara/attachments/{a['id']}", raw=True); check("download attachment bytes", s == 200 and body == PNG)
    files = os.listdir(os.path.join(data_dir, "attachments")); check("attachment is a file", len(files) == 1 and files[0].endswith(".png"), files)
    s, e = call("POST", f"/dara/messages/{m1['id']}/attachments", {"data": base64.b64encode(b"not an image at all").decode()}); check("non-image rejected -> 400", s == 400, e)
    s, conv = call("GET", f"/dara/conversations/{cid}"); check("attachments listed on the message", conv["messages"][0]["attachments"][0]["id"] == a["id"], conv)

    # model configs
    s, mc = call("PUT", "/dara/model-configs/dara-nano", {"kv": "q4", "flash": True, "vision": True, "context": 65536})
    check("put model config", s == 200 and mc["kv"] == "q4" and mc["flash"] == "on", mc)
    s, e = call("PUT", "/dara/model-configs/dara-nano", {"kv": "bogus"}); check("bad kv -> 400", s == 400, e)
    s, allc = call("GET", "/dara/model-configs"); check("list model configs", allc["dara-nano"]["context"] == 65536, allc)

    # persistence across a restart
    stop(p); p = start(data_dir)
    s, conv = call("GET", f"/dara/conversations/{cid}")
    check("conversation survives restart", s == 200 and conv["title"] == "Renamed" and len(conv["messages"]) == 2, conv)
    s, st = call("GET", "/dara/settings"); check("settings survive restart", st["theme"] == "dark", st)
    s, r = call("GET", "/dara/search?q=purple"); check("search index survives restart", len(r["data"]) >= 1, r)

    # truncate (regenerate / edit-and-resend)
    s, r = call("DELETE", f"/dara/conversations/{cid}/messages?from=1"); check("truncate from position 1", s == 200 and r["deleted"] == 1, r)
    s, conv = call("GET", f"/dara/conversations/{cid}"); check("one message left", len(conv["messages"]) == 1, conv)

    # export
    s, ex = call("GET", "/dara/data/export")
    check("export everything", s == 200 and ex["format"] == "dara-export" and len(ex["conversations"]) == 1 and ex["settings"]["theme"] == "dark", ex)

    # delete one conversation: cascade + files + search
    s, _ = call("DELETE", f"/dara/conversations/{cid}"); check("delete conversation", s == 200)
    check("attachment file removed with the conversation", os.listdir(os.path.join(data_dir, "attachments")) == [])
    s, r = call("GET", "/dara/search?q=purple"); check("search entries removed", r["data"] == [], r)
    s, _ = call("GET", f"/dara/conversations/{cid}"); check("deleted conversation -> 404", s == 404)

    # delete all needs the confirmation
    call("POST", "/dara/conversations", {"title": "x"})
    s, e = call("DELETE", "/dara/data", {}); check("delete-all without confirm -> 400", s == 400, e)
    s, r = call("DELETE", "/dara/data", {"confirm": "delete-all-data"})
    s2, lst = call("GET", "/dara/conversations")
    s3, allc = call("GET", "/dara/model-configs")
    check("delete all clears conversations and model configs", s == 200 and lst["total"] == 0 and allc == {}, (r, lst, allc))
    s, st = call("GET", "/dara/settings"); check("settings kept by delete all", st["theme"] == "dark", st)

    # corrupt file recovery
    stop(p)
    with open(os.path.join(data_dir, "dara.db"), "wb") as f: f.write(b"this is not a sqlite database" * 100)
    for suffix in ("-wal", "-shm"):
        try: os.remove(os.path.join(data_dir, "dara.db" + suffix))
        except OSError: pass
    p = start(data_dir)
    s, info = call("GET", "/dara/store")
    check("corrupt database is replaced and reported", s == 200 and "integrity" in info.get("note", ""), info)
    check("corrupt file kept for the user", any(n.startswith("dara.db.corrupt-") for n in os.listdir(data_dir)), os.listdir(data_dir))
    s, c = call("POST", "/dara/conversations", {"title": "after recovery"}); check("new database works", s == 201, c)
    stop(p)

    # the key from --api-key / environment is not the only gate: without a key the server is open (unchanged behaviour) and has the store
    p = start(data_dir, env_key=False)
    s, info = call("GET", "/dara/store", key=None); check("works without an API key when none is set", s == 200, info)
    stop(p)

    # --no-store -> 503
    p2 = subprocess.Popen([EXE, "serve", "--port", str(PORT), "--data-dir", data_dir, "--no-store", "--models-dir", os.path.join(data_dir, "models")], stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, errors="replace")
    for l in p2.stderr:
        if "listening" in l: break
    s, e = call("GET", "/dara/conversations", key=None); check("--no-store answers 503 store_unavailable", s == 503 and e["error"]["code"] == "store_unavailable", e)
    stop(p2)
finally:
    try: p.terminate()
    except Exception: pass
    shutil.rmtree(data_dir, ignore_errors=True)

print(f"\n{'ALL PASSED' if not failures else str(len(failures)) + ' FAILED: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
