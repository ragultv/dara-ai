r"""End-to-end test of the app database endpoints of `dara serve` (docs/ui/08-database.md). No GPU or model needed.
Usage: python -I tests/store_api_test.py [path\to\dara.exe]    (default build\cuda\dara.exe, or DARA_EXE)"""
import base64, json, os, shutil, sqlite3, subprocess, sys, tempfile, time, urllib.error, urllib.request

EXE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DARA_EXE", os.path.join("build", "cuda", "dara.exe")))
PORT = 11441
KEY = "test-key-123"
U = f"http://127.0.0.1:{PORT}"
failures = []
# test-only limits for version history (every server this test starts): an edit more than 1.5 s after the last save snapshots, 5 versions are kept per artifact
VERSION_ENV = {"DARA_VERSION_GAP_MS": "1500", "DARA_VERSION_CAP": "5",
               # test-only limits for the chat, comments and assets (docs/ui/10-artifact-collab.md): 6 chat messages, 3000 bytes per asset, 5000 per artifact
               "DARA_CHAT_CAP": "6", "DARA_ASSET_MAX_BYTES": "3000", "DARA_ASSET_ARTIFACT_MAX_BYTES": "5000"}


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


def call_raw(method, path, data=None, headers=None, key=KEY):
    """A request with a raw body and its own headers (the asset endpoints). Returns (status, headers, body bytes)."""
    h = dict(headers or {})
    if key: h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(U + path, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=30) as f: return f.status, dict(f.headers), f.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def blob(kind, n):
    """n bytes that start like a real file of that kind (the server checks the first bytes against the declared type)."""
    head = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff\xe0", "gif": b"GIF89a", "webp": b"RIFF\x00\x00\x00\x00WEBP",
            "mp4": b"\x00\x00\x00\x18ftypmp42", "webm": b"\x1a\x45\xdf\xa3"}[kind]
    return head + bytes((i * 7 + 3) % 256 for i in range(max(0, n - len(head))))


MIME = {"png": "image/png", "jpeg": "image/jpeg", "gif": "image/gif", "webp": "image/webp", "mp4": "video/mp4", "webm": "video/webm"}


def start(data_dir, env_key=True):
    env = dict(os.environ)
    env.update(VERSION_ENV)
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
    check("store info", s == 200 and info["schema_version"] == 4 and info["conversations"] == 0 and info["artifacts"] == 0 and info["artifact_comments"] == 0 and info["artifact_assets"] == 0, info)
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

    # artifacts (schema version 2): documents now, slides and designs later
    s, a = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Plan", "content": "# Plan\n\n- one\n- two\n" + "x" * 900})
    check("create a document", s == 201 and a["kind"] == "doc" and a["title"] == "Plan" and a["content"].startswith("# Plan") and a["chars"] > 900, a)
    aid = a["id"]
    s, d2 = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Second"}); check("a document may start empty (kind defaults to doc)", s == 201 and d2["content"] == "", d2)
    s, sl = call("POST", "/dara/artifacts", {"kind": "slides", "title": "Deck", "content": "{}"}); check("slides use the same table", s == 201 and sl["kind"] == "slides", sl)
    s, e = call("POST", "/dara/artifacts", {"kind": "spreadsheet"}); check("unknown kind -> 400", s == 400, e)
    s, e = call("POST", "/dara/artifacts", {"id": aid, "kind": "doc"}); check("duplicate id -> 409", s == 409, e)
    s, e = call("POST", "/dara/artifacts", {"id": "not-a-uuid", "kind": "doc"}); check("non-UUID id -> 400", s == 400, e)
    s, e = call("POST", "/dara/artifacts", {"kind": "doc", "content": "z" * (2 * 1024 * 1024 + 1)}); check("content over 2 MB -> 400", s == 400, e)
    s, lst = call("GET", "/dara/artifacts")
    check("list all artifacts, newest first, with a snippet and no full content", s == 200 and lst["total"] == 3 and "content" not in lst["data"][0] and "snippet" in lst["data"][0], lst)
    s, lst = call("GET", "/dara/artifacts?kind=doc")
    check("list filtered by kind", s == 200 and lst["total"] == 2 and all(x["kind"] == "doc" for x in lst["data"]), lst)
    plan = next(x for x in lst["data"] if x["id"] == aid)
    check("snippet is the start of the content (600 chars at most)", plan["snippet"].startswith("# Plan") and len(plan["snippet"]) == 600, len(plan["snippet"]))
    s, e = call("GET", "/dara/artifacts?kind=nope"); check("list with an unknown kind -> 400", s == 400, e)
    s, g = call("GET", f"/dara/artifacts/{aid}"); check("get one with the full content", s == 200 and g["content"].endswith("x" * 900), g.get("chars"))
    time.sleep(0.01)
    s, u = call("PATCH", f"/dara/artifacts/{aid}", {"title": "Plan v2", "content": "# New\n\nbody", "pinned": True})
    check("patch title, content and pinned", s == 200 and u["title"] == "Plan v2" and u["content"] == "# New\n\nbody" and u["pinned"] is True and u["updated_at"] >= g["updated_at"], u)
    s, lst = call("GET", "/dara/artifacts?kind=doc"); check("pinned documents come first", lst["data"][0]["id"] == aid, [x["id"] for x in lst["data"]])
    s, e = call("PATCH", "/dara/artifacts/00000000-0000-4000-8000-000000000000", {"title": "x"}); check("patch unknown -> 404", s == 404, e)
    s, e = call("GET", "/dara/artifacts/00000000-0000-4000-8000-000000000000"); check("get unknown -> 404", s == 404, e)
    s, st = call("GET", "/dara/store"); check("store info counts artifacts and reports schema 4", st.get("artifacts") == 3 and st.get("schema_version") == 4, st)
    s, ex = call("GET", "/dara/data/export"); check("export includes artifacts", len(ex.get("artifacts", [])) == 3 and any(x["content"] == "# New\n\nbody" for x in ex["artifacts"]), ex.get("artifacts"))
    s, _ = call("DELETE", f"/dara/artifacts/{sl['id']}"); check("delete an artifact", s == 200)
    s, _ = call("DELETE", f"/dara/artifacts/{sl['id']}"); check("delete again -> 404", s == 404)
    s, _ = call("PATCH", f"/dara/artifacts/{aid}", {"content": "<script>alert(1)</script> and **markdown**"})
    s, g = call("GET", f"/dara/artifacts/{aid}"); check("content is stored as text exactly as sent", g["content"] == "<script>alert(1)</script> and **markdown**", g)

    # version history (schema version 3): the states an artifact had BEFORE a change
    def versions(i, q=""):
        s, r = call("GET", f"/dara/artifacts/{i}/versions{q}"); return r if s == 200 else None
    def db_count(sql, *args):
        con = sqlite3.connect(os.path.join(data_dir, "dara.db"), timeout=10)
        try: return con.execute(sql, args).fetchone()[0]
        finally: con.close()

    s, v = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Doc", "content": "one"}); vid = v["id"]
    s, _ = call("PATCH", f"/dara/artifacts/{vid}", {"content": "two"})
    check("an ordinary save inside the gap makes no version", s == 200 and versions(vid)["total"] == 0, versions(vid))
    s, e = call("GET", "/dara/artifacts/00000000-0000-4000-8000-000000000000/versions"); check("versions of an unknown artifact -> 404", s == 404, e)
    s, u = call("PATCH", f"/dara/artifacts/{vid}", {"content": "three", "snapshot": "Before Dara edit"})
    lst = versions(vid)
    check("an explicit snapshot stores the state before the change", s == 200 and u["content"] == "three" and lst["total"] == 1 and lst["data"][0]["rev"] == 1
          and lst["data"][0]["label"] == "Before Dara edit" and lst["data"][0]["title"] == "Doc" and lst["data"][0]["chars"] == 3 and lst["data"][0]["snippet"] == "two" and "content" not in lst["data"][0], lst)
    s, g1 = call("GET", f"/dara/artifacts/{vid}/versions/1")
    check("get one version with its content", s == 200 and g1["content"] == "two" and g1["title"] == "Doc" and g1["label"] == "Before Dara edit" and g1["rev"] == 1 and g1["chars"] == 3 and g1["created_at"] > 0, g1)
    check("a version is stamped with when that state was last saved", g1["created_at"] <= u["updated_at"], (g1, u["updated_at"]))
    s, e = call("GET", f"/dara/artifacts/{vid}/versions/99"); check("unknown version -> 404", s == 404, e)
    s, e = call("GET", f"/dara/artifacts/{vid}/versions/abc"); check("non-numeric version -> 404", s == 404, e)
    call("PATCH", f"/dara/artifacts/{vid}", {"content": "three", "snapshot": "no change"})
    check("a snapshot is skipped when the patch changes nothing", versions(vid)["total"] == 1, versions(vid))
    s, e = call("PATCH", f"/dara/artifacts/{vid}", {"content": "x", "snapshot": "y" * 121}); check("snapshot label over 120 characters -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{vid}", {"content": "x", "snapshot": 5}); check("non-string snapshot -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{vid}", {"title": 7, "snapshot": "z"}); check("a bad field refuses the whole patch", s == 400, e)
    s, g = call("GET", f"/dara/artifacts/{vid}"); check("a refused patch changed and stored nothing", g["content"] == "three" and versions(vid)["total"] == 1, g)

    # no duplicates: the state being replaced equals the newest version
    call("PATCH", f"/dara/artifacts/{vid}", {"content": "two"})   # three -> two, inside the gap: no version (the newest version already holds "two")
    s, _ = call("PATCH", f"/dara/artifacts/{vid}", {"content": "U", "snapshot": "dup"})
    check("a snapshot identical to the newest version is not stored again", s == 200 and versions(vid)["total"] == 1, versions(vid))

    # automatic snapshot: the first content edit after a pause longer than the gap (the gap runs from the last save, a title save included)
    s, _ = call("PATCH", f"/dara/artifacts/{vid}", {"title": "Doc renamed"})
    check("a title-only edit makes no automatic version", versions(vid)["total"] == 1, versions(vid))
    time.sleep(1.7)
    s, _ = call("PATCH", f"/dara/artifacts/{vid}", {"content": "after the pause"})
    lst = versions(vid)
    check("the first content edit after a long pause snapshots the earlier text (label empty)", lst["total"] == 2 and lst["data"][0]["rev"] == 2 and lst["data"][0]["label"] == ""
          and lst["data"][0]["title"] == "Doc renamed" and lst["data"][0]["snippet"] == "U", lst)
    s, _ = call("PATCH", f"/dara/artifacts/{vid}", {"content": "right after"})
    check("a following edit inside the gap makes none", versions(vid)["total"] == 2, versions(vid))
    lst = versions(vid, "?limit=1&offset=1")
    check("versions list paging, newest first", lst["total"] == 2 and lst["limit"] == 1 and lst["offset"] == 1 and [x["rev"] for x in lst["data"]] == [1], lst)

    # an empty document is not worth a version
    s, em = call("POST", "/dara/artifacts", {"kind": "doc"})
    call("PATCH", f"/dara/artifacts/{em['id']}", {"content": "first words", "snapshot": "Before Dara edit"})
    check("an empty state is never stored", versions(em["id"])["total"] == 0, versions(em["id"]))

    # restore
    s, r = call("POST", f"/dara/artifacts/{vid}/restore", {"rev": 1})
    lst = versions(vid)
    check("restore returns the artifact with that version's text and title", s == 200 and r["id"] == vid and r["content"] == "two" and r["title"] == "Doc" and r["updated_at"] >= u["updated_at"], r)
    check("restore keeps the replaced state as 'Before restore'", lst["total"] == 3 and lst["data"][0]["label"] == "Before restore" and lst["data"][0]["snippet"] == "right after" and lst["data"][0]["rev"] == 3, lst)
    s, g = call("GET", f"/dara/artifacts/{vid}"); check("restored text is what a normal get returns", g["content"] == "two" and g["title"] == "Doc", g)
    s, e = call("POST", f"/dara/artifacts/{vid}/restore", {"rev": 99}); check("restore an unknown version -> 404", s == 404, e)
    s, e = call("POST", f"/dara/artifacts/{vid}/restore", {}); check("restore without rev -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{vid}/restore", {"rev": "1"}); check("restore with a non-integer rev -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{vid}/restore", {"rev": 1.5}); check("restore with a fractional rev -> 400", s == 400, e)
    s, e = call("POST", "/dara/artifacts/00000000-0000-4000-8000-000000000000/restore", {"rev": 1}); check("restore on an unknown artifact -> 404", s == 404, e)
    s, r = call("POST", f"/dara/artifacts/{vid}/restore", {"rev": 3})
    lst = versions(vid)
    check("restore twice walks back and forth (the undo of a restore is a version too)", s == 200 and r["content"] == "right after" and lst["total"] == 4 and lst["data"][0]["snippet"] == "two", lst)

    # the cap (5 in this test): the oldest versions go first
    s, cp = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Cap", "content": "c0"}); cid2 = cp["id"]
    for i in range(1, 9): call("PATCH", f"/dara/artifacts/{cid2}", {"content": f"c{i}", "snapshot": f"s{i}"})
    lst = versions(cid2)
    check("at most the cap (5) versions are kept, the oldest dropped", lst["total"] == 5 and [x["rev"] for x in lst["data"]] == [8, 7, 6, 5, 4] and lst["data"][0]["snippet"] == "c7", lst)
    s, _ = call("GET", f"/dara/artifacts/{cid2}/versions/3"); check("a version past the cap is gone", s == 404)

    # versions leave with their artifact
    check("versions are stored in the table", db_count("SELECT COUNT(*) FROM artifact_versions WHERE artifact_id=?", cid2) == 5)
    s, _ = call("DELETE", f"/dara/artifacts/{cid2}")
    check("deleting an artifact deletes its versions", s == 200 and db_count("SELECT COUNT(*) FROM artifact_versions WHERE artifact_id=?", cid2) == 0)
    s, e = call("GET", f"/dara/artifacts/{cid2}/versions"); check("versions of a deleted artifact -> 404", s == 404, e)
    check("other artifacts keep theirs", db_count("SELECT COUNT(*) FROM artifact_versions WHERE artifact_id=?", vid) == 4)
    s, ex = call("GET", "/dara/data/export"); check("export does not carry version history", all("versions" not in a for a in ex["artifacts"]), ex["artifacts"][:1])


    # ---- chat, comments, images and video that belong to an artifact (schema version 4, docs/ui/10-artifact-collab.md) ----
    NOPE = "00000000-0000-4000-8000-000000000000"
    s, ca = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Collab", "content": "hello world, hello again"}); A = ca["id"]
    s, cb = call("POST", "/dara/artifacts", {"kind": "doc", "title": "Other"}); B = cb["id"]

    # chat
    s, r = call("PUT", f"/dara/artifacts/{A}/chat/m-1", {"data": {"role": "user", "text": "write a plan", "n": [1, 2]}})
    check("chat: put a new message", s == 200 and r["id"] == "m-1" and r["seq"] == 1 and r["data"] == {"role": "user", "text": "write a plan", "n": [1, 2]} and r["created_at"] > 0 and r["created_at"] == r["updated_at"], r)
    s, r2 = call("PUT", f"/dara/artifacts/{A}/chat/m_2", {"data": {"role": "assistant", "text": "done"}})
    check("chat: the next message gets the next seq", s == 200 and r2["seq"] == 2, r2)
    time.sleep(0.01)
    s, r3 = call("PUT", f"/dara/artifacts/{A}/chat/m-1", {"data": {"role": "user", "text": "write a longer plan"}})
    check("chat: putting an existing id keeps its seq and replaces its data", s == 200 and r3["seq"] == 1 and r3["data"] == {"role": "user", "text": "write a longer plan"} and r3["created_at"] == r["created_at"] and r3["updated_at"] >= r["updated_at"], r3)
    s, lst = call("GET", f"/dara/artifacts/{A}/chat")
    check("chat: list is in seq order with data as objects", s == 200 and lst["object"] == "list" and lst["total"] == 2 and [x["id"] for x in lst["data"]] == ["m-1", "m_2"] and isinstance(lst["data"][0]["data"], dict), lst)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/bad%20id", {"data": {}}); check("chat: a message id with a space -> 400", s == 400, e)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/" + "x" * 65, {"data": {}}); check("chat: a message id over 64 characters -> 400", s == 400, e)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/ok", {"data": [1]}); check("chat: data that is not an object -> 400", s == 400, e)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/ok", {"data": "text"}); check("chat: string data -> 400", s == 400, e)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/ok", {}); check("chat: missing data -> 400", s == 400, e)
    s, e = call("PUT", f"/dara/artifacts/{A}/chat/ok", {"data": {"t": "x" * 210000}}); check("chat: data over 200 KB -> 400", s == 400, str(e)[:100])
    s, e = call("PUT", f"/dara/artifacts/{NOPE}/chat/ok", {"data": {}}); check("chat: put on an unknown artifact -> 404", s == 404, e)
    s, e = call("GET", f"/dara/artifacts/{NOPE}/chat"); check("chat: list on an unknown artifact -> 404", s == 404, e)
    s, e = call("DELETE", f"/dara/artifacts/{NOPE}/chat"); check("chat: delete on an unknown artifact -> 404", s == 404, e)
    s, lst = call("GET", f"/dara/artifacts/{B}/chat"); check("chat: another artifact has its own, empty", s == 200 and lst["total"] == 0 and lst["data"] == [], lst)
    for i in range(3, 9): call("PUT", f"/dara/artifacts/{A}/chat/m{i}", {"data": {"i": i}})   # the cap in this test is 6
    s, lst = call("GET", f"/dara/artifacts/{A}/chat")
    check("chat: at most the cap (6) messages are kept, the lowest seq dropped", lst["total"] == 6 and [x["seq"] for x in lst["data"]] == [3, 4, 5, 6, 7, 8] and lst["data"][0]["id"] == "m3", lst)
    s, r = call("DELETE", f"/dara/artifacts/{A}/chat"); check("chat: clear returns how many were deleted", s == 200 and r == {"deleted": 6}, r)
    s, lst = call("GET", f"/dara/artifacts/{A}/chat"); check("chat: empty after clear", lst["total"] == 0, lst)
    s, r = call("PUT", f"/dara/artifacts/{A}/chat/again", {"data": {"x": 1}}); check("chat: seq starts again after a clear", s == 200 and r["seq"] == 1, r)
    call("PUT", f"/dara/artifacts/{A}/chat/again2", {"data": {"x": 2}})

    # comments
    s, c1 = call("POST", f"/dara/artifacts/{A}/comments", {"body": "Is this right?", "quote": "hello", "nth": 1, "author": "dara"})
    check("comment: create a root comment", s == 201 and len(c1["id"]) == 36 and c1["parent_id"] is None and c1["author"] == "dara" and c1["body"] == "Is this right?" and c1["quote"] == "hello" and c1["nth"] == 1
          and c1["resolved"] is False and c1["created_at"] == c1["updated_at"], c1)
    s, c2 = call("POST", f"/dara/artifacts/{A}/comments", {"body": "Yes, keep it", "parent_id": c1["id"]})
    check("comment: reply (author defaults to user, no quote)", s == 201 and c2["parent_id"] == c1["id"] and c2["author"] == "user" and c2["quote"] == "" and c2["nth"] == 0, c2)
    s, c3 = call("POST", f"/dara/artifacts/{A}/comments", {"body": "About the whole document"})
    check("comment: no quote means the whole document", s == 201 and c3["quote"] == "" and c3["nth"] == 0, c3)
    mine = "33333333-3333-4333-8333-333333333333"
    s, c4 = call("POST", f"/dara/artifacts/{A}/comments", {"id": mine, "body": "own id"}); check("comment: the app can choose the id", s == 201 and c4["id"] == mine, c4)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"id": mine, "body": "again"}); check("comment: a duplicate id -> 409", s == 409, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"id": "nope", "body": "x"}); check("comment: a non-UUID id -> 400", s == 400, e)
    s, lst = call("GET", f"/dara/artifacts/{A}/comments")
    check("comment: list is flat, oldest first", s == 200 and lst["total"] == 4 and [x["id"] for x in lst["data"]] == [c1["id"], c2["id"], c3["id"], mine], lst)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "reply to a reply", "parent_id": c2["id"]}); check("comment: a reply to a reply -> 400 (threads are one level deep)", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "parent_id": NOPE}); check("comment: a reply to an unknown comment -> 404", s == 404, e)
    s, e = call("POST", f"/dara/artifacts/{B}/comments", {"body": "x", "parent_id": c1["id"]}); check("comment: a reply to a comment of another document -> 404", s == 404, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "parent_id": c1["id"], "quote": "hello"}); check("comment: a reply with a quote -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "author": "robot"}); check("comment: a bad author -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": ""}); check("comment: an empty body -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "  \n "}); check("comment: a blank body -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {}); check("comment: a missing body -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": 5}); check("comment: a non-string body -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x" * 8001}); check("comment: a body over 8000 characters -> 400", s == 400, str(e)[:100])
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "\u00e9" * 8000}); check("comment: 8000 characters is fine even when each takes two bytes", s == 201, str(e)[:100])
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "\u00e9" * 8001}); check("comment: limits count characters, not bytes", s == 400, str(e)[:100])
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "quote": "q" * 501}); check("comment: a quote over 500 characters -> 400", s == 400, str(e)[:100])
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "nth": -1}); check("comment: a negative nth -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{A}/comments", {"body": "x", "nth": "1"}); check("comment: a non-integer nth -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{NOPE}/comments", {"body": "x"}); check("comment: on an unknown artifact -> 404", s == 404, e)
    s, e = call("GET", f"/dara/artifacts/{NOPE}/comments"); check("comment: list on an unknown artifact -> 404", s == 404, e)
    time.sleep(0.01)
    s, u1 = call("PATCH", f"/dara/artifacts/{A}/comments/{c1['id']}", {"resolved": True, "body": "Is this right? (edited)", "quote": "hello world", "nth": 0})
    check("comment: patch resolved, body, quote and nth on a root", s == 200 and u1["resolved"] is True and u1["body"] == "Is this right? (edited)" and u1["quote"] == "hello world" and u1["nth"] == 0 and u1["updated_at"] > c1["updated_at"], u1)
    s, u2 = call("PATCH", f"/dara/artifacts/{A}/comments/{c2['id']}", {"body": "Yes, keep it as it is"}); check("comment: a reply's text can be edited", s == 200 and u2["body"] == "Yes, keep it as it is", u2)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{c2['id']}", {"resolved": True}); check("comment: resolving a reply -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{c2['id']}", {"quote": "x"}); check("comment: anchoring a reply -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{c1['id']}", {}); check("comment: an empty patch -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{c1['id']}", {"resolved": "yes"}); check("comment: resolved must be true or false", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{c1['id']}", {"body": " "}); check("comment: a blank body in a patch -> 400", s == 400, e)
    s, e = call("PATCH", f"/dara/artifacts/{A}/comments/{NOPE}", {"resolved": True}); check("comment: patch an unknown comment -> 404", s == 404, e)
    s, e = call("PATCH", f"/dara/artifacts/{B}/comments/{c1['id']}", {"resolved": False}); check("comment: patch through another document -> 404", s == 404, e)
    s, r = call("DELETE", f"/dara/artifacts/{B}/comments/{c1['id']}"); check("comment: delete through another document -> 404", s == 404, r)
    s, r = call("DELETE", f"/dara/artifacts/{A}/comments/{c2['id']}"); check("comment: delete a reply", s == 200 and r == {"deleted": 1}, r)
    s, c5 = call("POST", f"/dara/artifacts/{A}/comments", {"body": "again", "parent_id": c1["id"]})
    s, c6 = call("POST", f"/dara/artifacts/{A}/comments", {"body": "and again", "parent_id": c1["id"]})
    s, r = call("DELETE", f"/dara/artifacts/{A}/comments/{c1['id']}"); check("comment: deleting a root takes its replies with it (and counts them)", s == 200 and r == {"deleted": 3}, r)
    s, lst = call("GET", f"/dara/artifacts/{A}/comments")
    check("comment: the other threads stay", all(x["id"] not in (c1["id"], c5["id"], c6["id"]) for x in lst["data"]) and any(x["id"] == c3["id"] for x in lst["data"]), lst["total"])
    s, r = call("DELETE", f"/dara/artifacts/{A}/comments/{c1['id']}"); check("comment: delete again -> 404", s == 404, r)
    # the caps: 100 replies per thread, 1000 comments per document
    s, tb = call("POST", f"/dara/artifacts/{B}/comments", {"body": "thread"}); tid = tb["id"]
    codes = [call("POST", f"/dara/artifacts/{B}/comments", {"body": f"r{i}", "parent_id": tid})[0] for i in range(100)]
    s, e = call("POST", f"/dara/artifacts/{B}/comments", {"body": "one too many", "parent_id": tid})
    check("comment: 100 replies fit in a thread, the 101st -> 409", all(c == 201 for c in codes) and s == 409, (set(codes), s, e))
    s, tc = call("POST", "/dara/artifacts", {"kind": "doc"}); C = tc["id"]
    codes = [call("POST", f"/dara/artifacts/{C}/comments", {"body": f"c{i}"})[0] for i in range(1000)]
    s, e = call("POST", f"/dara/artifacts/{C}/comments", {"body": "one too many"})
    check("comment: 1000 comments fit in a document, the 1001st -> 409", all(c == 201 for c in codes) and s == 409, (set(codes), s, e))
    s, st = call("GET", "/dara/store"); check("store info counts comments", st["artifact_comments"] >= 1100, st)
    call("DELETE", f"/dara/artifacts/{C}")

    # images and video
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 200), {"Content-Type": "image/png", "X-Dara-Name": urllib.request.quote("my pic \u00e9.png")})
    a1 = json.loads(b)
    check("asset: upload an image (raw body, url-encoded name)", s == 201 and len(a1["id"]) == 36 and a1["name"] == "my pic \u00e9.png" and a1["mime"] == "image/png" and a1["size"] == 200 and a1["created_at"] > 0 and "data" not in a1, a1)
    s, h, b = call_raw("GET", f"/dara/artifacts/{A}/assets/{a1['id']}")
    check("asset: the bytes come back as uploaded", s == 200 and b == blob("png", 200), (s, len(b)))
    hl = {k.lower(): v for k, v in h.items()}
    check("asset: served with its type, a long private cache and nosniff", hl.get("content-type", "").startswith("image/png") and hl.get("cache-control") == "private, max-age=31536000, immutable" and hl.get("x-content-type-options") == "nosniff", hl)
    s, h, b = call_raw("GET", f"/dara/artifacts/{A}/assets/{a1['id']}", key=None); check("asset: fetching without the API key -> 401", s == 401, s)
    s, h, b = call_raw("GET", f"/dara/artifacts/{B}/assets/{a1['id']}"); check("asset: fetching through another document -> 404", s == 404, s)
    s, h, b = call_raw("GET", f"/dara/artifacts/{A}/assets/{NOPE}"); check("asset: an unknown asset -> 404", s == 404, s)
    got = {}
    for kind, mime in MIME.items():
        s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob(kind, 120), {"Content-Type": mime, "X-Dara-Name": kind + ".bin"})
        got[kind] = (s, json.loads(b))
    check("asset: PNG, JPEG, GIF, WebP, MP4 and WebM are accepted", all(v[0] == 201 and v[1]["mime"] == MIME[k] for k, v in got.items()), {k: v[0] for k, v in got.items()})
    s, h, b = call_raw("POST", f"/dara/artifacts/{B}/assets", blob("png", 100), {"Content-Type": "IMAGE/PNG; charset=binary"}); bj = json.loads(b)
    check("asset: the Content-Type is read without its parameters and in any case; the name defaults to 'file'", s == 201 and bj["mime"] == "image/png" and bj["name"] == "file", (s, bj))
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", b"<svg xmlns='http://www.w3.org/2000/svg'/>", {"Content-Type": "image/svg+xml"}); check("asset: SVG -> 415", s == 415, (s, b[:80]))
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", b"hello", {"Content-Type": "text/plain"}); check("asset: another type -> 415", s == 415, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 100), {}); check("asset: no Content-Type -> 415", s == 415, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("jpeg", 100), {"Content-Type": "image/png"}); check("asset: bytes that are not the declared type (a JPEG sent as PNG) -> 415", s == 415, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 100), {"Content-Type": "video/mp4"}); check("asset: a PNG sent as MP4 -> 415", s == 415, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", b"\x89PNG", {"Content-Type": "image/png"}); check("asset: a truncated header -> 415", s == 415, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", b"", {"Content-Type": "image/png"}); check("asset: an empty body -> 400", s == 400, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 3001), {"Content-Type": "image/png"}); check("asset: over the per-file limit (3000 bytes in this test) -> 413", s == 413, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 3000), {"Content-Type": "image/png", "X-Dara-Name": "n" * 201}); check("asset: a file name over 200 characters -> 400", s == 400, s)
    s, h, b = call_raw("POST", f"/dara/artifacts/{NOPE}/assets", blob("png", 100), {"Content-Type": "image/png"}); check("asset: upload to an unknown artifact -> 404", s == 404, s)
    try: s = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 100), {"Content-Type": "image/png"}, key=None)[0]
    except OSError: s = 401   # the server answers 401 before it reads the body and closes the connection; the client may see the reset first
    check("asset: upload without the API key is refused (401)", s == 401, s)
    s, lst = call("GET", f"/dara/artifacts/{A}/assets")
    check("asset: list is metadata only", s == 200 and lst["total"] == 7 and all(set(x) == {"id", "name", "mime", "size", "created_at"} for x in lst["data"]), lst)
    # per-artifact space: 5000 bytes in this test; the document holds 200 + 6 x 120 = 920 so far
    ok1 = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 2900), {"Content-Type": "image/png"})[0]
    ok2 = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 1000), {"Content-Type": "image/png"})[0]
    over = call_raw("POST", f"/dara/artifacts/{A}/assets", blob("png", 500), {"Content-Type": "image/png"})[0]
    check("asset: the space per document is limited (413 once it is used up)", ok1 == 201 and ok2 == 201 and over == 413, (ok1, ok2, over))
    s, r = call("DELETE", f"/dara/artifacts/{A}/assets/{a1['id']}"); check("asset: delete one", s == 200 and r == {"deleted": True}, r)
    s, r = call("DELETE", f"/dara/artifacts/{A}/assets/{a1['id']}"); check("asset: delete again -> 404", s == 404, r)
    s, r = call("DELETE", f"/dara/artifacts/{B}/assets/{got['png'][1]['id']}"); check("asset: delete through another document -> 404", s == 404, r)
    check("asset: the stored bytes are in the table", db_count("SELECT COUNT(*) FROM artifact_assets WHERE artifact_id=?", A) == 8 and db_count("SELECT SUM(size) FROM artifact_assets WHERE artifact_id=?", A) == 6 * 120 + 2900 + 1000)
    s, st = call("GET", "/dara/store"); check("store info counts assets", st["artifact_assets"] >= 9, st)

    # export: chat and comments are included, images and video only as metadata, no versions
    s, ex = call("GET", "/dara/data/export")
    ea = next((x for x in ex["artifacts"] if x["id"] == A), {})
    check("export: an artifact carries its chat and comments, and its assets as metadata only",
          [m["id"] for m in ea.get("chat", [])] == ["again", "again2"] and any(c["id"] == c3["id"] for c in ea.get("comments", [])) and len(ea.get("assets", [])) == 8
          and all("data" not in x for x in ea["assets"]) and "versions" not in ea, {k: (v if k != "content" else "...") for k, v in ea.items()})

    # duplicate: a copy with its own assets (ids rewritten in the text), without versions, chat or comments
    call_raw("POST", f"/dara/artifacts/{B}/assets", blob("jpeg", 150), {"Content-Type": "image/jpeg", "X-Dara-Name": "first.jpg"})
    s, h, b = call_raw("POST", f"/dara/artifacts/{B}/assets", blob("webm", 160), {"Content-Type": "video/webm", "X-Dara-Name": "clip.webm"}); v2 = json.loads(b)
    s, lst = call("GET", f"/dara/artifacts/{B}/assets"); old_ids = [x["id"] for x in lst["data"]]
    i1 = next(x for x in lst["data"] if x["name"] == "first.jpg")["id"]
    text = f"intro ![first](dara-asset:{i1}) middle ![clip](dara-asset:{v2['id']} \"video\") again ![first](dara-asset:{i1}) and dara-asset:{NOPE} stays"
    call("PATCH", f"/dara/artifacts/{B}", {"title": "Source", "content": text, "snapshot": "keep this"})
    call("PATCH", f"/dara/artifacts/{B}", {"content": text + " more", "snapshot": "keep this too"})
    call("PUT", f"/dara/artifacts/{B}/chat/x1", {"data": {"a": 1}})
    s, dup = call("POST", f"/dara/artifacts/{B}/duplicate", {})
    check("duplicate: a full copy with a new id and '(copy)' added to the title", s == 201 and dup["id"] != B and dup["title"] == "Source (copy)" and dup["kind"] == "doc" and dup["pinned"] is False, dup)
    s, lst2 = call("GET", f"/dara/artifacts/{dup['id']}/assets"); new_ids = [x["id"] for x in lst2["data"]]
    check("duplicate: the copy has its own assets with the same names, types and sizes", s == 200 and lst2["total"] == len(old_ids) and not set(new_ids) & set(old_ids)
          and sorted((x["name"], x["mime"], x["size"]) for x in lst2["data"]) == sorted((x["name"], x["mime"], x["size"]) for x in lst["data"]), lst2)
    nf = next(x for x in lst2["data"] if x["name"] == "first.jpg")["id"]; nc = next(x for x in lst2["data"] if x["name"] == "clip.webm")["id"]
    expect = text.replace(i1, nf).replace(v2["id"], nc) + " more"
    check("duplicate: every dara-asset id in the text points at the copy's asset (an unknown id is left alone)", dup["content"] == expect and dup["content"].count(nf) == 2 and i1 not in dup["content"] and v2["id"] not in dup["content"] and NOPE in dup["content"], dup["content"])
    s, h, b = call_raw("GET", f"/dara/artifacts/{dup['id']}/assets/{nf}"); check("duplicate: the copied bytes are identical", s == 200 and b == blob("jpeg", 150), (s, len(b)))
    check("duplicate: no versions, chat or comments are copied", call("GET", f"/dara/artifacts/{dup['id']}/versions")[1]["total"] == 0 and call("GET", f"/dara/artifacts/{dup['id']}/chat")[1]["total"] == 0 and call("GET", f"/dara/artifacts/{dup['id']}/comments")[1]["total"] == 0)
    s, g = call("GET", f"/dara/artifacts/{B}"); check("duplicate: the original is unchanged", g["content"] == text + " more" and call("GET", f"/dara/artifacts/{B}/assets")[1]["total"] == len(old_ids) and g["title"] == "Source", g["title"])
    myid = "44444444-4444-4444-8444-444444444444"
    s, d2 = call("POST", f"/dara/artifacts/{B}/duplicate", {"id": myid, "title": "Chosen"}); check("duplicate: the app can choose the id and the title", s == 201 and d2["id"] == myid and d2["title"] == "Chosen", d2)
    s, e = call("POST", f"/dara/artifacts/{B}/duplicate", {"id": myid}); check("duplicate: an id that exists -> 409", s == 409, e)
    s, e = call("POST", f"/dara/artifacts/{B}/duplicate", {"id": "nope"}); check("duplicate: a non-UUID id -> 400", s == 400, e)
    s, e = call("POST", f"/dara/artifacts/{NOPE}/duplicate", {}); check("duplicate: an unknown artifact -> 404", s == 404, e)
    s, d3 = call("POST", f"/dara/artifacts/{B}/duplicate", {"title": ""}); check("duplicate: an explicitly empty title is kept", s == 201 and d3["title"] == "", d3)
    s, em2 = call("POST", "/dara/artifacts", {"kind": "doc"}); s, d4 = call("POST", f"/dara/artifacts/{em2['id']}/duplicate", {}); check("duplicate: an untitled document stays untitled", s == 201 and d4["title"] == "", d4)

    # everything leaves with its artifact (checked in the file itself)
    child_sql = ("SELECT (SELECT COUNT(*) FROM artifact_chat WHERE artifact_id=?1) + (SELECT COUNT(*) FROM artifact_comments WHERE artifact_id=?1) + "
                 "(SELECT COUNT(*) FROM artifact_assets WHERE artifact_id=?1) + (SELECT COUNT(*) FROM artifact_versions WHERE artifact_id=?1)")
    check("cascade: the source has chat, comments, assets and versions in the file", db_count(child_sql.replace("?1", "'" + B + "'")) > 0)
    s, r = call("DELETE", f"/dara/artifacts/{B}"); check("cascade: delete the source artifact", s == 200, r)
    check("cascade: its chat, comments, assets and versions are gone from the file", db_count(child_sql.replace("?1", "'" + B + "'")) == 0)
    check("cascade: the copy keeps its own assets", db_count("SELECT COUNT(*) FROM artifact_assets WHERE artifact_id=?", dup["id"]) == len(old_ids))
    check("cascade: the other document keeps its things", db_count("SELECT COUNT(*) FROM artifact_assets WHERE artifact_id=?", A) == 8 and db_count("SELECT COUNT(*) FROM artifact_chat WHERE artifact_id=?", A) == 2)
    s, e = call("GET", f"/dara/artifacts/{B}/comments"); check("cascade: the deleted artifact's comments -> 404", s == 404, e)
    s, r = call("DELETE", f"/dara/artifacts/{dup['id']}"); check("cascade: delete the copy", s == 200 and db_count("SELECT COUNT(*) FROM artifact_assets WHERE artifact_id=?", dup["id"]) == 0)

    # delete all clears them too (checked after the next block)

    # delete all needs the confirmation
    call("POST", "/dara/conversations", {"title": "x"})
    s, e = call("DELETE", "/dara/data", {}); check("delete-all without confirm -> 400", s == 400, e)
    s, r = call("DELETE", "/dara/data", {"confirm": "delete-all-data"})
    s2, lst = call("GET", "/dara/conversations")
    s3, allc = call("GET", "/dara/model-configs")
    check("delete all clears conversations and model configs", s == 200 and lst["total"] == 0 and allc == {}, (r, lst, allc))
    s4, arts = call("GET", "/dara/artifacts"); check("delete all clears artifacts", s4 == 200 and arts["total"] == 0, arts)
    check("delete all clears version history", db_count("SELECT COUNT(*) FROM artifact_versions") == 0)
    check("delete all clears chat, comments and assets", db_count("SELECT (SELECT COUNT(*) FROM artifact_chat) + (SELECT COUNT(*) FROM artifact_comments) + (SELECT COUNT(*) FROM artifact_assets)") == 0)
    s, st = call("GET", "/dara/settings"); check("settings kept by delete all", st["theme"] == "dark", st)

    # migration from schema version 1 (a database written by the previous build is upgraded in place, after a backup copy)
    prev = os.path.join(os.path.dirname(EXE), "dara_prev.exe")
    if os.path.exists(prev):
        stop(p)   # the migration test uses the same port
        mdir = tempfile.mkdtemp(prefix="dara_store_mig_")
        old_exe, EXE = EXE, prev
        pm = start(mdir)
        s, c = call("POST", "/dara/conversations", {"title": "kept across the upgrade"})
        s, info1 = call("GET", "/dara/store"); check("previous build writes schema 1", info1.get("schema_version") == 1 and "artifacts" not in info1, info1)
        stop(pm)
        EXE = old_exe
        pm = start(mdir)
        s, info2 = call("GET", "/dara/store")
        check("new build upgrades the old database to schema 4", s == 200 and info2["schema_version"] == 4 and info2["artifacts"] == 0 and info2["conversations"] == 1, info2)
        check("a backup of the version 1, 2 and 3 file was kept", all(os.path.exists(os.path.join(mdir, "dara.db.bak-v" + n)) for n in "123"), os.listdir(mdir))
        s, nc = call("POST", "/dara/artifacts", {"kind": "doc", "title": "after upgrade"}); check("artifacts work on the upgraded database", s == 201, nc)
        stop(pm)
        shutil.rmtree(mdir, ignore_errors=True)
        p = start(data_dir)
    else:
        print("skip  migration test (build\\cuda\\dara_prev.exe, a schema 1 build, not found)")


    # migration from schema version 2 (a database written by the build with artifacts but no version history). Built here from the
    # shipped SQL, exactly what that build writes: schema.sql, then 0002_artifacts.sql, user_version 2.
    stop(p)
    m2 = tempfile.mkdtemp(prefix="dara_store_mig2_")
    db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "ui", "db")
    OLD = "11111111-1111-4111-8111-111111111111"
    try:
        con = sqlite3.connect(os.path.join(m2, "dara.db"))
        con.executescript(open(os.path.join(db_dir, "schema.sql"), encoding="utf8").read())
        con.execute("PRAGMA user_version=1")
        con.executescript(open(os.path.join(db_dir, "0002_artifacts.sql"), encoding="utf8").read())
        con.execute("INSERT INTO artifacts(id,kind,title,content,pinned,created_at,updated_at) VALUES(?,?,?,?,0,1000,1000)", (OLD, "doc", "Old doc", "old text"))
        con.commit(); con.execute("PRAGMA user_version=2"); con.commit(); con.close()
        built = True
    except sqlite3.Error as ex:
        print("skip  schema 2 migration test (this Python's SQLite cannot build the schema:", ex, ")")
        built = False
    if built:
        pm = start(m2)
        s, info3 = call("GET", "/dara/store")
        check("a schema 2 database is upgraded to schema 4 in place", s == 200 and info3["schema_version"] == 4 and info3["artifacts"] == 1, info3)
        check("a backup of the version 2 and 3 file was kept", os.path.exists(os.path.join(m2, "dara.db.bak-v2")) and os.path.exists(os.path.join(m2, "dara.db.bak-v3")) and not os.path.exists(os.path.join(m2, "dara.db.bak-v1")), os.listdir(m2))
        s, g = call("GET", f"/dara/artifacts/{OLD}"); check("the old document survived the upgrade", s == 200 and g["content"] == "old text", g)
        s, r = call("GET", f"/dara/artifacts/{OLD}/versions"); check("the old document has an empty history that works", s == 200 and r["total"] == 0 and r["data"] == [], r)
        s, _ = call("PATCH", f"/dara/artifacts/{OLD}", {"content": "new text", "snapshot": "Before edit"})
        s, r = call("GET", f"/dara/artifacts/{OLD}/versions"); check("versions work on the upgraded database", r["total"] == 1 and r["data"][0]["snippet"] == "old text", r)
        stop(pm)
    shutil.rmtree(m2, ignore_errors=True)
    p = start(data_dir)

    # migration from schema version 3 (the build with version history but no chat, comments or assets): upgraded in place, after a backup copy
    ver3 = os.path.join(os.path.dirname(EXE), "dara_ver.exe")
    if os.path.exists(ver3):
        stop(p)
        m3 = tempfile.mkdtemp(prefix="dara_store_mig3_")
        old_exe, EXE = EXE, ver3
        pm = start(m3)
        s, a3 = call("POST", "/dara/artifacts", {"kind": "doc", "title": "From schema 3", "content": "kept text"})
        call("PATCH", f"/dara/artifacts/{a3['id']}", {"content": "newer text", "snapshot": "Before edit"})
        s, info4 = call("GET", "/dara/store"); check("the schema 3 build writes schema 3", info4.get("schema_version") == 3 and "artifact_comments" not in info4, info4)
        stop(pm)
        EXE = old_exe
        pm = start(m3)
        s, info5 = call("GET", "/dara/store")
        check("a schema 3 database is upgraded to schema 4 in place", s == 200 and info5["schema_version"] == 4 and info5["artifacts"] == 1 and info5["artifact_comments"] == 0 and info5["artifact_assets"] == 0, info5)
        check("a backup of the version 3 file was kept before the upgrade", os.path.exists(os.path.join(m3, "dara.db.bak-v3")) and not os.path.exists(os.path.join(m3, "dara.db.bak-v4")), os.listdir(m3))
        s, g = call("GET", f"/dara/artifacts/{a3['id']}"); check("the document survived the upgrade", s == 200 and g["content"] == "newer text" and g["title"] == "From schema 3", g)
        s, r = call("GET", f"/dara/artifacts/{a3['id']}/versions"); check("its version history survived the upgrade", r["total"] == 1 and r["data"][0]["snippet"] == "kept text", r)
        s, r = call("POST", f"/dara/artifacts/{a3['id']}/comments", {"body": "works after the upgrade", "quote": "newer"}); check("comments work on the upgraded database", s == 201, r)
        s, r = call("PUT", f"/dara/artifacts/{a3['id']}/chat/m1", {"data": {"ok": True}}); check("chat works on the upgraded database", s == 200, r)
        s, h, b = call_raw("POST", f"/dara/artifacts/{a3['id']}/assets", blob("png", 90), {"Content-Type": "image/png"}); check("assets work on the upgraded database", s == 201, b[:100])
        stop(pm)
        shutil.rmtree(m3, ignore_errors=True)
        p = start(data_dir)
    else:
        print("skip  schema 3 migration test (build\\cuda\\dara_ver.exe, a schema 3 build, not found)")

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
