# Local database

All app data (chat history, settings, per-model configuration) lives in one SQLite database that belongs to the **Dara server** (`dara.exe serve`), next to the endpoints that use it. The desktop app, the command line and any other client use the HTTP API; nothing but the server opens the file. Schema: [db/schema.sql](db/schema.sql) is schema version 1 and never changes; later versions arrive as numbered migrations next to it (`db/0002_artifacts.sql`, `db/0003_artifact_versions.sql`, ...). `tools/gen_store_schema.py` embeds both in `src/serve/store_schema.h`. Code: `src/serve/store.h`, `src/serve/store.cpp`. Test: `tests/store_api_test.py` (116 checks, no GPU needed; the schema 1 upgrade test needs `build\cuda\dara_prev.exe`, a build from before artifacts, and is skipped without it; the schema 2 upgrade test builds its database from the shipped SQL and needs nothing). The test starts the server with `DARA_VERSION_GAP_MS=1500` and `DARA_VERSION_CAP=5` (test-only overrides of the version-history limits).

## Where it lives

`%LOCALAPPDATA%\Dara\dara.db`: the same per-user Dara data folder as the models and `calibration.json`. Not the install folder: Windows installs go under `Program Files`, which a normal user cannot write to, and an uninstall or update would delete the data. Override with `--data-dir DIR` or `DARA_DATA_DIR`. Images are files in `<data dir>\attachments\`; the database stores only their names. Because the server owns the file, the folder follows the user's profile and its default Windows permissions (readable by that user only).

Settings can move the *models* folder; the database and attachments stay in the data folder.

`--no-store` runs the server without the database (the `/dara/` store endpoints then answer 503 `store_unavailable`). If the database cannot be opened, the server still starts and chat works; only the history endpoints answer 503, and `/dara/store` reports why.

## Tables (summary)

| Table | Holds |
| --- | --- |
| `settings` | key/value JSON: theme, port, models folder, LAN toggle, defaults for K/V format, Flash, VRAM reserve, vision, sampling, system prompt, tray and autostart |
| `conversations` | id, title, model, system prompt, per-conversation parameters, pinned, timestamps |
| `messages` | conversation, position, role, content (text/Markdown), finish reason, error, stats JSON (usage, timings, context, speculative), model, time |
| `attachments` | image files of a message: mime, file name, size |
| `model_configs` | what the user chose per model (K/V format, Flash, vision, context); no row means Auto |
| `messages_fts` | full-text search index over message content (kept in sync by triggers) |
| `artifacts` | (schema 2) documents, later slides and designs: kind, title, content (Markdown for a document), pinned, timestamps. See [09-artifacts.md](09-artifacts.md) |
| `artifact_versions` | (schema 3) the states an artifact had before a change: rev, title, content, label, time. At most 200 per artifact; deleted with the artifact |

What is **not** stored: the API key, model files and calibration measurements (separate files the server already keeps), message HTML (only the raw text; the UI renders and sanitizes Markdown at display time). The chat API (`/v1/chat/completions`) stays stateless: it never reads or writes this database; the app saves messages through the endpoints below.

## Endpoints

All under `/dara/`, JSON, protected by the API key like the rest. Real responses: `samples/conversation.json`, `conversations.json`, `settings.json`, `store-info.json`, `search.json`, `model-configs.json`. Types: `dara-api.d.ts`.

| Call | Purpose |
| --- | --- |
| `GET /dara/store` | database path, schema version, size, row counts, a `note` if a damaged file was replaced |
| `GET /dara/settings` | all settings as `{key: value}` |
| `PATCH /dara/settings` | body `{key: value, ...}` upserts; keys are lowercase letters, digits, `_ . -` (max 64); a value is any JSON up to 64 KB |
| `DELETE /dara/settings/{key}` | remove a setting |
| `GET /dara/conversations?limit=50&offset=0` | `{data: [...], total}`, pinned first, newest first, each with `message_count` (no messages) |
| `POST /dara/conversations` | create `{id?, title?, model?, system_prompt?, params?, pinned?}`; 201; `id` must be a UUID if given (the app may generate it so it can start streaming before the server answers); duplicate id: 409 |
| `GET /dara/conversations/{id}` | the conversation with all messages and their attachments |
| `PATCH /dara/conversations/{id}` | change `title`, `model`, `system_prompt`, `params`, `pinned` |
| `DELETE /dara/conversations/{id}` | deletes its messages, attachment rows **and files**, and search entries |
| `POST /dara/conversations/{id}/messages` | append `{id?, role: system\|user\|assistant, content, finish_reason?, error?, stats?, model?, created_at?}`; the server assigns the next `position`; 201 |
| `DELETE /dara/conversations/{id}/messages?from=N` | delete messages at positions N and later (regenerate, edit and resend); returns the count |
| `PATCH /dara/messages/{id}` | update `content`, `finish_reason`, `error`, `stats`; used when a streamed reply finishes, and as a checkpoint every few seconds while it streams so a crash loses little |
| `POST /dara/messages/{id}/attachments` | `{data: "<base64>"}` (a `data:` URI prefix is accepted); PNG, JPEG or WebP detected from the bytes; max 20 MB; returns `{id, mime, bytes, url}` |
| `GET /dara/attachments/{id}` | the image bytes. An `<img src>` cannot send the `Authorization` header: fetch it with the header and show it through a `blob:` URL |
| `DELETE /dara/attachments/{id}` | remove one image |
| `GET /dara/search?q=...&limit=20` | full-text search over messages: `{data: [{message_id, conversation_id, position, role, snippet, title}]}`; words are matched as prefixes; the query is never interpreted as search syntax |
| `GET /dara/artifacts?kind=&limit=&offset=` | artifacts for the gallery, each with a short `snippet`; `POST` creates, `GET`/`PATCH`/`DELETE /dara/artifacts/{id}` read, change and remove one; `GET /dara/artifacts/{id}/versions`, `GET .../versions/{rev}` and `POST .../restore` give the version history (see [09-artifacts.md](09-artifacts.md)) |
| `GET /dara/model-configs` | `{model: {kv, flash, vision, context, ...}}` |
| `PUT /dara/model-configs/{model}` | `{kv: auto\|q8\|k8v4\|q4\|tq4\|k16v8, flash: auto\|on\|off\|true\|false, vision, context: int\|null}` |
| `DELETE /dara/model-configs/{model}` | back to Auto |
| `GET /dara/data/export` | everything as one JSON document (download); artifacts are included, their version history is not |
| `DELETE /dara/data` | body `{"confirm": "delete-all-data"}` deletes conversations, messages, attachment files and model configs, then compacts the file; settings are kept |

Limits: 2 MB of text per message, 20 MB per image, 96 MB request body overall. Errors use the usual `{error: {message, type}}` with 400 (invalid input), 404, 409, 413, 503 (`store_unavailable`).

## How the app should use it

- On startup: `GET /dara/settings`, `GET /dara/model-configs`, `GET /dara/conversations`. Keep the active conversation in React state; save through the endpoints, not browser storage.
- Sending a message: append the user message, append an empty assistant message, stream from `/v1/chat/completions`, then `PATCH` the assistant message with the final text, `finish_reason` and the final chunk's stats (and checkpoint every few seconds while streaming). On interruption, patch it with `finish_reason: "interrupted"`.
- Applying a model's options saves `PUT /dara/model-configs/{id}` as well as calling `POST /dara/models/load`.
- Settings that only the app understands (theme, layout) are plain keys; the server stores them without interpreting them.

## Implementation notes

- SQLite 3.45.3 is vendored (`third_party/sqlite`, public domain) and compiled with FTS5. Connection setup: WAL, `synchronous=NORMAL`, foreign keys on, busy timeout 5 s, page cache about 2 MB (the database adds almost nothing to the memory budget).
- `PRAGMA user_version` holds the schema version. A database from a newer Dara is refused (clear message). Older ones are migrated forward-only inside a transaction after a `dara.db.bak-v<N>` copy; an applied migration is never edited (add a new one). Schema version 2 added `artifacts` (`db/0002_artifacts.sql`); a database from the previous build is upgraded in place on first start and `dara.db.bak-v1` keeps the old file. Schema version 3 added `artifact_versions` (`db/0003_artifact_versions.sql`); a schema 2 database is upgraded in place and `dara.db.bak-v2` keeps the old file (a schema 1 database passes through both, leaving `bak-v1` and `bak-v2`). New migrations are numbered files in `docs/ui/db/` (no gaps); `gen_store_schema.py` refuses a gap.
- On start `PRAGMA quick_check` runs. A damaged file is kept as `dara.db.corrupt-<time>`, a fresh database is created, and `/dara/store` returns a `note`.
- Every statement is parameterized; ids are UUIDs; attachment file names are generated by the server (never taken from a request) and checked on read; user text is never HTML.

## Privacy and deletion

- Local only; never uploaded; no analytics rows; the server never logs message content.
- "Delete all data" removes rows, runs `VACUUM`, and deletes every file in `attachments`. Deleting a conversation removes its attachment files too.
- Encryption at rest is not on by default (the data is protected by the user's Windows profile). If users ask for it, SQLCipher or Windows DPAPI-wrapped keys are the options; do not invent a home-made scheme.
- Anyone who can call the API with the key can read the history; that is why the key is per run and the server listens on loopback only (see 05).
