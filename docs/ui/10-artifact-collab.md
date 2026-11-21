# Artifact chat, comments, images and video (server API)

What belongs to an artifact besides its text: the **chat** next to it, its **comments**, and the **images and videos** in it. Schema version 4 ([`db/0004_artifact_collab.sql`](db/0004_artifact_collab.sql)); code in `src/serve/store.cpp` (`Store::artifact_chat_*`, `artifact_comment_*`, `artifact_asset_*`, `artifact_duplicate`). The artifact itself, its endpoints and its version history are in [09-artifacts.md](09-artifacts.md). All of this is under `/dara/`, JSON unless noted, protected by the API key like the rest, and `404` when the artifact does not exist. Types: `dara-api.d.ts`. Test: `tests/store_api_test.py`.

Everything is deleted with its artifact and by "Delete all data". The server does this explicitly; it does not rely on the foreign keys.

## Chat

The conversation beside a document. The server stores one row per message and never reads it: `data` is a JSON object the app owns (what was asked, what Dara did, its steps and thinking). Sample: [samples/artifact-chat.json](samples/artifact-chat.json).

| Call | Purpose |
| --- | --- |
| `GET /dara/artifacts/{id}/chat` | `{object: "list", data: [{id, seq, data, created_at, updated_at}], total}` in `seq` order. `data` is a real JSON object, not a string |
| `PUT /dara/artifacts/{id}/chat/{mid}` | body `{"data": {...}}`. **Upsert:** a new message id gets the next `seq` (1, 2, 3 ... per artifact); an existing one keeps its `seq` and gets the new `data` and `updated_at`. Returns the row. `mid` is 1 to 64 characters of `A-Z a-z 0-9 _ -`; `data` must be a JSON object of at most 200 KB (400 otherwise) |
| `DELETE /dara/artifacts/{id}/chat` | clears the chat: `{"deleted": n}` |

At most **500** messages are kept per artifact; saving a new one drops the lowest `seq` beyond that. (Test-only: `DARA_CHAT_CAP`.)

## Comments

A **thread** is a root comment (`parent_id: null`) that carries the `quote` it points at, plus its replies (`parent_id` = the root's id). Threads are one level deep. Sample: [samples/artifact-comments.json](samples/artifact-comments.json).

A comment: `{id, parent_id, author, body, quote, nth, resolved, created_at, updated_at}`.

- `author` is `user` or `dara`.
- `body` is 1 to 8000 characters and not blank. Limits count characters, not bytes.
- `quote` is the exact text the thread is anchored to (at most 500 characters); empty means the whole document. `nth` says which occurrence of the quote in the document (0 = the first). The server stores them as given: finding the text in the document, and what to do when it is gone, is the app's job.
- `resolved` is only meaningful on a root.

| Call | Purpose |
| --- | --- |
| `GET /dara/artifacts/{id}/comments` | `{object: "list", data: [...], total}`: every comment of the artifact, flat, oldest first |
| `POST /dara/artifacts/{id}/comments` | `{id? (UUID), parent_id?, author? (default "user"), body, quote?, nth?}` -> 201 the comment. A given `id` that exists is 409. `parent_id` must be a root comment of the same artifact (404 when it is not found there, 400 when it is itself a reply). A reply with a non-empty `quote` is 400 |
| `PATCH /dara/artifacts/{id}/comments/{cid}` | any of `body`, `resolved` (boolean), `quote`, `nth`; refreshes `updated_at`; returns the comment. `resolved`, `quote` and `nth` on a reply are 400; an empty patch is 400 |
| `DELETE /dara/artifacts/{id}/comments/{cid}` | `{"deleted": n}`: deleting a root deletes its replies too, and `n` counts every row removed; 404 when absent |

Limits: **1000** comments per artifact and **100** replies per thread (the next `POST` is 409).

## Images and video

Stored in the database with the document; the document's Markdown refers to one as `dara-asset:<id>` (for example `![Roadmap](dara-asset:5f0c...)`). Samples: [samples/artifact-asset.json](samples/artifact-asset.json), [samples/artifact-assets.json](samples/artifact-assets.json).

| Call | Purpose |
| --- | --- |
| `POST /dara/artifacts/{id}/assets` | the **request body is the raw file**. `Content-Type` is its type; `X-Dara-Name` is its URL-encoded file name (default `file`, at most 200 characters, else 400). 201 `{id, name, mime, size, created_at}` |
| `GET /dara/artifacts/{id}/assets` | `{object: "list", data: [{id, name, mime, size, created_at}], total}`: metadata only |
| `GET /dara/artifacts/{id}/assets/{aid}` | the bytes, with the stored `Content-Type`, `Cache-Control: private, max-age=31536000, immutable` and `X-Content-Type-Options: nosniff`. The app fetches it with its Authorization header and shows it through a blob URL (an `<img src>` cannot send the header) |
| `DELETE /dara/artifacts/{id}/assets/{aid}` | `{"deleted": true}`; 404 when absent |

What is accepted: `image/png`, `image/jpeg`, `image/gif`, `image/webp`, `video/mp4`, `video/webm`. Anything else, **SVG included**, is `415`. The first bytes must match the declared type (PNG signature, JPEG `FF D8 FF`, `GIF87a`/`GIF89a`, `RIFF....WEBP`, an `ftyp` box for MP4, the EBML header for WebM); a mislabeled file is `415` and nothing is stored. The `Content-Type` is read without its parameters and in any case. An empty body is `400`.

Limits: **25 MB** per file and **200 MB** per artifact (`413`; test-only overrides `DARA_ASSET_MAX_BYTES` and `DARA_ASSET_ARTIFACT_MAX_BYTES`). The HTTP layer's own cap on any request body is 96 MB; a larger body gets `413` from it before the route runs.

CORS: the `X-Dara-Name` header is allowed in the preflight answer (`Access-Control-Allow-Headers`).

Known quirk: a request refused before its body is read (a missing or wrong API key, a body over the 96 MB cap) can make the client see a closed connection instead of the 401 or 413 when the body is large; the server is not affected.

## Duplicate

`POST /dara/artifacts/{id}/duplicate` with `{id? (UUID), title?}` -> `201` the full new artifact. It copies the kind and the text, **and every image and video** (new asset ids; each `dara-asset:<oldid>` in the copied text is rewritten to the new id, ids that are not this artifact's assets are left alone). It does **not** copy versions, chat or comments. The title is the given `title`, else the original's plus ` (copy)` (an untitled document stays untitled). One transaction. 404 for an unknown artifact, 409 for an `id` that exists, 400 for an `id` that is not a UUID.

## Export and counts

- `GET /dara/data/export`: each artifact in `artifacts` also carries `chat` and `comments` (the rows above) and `assets` (metadata only, no bytes: fetch those from the asset endpoint). Version history is still not exported.
- `GET /dara/store` adds `artifact_comments` and `artifact_assets` row counts.
