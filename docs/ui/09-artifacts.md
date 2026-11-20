# Artifacts

Documents (now) and slides and designs (later) that you write yourself or ask Dara to write, kept on this computer. The gallery is the **Artifacts** page; a document opens in its own editor. The model is the local one already loaded for chat; nothing is sent anywhere.

Status: **Milestone 1, documents.** Slides and designs show as "Soon" tiles and use the same table when they arrive.

## What the user sees

| Screen | Route | What it does |
| --- | --- | --- |
| Gallery | `#/artifacts` | "Make something new" tiles (Docs works; Slides and Design say Soon), search, and a card per artifact: a shrunken preview of the start of the document, its title, and "Docs · 5 minutes ago". A card's menu duplicates or deletes (delete is confirmed). An empty gallery explains itself and offers New doc. |
| Document | `#/artifacts/<uuid>` | Back arrow, the title menu (Rename, Duplicate, Export as Markdown, Delete, and when it was last updated), a large serif title, the editor, and at the bottom a prompt ("Describe a change") above a formatting bar. Saved by itself a moment after the last keystroke ("Saving...", "Saved", or "Not saved" in red). |

The editor is what-you-see-is-what-you-get: headings 1 to 3, bold, italic, inline code, bulleted, numbered and task lists, tables, code blocks, quotes and dividers. The page is set in the reply font from Settings > Appearance (a serif by default), like Dara's chat answers.

## Ask Dara

The prompt under the page ("Describe a change") sends the **whole document as Markdown and one instruction** to `POST /v1/chat/completions` (streaming, temperature 0.3, thinking off) and puts the answer back as the document while it streams. The system prompt tells the model to return the complete updated document in Markdown and nothing else (`app/src/lib/artifacts/askDara.ts`).

- The editor is read-only while Dara writes; **Stop** keeps what has arrived.
- When it finishes, **"Dara edited this document · Undo"** puts the earlier text back with one click. Nothing is lost, and the result is saved like any edit.
- An empty answer never replaces a document; an error restores the text from before.
- A fence around the whole answer (```` ```markdown ````), a lead-in such as "Here is the updated document:" and `<document>` tags are removed. Anything inside the document is left alone.
- With no model loaded the prompt says so and offers a button to the Models page.
- The reply budget is sized from the document (`max_tokens` = about 1.4 x its length plus margin, at most 8192), so a very long document may not fit the model's context; the engine then reports the error and the document is restored.

## Storage and API

One table, `artifacts` (schema version 2, migration [`db/0002_artifacts.sql`](db/0002_artifacts.sql)): `id` (UUID), `kind` (`doc`, `slides`, `design`), `title`, `content`, `pinned`, `created_at`, `updated_at`. A document's `content` is **Markdown exactly as written, never HTML**; Markdown is also what the local model reads and writes, and what Export gives. Slides and designs will keep JSON in the same column.

| Call | Purpose |
| --- | --- |
| `GET /dara/artifacts?kind=&limit=&offset=` | `{data: [...], total}`, pinned first then newest first. Each item has a `snippet` (the first 600 characters of the content) and `chars`, never the whole body. Sample: [samples/artifacts.json](samples/artifacts.json) |
| `POST /dara/artifacts` | `{id?, kind? (default doc), title?, content?, pinned?}`; 201. A given `id` must be a UUID (the app makes one so it can open the editor at once); a duplicate id is 409; an unknown `kind` is 400 |
| `GET /dara/artifacts/{id}` | one artifact with its full `content`. Sample: [samples/artifact.json](samples/artifact.json) |
| `PATCH /dara/artifacts/{id}` | change `title`, `content`, `pinned`; refreshes `updated_at` |
| `DELETE /dara/artifacts/{id}` | remove it (404 when absent) |

Limits: 2 MB of text per artifact, 500 characters of title. `GET /dara/store` counts artifacts, `GET /dara/data/export` includes them, and "Delete all data" removes them (the same confirmation as chats). Errors and the API key work as for the other `/dara/` calls (see [08-database.md](08-database.md)).

## Where the code is

| Part | File |
| --- | --- |
| Table, migration, endpoints | `docs/ui/db/0002_artifacts.sql`, `src/serve/store.cpp` (`Store::artifact_*`, routes), `tools/gen_store_schema.py` |
| Server test (schema 1 to 2 upgrade included) | `tests/store_api_test.py` |
| Gallery and list/open/save state | `app/src/routes/ArtifactsPage.tsx`, `app/src/lib/stores/useArtifactsStore.ts` (debounced autosave; the page is flushed when you leave it) |
| Editor | `app/src/routes/ArtifactEditorPage.tsx`, `app/src/lib/artifacts/useDocEditor.ts` (Tiptap with `tiptap-markdown`), `DocToolbar.tsx` |
| Ask Dara | `app/src/lib/artifacts/askDara.ts` |
| Unit tests | `app/tests/unit/artifacts.test.ts` |

## Safety

- A document is plain text. The editor never renders raw HTML from Markdown (`html: false`) and keeps Tiptap's link protocol check, so a document written by the model cannot run anything.
- The gallery preview goes through the same sanitized Markdown renderer as chat.
- Everything stays local; the only network call is to the local engine.

## Next

Slides (a slide list, text, image and shape tools, present mode) and designs on the same table and gallery; version history (a table of saved versions per artifact); search across document text; "Save as artifact" from a chat answer.
