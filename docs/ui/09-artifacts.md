# Artifacts

Documents (now) and slides and designs (later) that you write yourself or ask Dara to write, kept on this computer. The gallery is the **Artifacts** page; a document opens in its own editor. The model is the local one already loaded for chat; nothing is sent anywhere.

Status: **Milestone 1, documents (with the chat panel).** Slides and designs show as "Soon" tiles and use the same table when they arrive.

## What the user sees

| Screen | Route | What it does |
| --- | --- | --- |
| Gallery | `#/artifacts` | "Make something new" tiles (Docs works; Slides and Design say Soon), search, and a card per artifact: a shrunken preview of the start of the document, its title, and "Docs · 5 minutes ago". A card's menu duplicates or deletes (delete is confirmed). An empty gallery explains itself and offers New doc. |
| Document | `#/artifacts/<uuid>` | Back arrow, the title menu (Rename, Duplicate, Export as Markdown, Delete, and when it was last updated), a large serif title, the editor with a floating formatting bar, and the **chat** panel on the right (hide or show it with "Show chat"). Saved by itself a moment after the last keystroke ("Saving...", "Saved", or "Not saved" in red). |

The editor is what-you-see-is-what-you-get: headings 1 to 3, bold, italic, inline code, bulleted, numbered and task lists, tables, code blocks, quotes and dividers. The page is set in the reply font from Settings > Appearance (a serif by default), like Dara's chat answers.

## Chat with Dara (the panel beside the document)

The conversation lives in the panel, never in the page: the page only ever receives document content. The panel shows what you asked, Dara's thinking (the same effort levels and "Thought for 12s" block as chat; the effort pill sits in the composer), a short progress list, a one-line result and **Undo**. Messages are kept in memory for the app session, one list per document; the document itself is what is saved.

Two ways of working, chosen by whether the document has content (`app/src/lib/artifacts/assistant.ts`):

- **Empty document: plan, then fill.** Dara first returns a plan (a title and up to 8 sections, `plan.ts`). The outline appears in the page at once, each section as a heading with a "Dara is writing this section..." placeholder, and the title is filled in if it was empty. Then one request per section writes its body into place while you watch (each request sees the outline and the sections already written). Thinking applies to the planning step only. If the model ignores the plan format, Dara writes the whole document in one go instead.
- **Document with content: block operations.** The document is split into numbered top-level blocks (`blocks.ts`; a list, a table or a fenced code block is one block). The model answers with `@replace N`, `@insert N` (after block N, 0 = start) and `@delete N` lines and nothing else; the app applies them. **Blocks that are not mentioned stay byte for byte as they were.** Numbers refer to the original numbering; a number that does not exist is skipped and reported. If the model answers in words instead (a question about the document), that answer is shown in the chat and nothing changes.

Common to both:

- The editor is read-only while Dara writes; **Stop** keeps what has been written (unfinished sections are dropped, not left as placeholders).
- **Undo** (under the newest answer) puts back the text and the title from before it. An error restores both.
- Only the newest answer can be undone: sending the next message drops older undo snapshots.
- With no model loaded the panel says so and offers a button to the Models page.
- Raw Markdown from the model goes through the same cleaning as before (`cleanEditResult`: a fence around the whole answer, a lead-in line and `<document>` tags are removed).

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
| Chat panel | `app/src/lib/artifacts/DocChat.tsx`, `useDocAssistant.ts` (runs it against the editor), `app/src/lib/stores/useDocChatStore.ts` |
| Assistant logic (pure, tested) | `assistant.ts` (the two flows), `plan.ts` (plan and section prompts and parsing), `blocks.ts` (split, operations, apply), `completion.ts` (one streamed request with thinking), `askDara.ts` (cleaning, whole-document fallback) |
| Unit tests | `app/tests/unit/artifacts.test.ts`, `app/tests/unit/doc-assistant.test.ts` |

## Safety

- A document is plain text. The editor never renders raw HTML from Markdown (`html: false`) and keeps Tiptap's link protocol check, so a document written by the model cannot run anything.
- The gallery preview goes through the same sanitized Markdown renderer as chat.
- Everything stays local; the only network call is to the local engine.

## Next

Slides (a slide list, text, image and shape tools, present mode) and designs on the same table and gallery; moving the assistant loop to the server (`POST /dara/artifacts/{id}/assist`, streaming steps, thinking and document patches) so other clients can use it and a run survives leaving the page; saving the chat per document; version history (a table of saved versions per artifact); search across document text; "Save as artifact" from a chat answer.
