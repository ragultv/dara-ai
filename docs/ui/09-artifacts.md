# Artifacts

Documents (now) and slides and designs (later) that you write yourself or ask Dara to write, kept on this computer. The gallery is the **Artifacts** page; a document opens in its own editor. The model is the local one already loaded for chat; nothing is sent anywhere.

Status: **Milestone 1, documents (with the chat panel).** Slides and designs show as "Soon" tiles and use the same table when they arrive.

## What the user sees

| Screen | Route | What it does |
| --- | --- | --- |
| Gallery | `#/artifacts` | "Make something new" tiles (Docs works; Slides and Design say Soon), search, and a card per artifact: a shrunken preview of the start of the document, its title, and "Docs · 5 minutes ago". A card's menu duplicates or deletes (delete is confirmed). An empty gallery explains itself and offers New doc. |
| Document | `#/artifacts/<uuid>` | Back arrow, the title menu (Rename, Duplicate, Export as Markdown, Delete, and when it was last updated), a large serif title, the editor with a floating formatting bar, and the **chat** panel on the right (hide or show it with "Show chat"). Saved by itself a moment after the last keystroke ("Saving...", "Saved", or "Not saved" in red). |

The editor is what-you-see-is-what-you-get: headings 1 to 3, bold, italic, inline code, bulleted, numbered and task lists, tables, code blocks, quotes and dividers. The page is set in the reply font from Settings > Appearance (a serif by default), like Dara's chat answers.

- **Selected text** shows a small menu: text style, bold, italic, strikethrough, inline code, bulleted list and link (a link must be http(s) or mailto; "example.com" becomes https://example.com).
- **Tables** (the toolbar inserts a 3 by 3 table with a header row) get controls around the table the cursor is in: a grip above the column and one beside the row (insert left/right or above/below, delete), a "+" strip on the right edge and below the bottom edge to add a column or a row, and a menu to delete the table. **Drag the border between two columns to change their width.** Column widths are not saved: the document is stored as Markdown, which has no widths.
- The title and the page never show the browser's focus ring.

## Sections, comments, images and widgets

- **Collapsible sections.** A chevron appears before every heading (on hover; always when folded). It hides everything under the heading up to the next heading of the same or a higher level. Only the view changes: the Markdown and what Dara reads are untouched, and the folds last until the page is closed.
- **Comments.** Select words (or put the cursor in a paragraph) and use the comment button in the selection menu or the toolbar. A thread points at the **exact words** (`quote`) and which occurrence of them it is (`nth`), so it survives edits elsewhere; if the words themselves are edited away the thread stays in the list without a highlight. Threads live in the **Comments** tab (a badge counts the open ones): a card shows the words, the messages (you and Dara), a reply box, Resolve and Delete, and a click on the words or the highlight finds the other. Comments are stored with the document ([10-artifact-collab.md](10-artifact-collab.md)).
- **Dara and comments.** Dara acts on a thread when you press **Ask Dara** on a new comment, when you reply to a thread she took part in, or when you write `@dara` in a reply. She reads the thread and the document, changes the document with block operations where it calls for that (one **Before Dara** version is kept, so it can be undone), and answers in the thread. The same run is shown in the Chat tab. When Dara is unsure about something she does not guess: the block operation `@comment N` puts a question next to block N's words as a comment from Dara (the text is left alone), and your reply makes her act on the answer.
- **Images and video.** The image button in the toolbar (and paste or drop) stores the file with the document and puts it in the page: PNG, JPEG, GIF, WebP images and MP4 or WebM video, up to 25 MB each. The Markdown holds only `![name](dara-asset:<id>)` (`"video"` as the title for a video), never the bytes, so Dara and the gallery never see base64. The app fetches the file with the API key and shows it from a blob URL. Media from anywhere else on the internet is not loaded (a note takes its place). Duplicating a document copies its media.
- **Widgets.** A fenced code block whose language is `widget` is a small interactive piece written as one self-contained HTML page (its own CSS and JavaScript, no outside files). It is shown live under its code, in a frame with `sandbox="allow-scripts"` (no same-origin: no access to the app, its storage or the API key; no popups, no navigation) and a Content-Security-Policy of `default-src 'none'` plus inline script and style, so it has **no network**. The code is hidden behind "Show code" until the cursor is in it. The frame is a `data:` page, which is why the app's CSP allows `frame-src data:` (and `media-src blob:` for video); a `srcdoc` frame would inherit the app's own CSP and could not run its script.
- **Chat is saved.** The conversation in the Chat tab belongs to the document: each message is saved with it and comes back when it is opened again. A message that was still running when the app closed is shown as interrupted. "Clear chat" deletes the saved copy. The undo snapshot is not saved (the History tab covers going back).
- **A task, then an answer.** Dara's work shows as a task timeline (the assistant-ui Timeline element: the plan, then each section or step, with how far into the task it began) and her answer is written only when the task has finished; nothing is chatted while it runs.

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

The table `artifacts` (schema version 2, migration [`db/0002_artifacts.sql`](db/0002_artifacts.sql); its history is in `artifact_versions`, below): `id` (UUID), `kind` (`doc`, `slides`, `design`), `title`, `content`, `pinned`, `created_at`, `updated_at`. A document's `content` is **Markdown exactly as written, never HTML**; Markdown is also what the local model reads and writes, and what Export gives. Slides and designs will keep JSON in the same column.

| Call | Purpose |
| --- | --- |
| `GET /dara/artifacts?kind=&limit=&offset=` | `{data: [...], total}`, pinned first then newest first. Each item has a `snippet` (the first 600 characters of the content) and `chars`, never the whole body. Sample: [samples/artifacts.json](samples/artifacts.json) |
| `POST /dara/artifacts` | `{id?, kind? (default doc), title?, content?, pinned?}`; 201. A given `id` must be a UUID (the app makes one so it can open the editor at once); a duplicate id is 409; an unknown `kind` is 400 |
| `GET /dara/artifacts/{id}` | one artifact with its full `content`. Sample: [samples/artifact.json](samples/artifact.json) |
| `PATCH /dara/artifacts/{id}` | change `title`, `content`, `pinned`; refreshes `updated_at`. An optional `snapshot` (a label of at most 120 characters) first stores the state before the change as a version (see Version history); a non-string or longer label is 400 and nothing is written |
| `DELETE /dara/artifacts/{id}` | remove it and its versions (404 when absent) |
| `GET /dara/artifacts/{id}/versions?limit=&offset=` | the history, newest first: `{data: [{rev, title, label, created_at, chars, snippet}], total}` (`snippet` is the first 200 characters, no full content); 404 for an unknown artifact. Sample: [samples/artifact-versions.json](samples/artifact-versions.json) |
| `GET /dara/artifacts/{id}/versions/{rev}` | one version with its full `content`: `{rev, title, content, label, created_at, chars}`; 404 when absent. Sample: [samples/artifact-version.json](samples/artifact-version.json) |
| `POST /dara/artifacts/{id}/restore` | body `{"rev": N}`: keeps the current state as a version labelled "Before restore", then sets the title and content from version N and refreshes `updated_at`; returns the artifact like `GET`. 400 for a missing or non-integer `rev`, 404 for an unknown artifact or version |

Limits: 2 MB of text per artifact, 500 characters of title. `GET /dara/store` counts artifacts, `GET /dara/data/export` includes them, and "Delete all data" removes them (the same confirmation as chats). Errors and the API key work as for the other `/dara/` calls (see [08-database.md](08-database.md)).

## Version history

Schema version 3 ([`db/0003_artifact_versions.sql`](db/0003_artifact_versions.sql)) adds `artifact_versions(artifact_id, rev, title, content, label, created_at)`. A version is **the state an artifact had before a change**, so the newest text is always the artifact itself and a version is something you can go back to. `rev` counts 1, 2, 3 ... per artifact. `created_at` is when that state was last saved (the artifact's `updated_at` at the moment it was set aside), so the list reads "how the document looked at 3:04 pm".

When the server stores one (all in the same transaction as the `PATCH`):

- **Explicit:** the patch carries `snapshot` (the app sends a label such as "Before Dara edit" before a big change it is about to save).
- **Automatic:** the patch changes the `content` and the artifact was last saved more than **120 seconds** ago (`kVersionGapMs`, so a long writing session is not one version per keystroke, but "the text before this sitting" is kept). A title-only change never makes an automatic version.
- **Restore:** the current state is kept as "Before restore" first, so a restore can itself be undone.

What the server never stores: a state that is empty (empty title and content), a state identical to the newest version, or anything for a patch that changes neither `title` nor `content`. At most **200 versions** per artifact are kept (`kVersionCap`); the oldest are removed first. Versions are deleted with their artifact and by "Delete all data". `GET /dara/data/export` does not include them. For tests only, `DARA_VERSION_GAP_MS` and `DARA_VERSION_CAP` in the server's environment replace the two limits.

### In the editor

The panel beside the page has two tabs, **Chat** and **History**. History lists "Current version" and then each saved version (its label, or "Autosaved version", when it was saved and its size). Click one to preview it, then **Restore this version** (the text from before the restore is kept as "Before restore"). The list follows the document: it reloads after every save. Before Dara changes a document the app saves what is on screen and sends the change with `snapshot: "Before Dara: <your request>"`, so every Dara answer can be gone back to even after the app is closed (the chat's **Undo** only lasts until then).

## Where the code is

| Part | File |
| --- | --- |
| Table, migrations, endpoints | `docs/ui/db/0002_artifacts.sql`, `docs/ui/db/0003_artifact_versions.sql`, `src/serve/store.cpp` (`Store::artifact_*`, routes), `tools/gen_store_schema.py` |
| Server test (schema 1 and 2 upgrades, version history included) | `tests/store_api_test.py` |
| Gallery and list/open/save state | `app/src/routes/ArtifactsPage.tsx`, `app/src/lib/stores/useArtifactsStore.ts` (debounced autosave; the page is flushed when you leave it) |
| Editor | `app/src/routes/ArtifactEditorPage.tsx`, `app/src/lib/artifacts/useDocEditor.ts` (Tiptap with `tiptap-markdown`), `DocToolbar.tsx`, `SelectionMenu.tsx`, `TableControls.tsx` |
| History panel | `app/src/lib/artifacts/DocHistory.tsx` (list, preview, restore), `useArtifactsStore.restore` |
| Chat panel | `app/src/lib/artifacts/DocChat.tsx` (tabs Chat, Comments, History; the task timeline is `app/src/lib/components/assistant-ui/elements/timeline.tsx`), `useDocAssistant.ts` (runs it against the editor), `app/src/lib/stores/useDocChatStore.ts` (saved with the document) |
| Comments | `DocComments.tsx`, `threads.ts`, `extensions/commentMarks.ts` (anchors and highlights), `app/src/lib/stores/useCommentsStore.ts` |
| Folding, media, widgets | `extensions/sectionFold.ts`, `extensions/media.ts`, `extensions/widgets.ts` |
| Assistant logic (pure, tested) | `assistant.ts` (the two flows), `plan.ts` (plan and section prompts and parsing), `blocks.ts` (split, operations, apply), `completion.ts` (one streamed request with thinking), `askDara.ts` (cleaning, whole-document fallback) |
| Unit tests | `app/tests/unit/artifacts.test.ts`, `doc-assistant.test.ts`, `collab.test.ts` |

## Safety

- A document is plain text. The editor never renders raw HTML from Markdown (`html: false`) and keeps Tiptap's link protocol check, so a document written by the model cannot run anything.
- The gallery preview goes through the same sanitized Markdown renderer as chat.
- Everything stays local; the only network call is to the local engine.

## Next

Slides (a slide list, text, image and shape tools, present mode) and designs on the same table and gallery; moving the assistant loop to the server (`POST /dara/artifacts/{id}/assist`, streaming steps, thinking and document patches) so other clients can use it and a run survives leaving the page; saving the chat per document; search across document text; "Save as artifact" from a chat answer.
