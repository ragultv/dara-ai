-- Migration 0004 (schema version 3 -> 4): the chat, the comments and the images/videos that belong to an artifact.
-- Applied once, inside a transaction, after a dara.db.bak-v3 copy. Never edit an applied migration: add 0005_*.sql instead.
-- Embedded by tools/gen_store_schema.py (src/serve/store_schema.h). See docs/ui/10-artifact-collab.md.
-- The server deletes these rows explicitly when an artifact (or all data) is deleted; it does not rely on the foreign keys.

-- The conversation next to a document. `data` is a JSON object the app owns (what was asked, what Dara did); the server does not read it.
CREATE TABLE artifact_chat (
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  id          TEXT NOT NULL,                       -- the app's message id, [A-Za-z0-9_-]{1,64}
  seq         INTEGER NOT NULL,                    -- order in the conversation: 1, 2, 3 ... per artifact; an update keeps its seq
  data        TEXT NOT NULL,                       -- a JSON object, at most 200 KB
  created_at  INTEGER NOT NULL,                    -- unix ms
  updated_at  INTEGER NOT NULL,
  PRIMARY KEY (artifact_id, id)
);
CREATE INDEX idx_artifact_chat_seq ON artifact_chat (artifact_id, seq);

-- Comments. A root comment (parent_id NULL) starts a thread and carries the quote it points at; replies point at the root (one level deep).
CREATE TABLE artifact_comments (
  id          TEXT PRIMARY KEY,                    -- UUID
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  parent_id   TEXT,                                -- NULL = a thread's first comment; else the id of that root
  author      TEXT NOT NULL CHECK (author IN ('user','dara')),
  body        TEXT NOT NULL,                       -- 1 to 8000 characters
  quote       TEXT NOT NULL DEFAULT '',            -- the exact text the thread is anchored to (at most 500 characters); '' = the whole document; '' on replies
  nth         INTEGER NOT NULL DEFAULT 0,          -- which occurrence of the quote in the document (0 = the first)
  resolved    INTEGER NOT NULL DEFAULT 0 CHECK (resolved IN (0,1)),
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX idx_artifact_comments_artifact ON artifact_comments (artifact_id, created_at);

-- Images and video stored with a document. The document's Markdown refers to one as dara-asset:<id>.
CREATE TABLE artifact_assets (
  id          TEXT PRIMARY KEY,                    -- UUID
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  name        TEXT NOT NULL,                       -- the file name it came with (at most 200 characters)
  mime        TEXT NOT NULL,                       -- image/png, image/jpeg, image/gif, image/webp, video/mp4 or video/webm
  size        INTEGER NOT NULL,                    -- bytes
  data        BLOB NOT NULL,
  created_at  INTEGER NOT NULL
);
CREATE INDEX idx_artifact_assets_artifact ON artifact_assets (artifact_id);
