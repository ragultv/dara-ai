-- Migration 0002 (schema version 1 -> 2): Dara Artifacts. Documents first; slides and designs use the same table later.
-- Applied once, inside a transaction, after a dara.db.bak-v1 copy. Never edit an applied migration: add 0003_*.sql instead.
-- Embedded by tools/gen_store_schema.py (src/serve/store_schema.h).

CREATE TABLE artifacts (
  id         TEXT PRIMARY KEY,                     -- UUID v4 text
  kind       TEXT NOT NULL CHECK (kind IN ('doc','slides','design')),
  title      TEXT NOT NULL DEFAULT '',
  content    TEXT NOT NULL DEFAULT '',             -- doc: Markdown as written (never HTML); slides / design: JSON (later)
  pinned     INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0,1)),
  created_at INTEGER NOT NULL,                     -- unix ms
  updated_at INTEGER NOT NULL
);
CREATE INDEX idx_artifacts_updated ON artifacts (kind, pinned DESC, updated_at DESC);
