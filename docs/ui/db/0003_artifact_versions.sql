-- Migration 0003 (schema version 2 -> 3): version history for artifacts.
-- Applied once, inside a transaction, after a dara.db.bak-v2 copy. Never edit an applied migration: add 0004_*.sql instead.
-- Embedded by tools/gen_store_schema.py (src/serve/store_schema.h).
-- A row is the state an artifact had BEFORE a change (see docs/ui/09-artifacts.md, "Version history"). The server writes them; it keeps
-- at most 200 per artifact and deletes them with the artifact (explicitly: it does not rely on the foreign key).

CREATE TABLE artifact_versions (
  artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  rev         INTEGER NOT NULL,                    -- 1, 2, 3 ... per artifact; never reused while newer versions exist
  title       TEXT NOT NULL DEFAULT '',
  content     TEXT NOT NULL DEFAULT '',
  label       TEXT NOT NULL DEFAULT '',            -- '' for an automatic snapshot, else what the client asked for ("Before Dara edit", "Before restore")
  created_at  INTEGER NOT NULL,                    -- unix ms: when that state was last saved (the artifact's updated_at at snapshot time)
  PRIMARY KEY (artifact_id, rev)
);
