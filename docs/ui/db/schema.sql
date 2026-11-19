-- Dara app database, schema version 1. File: %LOCALAPPDATA%\Dara\dara.db (SQLite, WAL). Owned by the app's Rust side; see docs/ui/08-database.md.
-- Applied with PRAGMA user_version = 1. Forward-only migrations live next to this file as 0002_*.sql, ...
PRAGMA foreign_keys = ON;

-- Key/value settings (theme, port, defaults, window state ...). value is JSON.
CREATE TABLE settings (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL CHECK (json_valid(value)),
  updated_at INTEGER NOT NULL                      -- unix ms
) WITHOUT ROWID;

-- Chat history.
CREATE TABLE conversations (
  id         TEXT PRIMARY KEY,                     -- UUID v4 text
  title      TEXT NOT NULL DEFAULT '',
  model      TEXT,                                 -- dara model id last used (dara-nano ...)
  system_prompt TEXT NOT NULL DEFAULT '',
  params     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(params)),   -- temperature, max_tokens, flash ... for this conversation
  pinned     INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0,1)),
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX idx_conversations_updated ON conversations (pinned DESC, updated_at DESC);

CREATE TABLE messages (
  id              TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  position        INTEGER NOT NULL,                -- order inside the conversation, 0-based
  role            TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
  content         TEXT NOT NULL DEFAULT '',        -- plain text / Markdown as received, never HTML
  finish_reason   TEXT,                            -- 'stop' | 'length' | 'interrupted' | 'error' (assistant messages)
  error           TEXT,
  stats           TEXT CHECK (stats IS NULL OR json_valid(stats)),    -- usage, timings, context, speculative (the final chunk's numbers)
  model           TEXT,
  created_at      INTEGER NOT NULL,
  UNIQUE (conversation_id, position)
);
CREATE INDEX idx_messages_conv ON messages (conversation_id, position);

-- Images attached to messages: the file lives in %LOCALAPPDATA%\Dara\attachments\, only its relative name is stored.
CREATE TABLE attachments (
  id         TEXT PRIMARY KEY,
  message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
  mime       TEXT NOT NULL CHECK (mime IN ('image/png','image/jpeg','image/webp')),
  file       TEXT NOT NULL,                        -- relative file name, no path separators
  thumb      TEXT,                                 -- relative thumbnail file name
  bytes      INTEGER NOT NULL,
  width      INTEGER, height INTEGER
);
CREATE INDEX idx_attachments_message ON attachments (message_id);

-- The configuration the user chose per model (applied with POST /dara/models/load). Absent row = "Auto".
CREATE TABLE model_configs (
  model        TEXT PRIMARY KEY,                   -- dara model id
  kv           TEXT NOT NULL DEFAULT 'auto' CHECK (kv IN ('auto','q8','k8v4','q4','tq4','k16v8')),
  flash        TEXT NOT NULL DEFAULT 'auto' CHECK (flash IN ('auto','on','off')),
  vision       INTEGER NOT NULL DEFAULT 1 CHECK (vision IN (0,1)),
  context      INTEGER,                            -- NULL = auto
  last_loaded_at INTEGER,
  updated_at   INTEGER NOT NULL
) WITHOUT ROWID;

-- Full-text search over messages (external-content index kept in sync by triggers).
CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content); END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content); END;
CREATE TRIGGER messages_au AFTER UPDATE OF content ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Default settings, inserted on first run (the app may add more keys; unknown keys are kept).
INSERT INTO settings (key, value, updated_at) VALUES
  ('theme',               '"system"', 0),
  ('server.port',         '11435',    0),
  ('server.models_dir',   'null',     0),   -- null = default (%LOCALAPPDATA%\Dara\models)
  ('server.allow_lan',    'false',    0),
  ('defaults.kv',         '"auto"',   0),
  ('defaults.flash',      '"auto"',   0),
  ('defaults.vram_reserve_mb', '64',  0),
  ('defaults.vision',     'true',     0),
  ('chat.temperature',    '0.7',      0),
  ('chat.max_tokens',     '4096',     0),
  ('chat.system_prompt',  '""',       0),
  ('app.start_at_login',  'false',    0),
  ('app.minimize_to_tray','true',     0);
