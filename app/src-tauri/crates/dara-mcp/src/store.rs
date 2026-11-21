//! SQLite: connectors, what the user allowed each tool to do, and the audit log.
//! Secret values are never written here (see `secrets`); only the names of the variables are.

use crate::client::RemoteTool;
use crate::error::McpError;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Forward-only. Entry N is migration N+1; the version of a database is how many of these have run.
const MIGRATIONS: [&str; 1] = [r#"
CREATE TABLE connectors (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('stdio','http')),
  command TEXT,
  args_json TEXT NOT NULL DEFAULT '[]',
  env_names_json TEXT NOT NULL DEFAULT '[]',
  url TEXT,
  enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0,1)),
  created_at INTEGER NOT NULL
);
CREATE TABLE connector_tools (
  connector_id TEXT NOT NULL REFERENCES connectors(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  fingerprint TEXT NOT NULL,
  permission TEXT NOT NULL DEFAULT 'ask' CHECK (permission IN ('allow','ask','deny')),
  changed INTEGER NOT NULL DEFAULT 0 CHECK (changed IN (0,1)),
  available INTEGER NOT NULL DEFAULT 1 CHECK (available IN (0,1)),
  meta_json TEXT NOT NULL DEFAULT '{}',
  updated_at INTEGER NOT NULL,
  PRIMARY KEY (connector_id, name)
);
CREATE TABLE connector_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  connector_id TEXT NOT NULL REFERENCES connectors(id) ON DELETE CASCADE,
  tool TEXT NOT NULL,
  args_json TEXT NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('ok','error','denied','needs_approval')),
  duration_ms INTEGER NOT NULL DEFAULT 0,
  result_chars INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);
CREATE INDEX idx_connector_audit ON connector_audit (connector_id, id DESC);
"#];

pub const AUDIT_KEEP: i64 = 5000;
const AUDIT_ARGS_MAX: usize = 8 * 1024;

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectorRow {
    pub id: String,
    pub name: String,
    /// "stdio" or "http"
    pub kind: String,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env_names: Vec<String>,
    pub url: Option<String>,
    pub enabled: bool,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolRow {
    pub connector_id: String,
    pub name: String,
    pub fingerprint: String,
    /// "allow", "ask" or "deny". Only the user sets it.
    pub permission: String,
    pub changed: bool,
    pub available: bool,
    /// The tool as last listed (title, description, schema, annotations).
    pub meta: Value,
    pub updated_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuditRow {
    pub id: i64,
    pub connector_id: String,
    pub tool: String,
    pub args_json: String,
    pub outcome: String,
    pub duration_ms: i64,
    pub result_chars: i64,
    pub created_at: i64,
}

pub struct Store {
    conn: Mutex<Connection>,
}

fn json_list(s: &str) -> Vec<String> {
    serde_json::from_str(s).unwrap_or_default()
}

impl Store {
    pub fn open(path: &Path) -> Result<Self, McpError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| McpError::Storage(format!("could not create {}: {e}", dir.display())))?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let store = Self { conn: Mutex::new(conn) };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<(), McpError> {
        let mut conn = self.conn.lock().expect("db");
        conn.execute_batch("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);")?;
        let current: i64 = conn
            .query_row("SELECT COALESCE(MAX(version), 0) FROM schema_version", [], |r| r.get(0))?;
        if current as usize > MIGRATIONS.len() {
            return Err(McpError::Storage(format!(
                "the connectors database is version {current}, newer than this app knows ({})",
                MIGRATIONS.len()
            )));
        }
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(current as usize) {
            let tx = conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.execute("INSERT INTO schema_version (version) VALUES (?1)", params![(i + 1) as i64])?;
            tx.commit()?;
        }
        Ok(())
    }

    pub fn schema_version(&self) -> Result<i64, McpError> {
        Ok(self
            .conn
            .lock()
            .expect("db")
            .query_row("SELECT COALESCE(MAX(version), 0) FROM schema_version", [], |r| r.get(0))?)
    }

    // ---- connectors ----

    pub fn count_connectors(&self) -> Result<usize, McpError> {
        Ok(self
            .conn
            .lock()
            .expect("db")
            .query_row("SELECT COUNT(*) FROM connectors", [], |r| r.get::<_, i64>(0))? as usize)
    }

    pub fn insert_connector(&self, c: &ConnectorRow) -> Result<(), McpError> {
        self.conn.lock().expect("db").execute(
            "INSERT INTO connectors (id, name, kind, command, args_json, env_names_json, url, enabled, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![
                c.id,
                c.name,
                c.kind,
                c.command,
                serde_json::to_string(&c.args).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&c.env_names).unwrap_or_else(|_| "[]".into()),
                c.url,
                c.enabled as i64,
                c.created_at
            ],
        )?;
        Ok(())
    }

    pub fn update_connector(&self, c: &ConnectorRow) -> Result<(), McpError> {
        let n = self.conn.lock().expect("db").execute(
            "UPDATE connectors SET name=?2, command=?3, args_json=?4, env_names_json=?5, url=?6, enabled=?7 WHERE id=?1",
            params![
                c.id,
                c.name,
                c.command,
                serde_json::to_string(&c.args).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&c.env_names).unwrap_or_else(|_| "[]".into()),
                c.url,
                c.enabled as i64
            ],
        )?;
        if n == 0 {
            return Err(McpError::NotFound(format!("connector {}", c.id)));
        }
        Ok(())
    }

    /// Removes the connector and, with it, its tools and audit rows.
    pub fn delete_connector(&self, id: &str) -> Result<(), McpError> {
        self.conn
            .lock()
            .expect("db")
            .execute("DELETE FROM connectors WHERE id=?1", params![id])?;
        Ok(())
    }

    fn connector_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<ConnectorRow> {
        Ok(ConnectorRow {
            id: r.get(0)?,
            name: r.get(1)?,
            kind: r.get(2)?,
            command: r.get(3)?,
            args: json_list(&r.get::<_, String>(4)?),
            env_names: json_list(&r.get::<_, String>(5)?),
            url: r.get(6)?,
            enabled: r.get::<_, i64>(7)? != 0,
            created_at: r.get(8)?,
        })
    }

    pub fn list_connectors(&self) -> Result<Vec<ConnectorRow>, McpError> {
        let conn = self.conn.lock().expect("db");
        let mut stmt = conn.prepare(
            "SELECT id, name, kind, command, args_json, env_names_json, url, enabled, created_at
             FROM connectors ORDER BY created_at, id",
        )?;
        let rows = stmt.query_map([], Self::connector_from)?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn get_connector(&self, id: &str) -> Result<Option<ConnectorRow>, McpError> {
        Ok(self
            .conn
            .lock()
            .expect("db")
            .query_row(
                "SELECT id, name, kind, command, args_json, env_names_json, url, enabled, created_at
                 FROM connectors WHERE id=?1",
                params![id],
                Self::connector_from,
            )
            .optional()?)
    }

    // ---- tools and permissions ----

    /// Applies a fresh tool list. A new tool starts at "ask". A tool whose definition changed goes back to "ask" and is flagged
    /// until the user chooses again. A tool that disappeared is kept (the user's choice stays) but marked unavailable.
    pub fn sync_tools(&self, connector_id: &str, tools: &[RemoteTool]) -> Result<(), McpError> {
        let mut conn = self.conn.lock().expect("db");
        let tx = conn.transaction()?;
        let now = now_ms();
        let mut seen: Vec<&str> = Vec::with_capacity(tools.len());
        for t in tools {
            seen.push(&t.name);
            let fp = t.fingerprint();
            let meta = t.to_json().to_string();
            let existing: Option<String> = tx
                .query_row(
                    "SELECT fingerprint FROM connector_tools WHERE connector_id=?1 AND name=?2",
                    params![connector_id, t.name],
                    |r| r.get(0),
                )
                .optional()?;
            match existing {
                None => {
                    tx.execute(
                        "INSERT INTO connector_tools (connector_id, name, fingerprint, permission, changed, available, meta_json, updated_at)
                         VALUES (?1,?2,?3,'ask',0,1,?4,?5)",
                        params![connector_id, t.name, fp, meta, now],
                    )?;
                }
                Some(old) if old != fp => {
                    tx.execute(
                        "UPDATE connector_tools SET fingerprint=?3, permission='ask', changed=1, available=1, meta_json=?4, updated_at=?5
                         WHERE connector_id=?1 AND name=?2",
                        params![connector_id, t.name, fp, meta, now],
                    )?;
                }
                Some(_) => {
                    tx.execute(
                        "UPDATE connector_tools SET available=1, meta_json=?3 WHERE connector_id=?1 AND name=?2",
                        params![connector_id, t.name, meta],
                    )?;
                }
            }
        }
        // what is no longer listed
        let known: Vec<String> = {
            let mut stmt = tx.prepare("SELECT name FROM connector_tools WHERE connector_id=?1")?;
            let rows = stmt
                .query_map(params![connector_id], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            rows
        };
        for name in known {
            if !seen.contains(&name.as_str()) {
                tx.execute(
                    "UPDATE connector_tools SET available=0 WHERE connector_id=?1 AND name=?2",
                    params![connector_id, name],
                )?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    fn tool_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<ToolRow> {
        Ok(ToolRow {
            connector_id: r.get(0)?,
            name: r.get(1)?,
            fingerprint: r.get(2)?,
            permission: r.get(3)?,
            changed: r.get::<_, i64>(4)? != 0,
            available: r.get::<_, i64>(5)? != 0,
            meta: serde_json::from_str(&r.get::<_, String>(6)?).unwrap_or(Value::Null),
            updated_at: r.get(7)?,
        })
    }

    pub fn tools(&self, connector_id: &str) -> Result<Vec<ToolRow>, McpError> {
        let conn = self.conn.lock().expect("db");
        let mut stmt = conn.prepare(
            "SELECT connector_id, name, fingerprint, permission, changed, available, meta_json, updated_at
             FROM connector_tools WHERE connector_id=?1 ORDER BY name",
        )?;
        let rows = stmt
            .query_map(params![connector_id], Self::tool_from)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn tool(&self, connector_id: &str, name: &str) -> Result<Option<ToolRow>, McpError> {
        Ok(self
            .conn
            .lock()
            .expect("db")
            .query_row(
                "SELECT connector_id, name, fingerprint, permission, changed, available, meta_json, updated_at
                 FROM connector_tools WHERE connector_id=?1 AND name=?2",
                params![connector_id, name],
                Self::tool_from,
            )
            .optional()?)
    }

    /// The user's choice. Clears the "changed" flag: they have looked at the tool again.
    pub fn set_permission(&self, connector_id: &str, tool: &str, permission: &str) -> Result<(), McpError> {
        if !matches!(permission, "allow" | "ask" | "deny") {
            return Err(McpError::Invalid(format!("permission must be allow, ask or deny, not {permission}")));
        }
        let n = self.conn.lock().expect("db").execute(
            "UPDATE connector_tools SET permission=?3, changed=0, updated_at=?4 WHERE connector_id=?1 AND name=?2",
            params![connector_id, tool, permission, now_ms()],
        )?;
        if n == 0 {
            return Err(McpError::NotFound(format!("tool {tool}")));
        }
        Ok(())
    }

    // ---- audit ----

    /// Records one call (arguments cut at 8 KB) and keeps only the newest 5000 rows.
    pub fn audit_add(
        &self,
        connector_id: &str,
        tool: &str,
        args_json: &str,
        outcome: &str,
        duration_ms: i64,
        result_chars: i64,
    ) -> Result<(), McpError> {
        let mut end = args_json.len().min(AUDIT_ARGS_MAX);
        while end > 0 && !args_json.is_char_boundary(end) {
            end -= 1;
        }
        let conn = self.conn.lock().expect("db");
        conn.execute(
            "INSERT INTO connector_audit (connector_id, tool, args_json, outcome, duration_ms, result_chars, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![connector_id, tool, &args_json[..end], outcome, duration_ms, result_chars, now_ms()],
        )?;
        conn.execute(
            "DELETE FROM connector_audit WHERE id <= (SELECT id FROM connector_audit ORDER BY id DESC LIMIT 1 OFFSET ?1)",
            params![AUDIT_KEEP],
        )?;
        Ok(())
    }

    pub fn audit_list(&self, connector_id: Option<&str>, limit: usize) -> Result<Vec<AuditRow>, McpError> {
        let limit = limit.clamp(1, 1000) as i64;
        let conn = self.conn.lock().expect("db");
        let map = |r: &rusqlite::Row<'_>| -> rusqlite::Result<AuditRow> {
            Ok(AuditRow {
                id: r.get(0)?,
                connector_id: r.get(1)?,
                tool: r.get(2)?,
                args_json: r.get(3)?,
                outcome: r.get(4)?,
                duration_ms: r.get(5)?,
                result_chars: r.get(6)?,
                created_at: r.get(7)?,
            })
        };
        let sql_all = "SELECT id, connector_id, tool, args_json, outcome, duration_ms, result_chars, created_at
                       FROM connector_audit ORDER BY id DESC LIMIT ?1";
        let sql_one = "SELECT id, connector_id, tool, args_json, outcome, duration_ms, result_chars, created_at
                       FROM connector_audit WHERE connector_id=?2 ORDER BY id DESC LIMIT ?1";
        let rows = match connector_id {
            Some(id) => conn
                .prepare(sql_one)?
                .query_map(params![limit, id], map)?
                .collect::<Result<Vec<_>, _>>()?,
            None => conn.prepare(sql_all)?.query_map(params![limit], map)?.collect::<Result<Vec<_>, _>>()?,
        };
        Ok(rows)
    }

    pub fn audit_count(&self) -> Result<i64, McpError> {
        Ok(self
            .conn
            .lock()
            .expect("db")
            .query_row("SELECT COUNT(*) FROM connector_audit", [], |r| r.get(0))?)
    }
}
