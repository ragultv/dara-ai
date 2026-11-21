//! Connectors, their connections, and the rules about what may be called.
//!
//! The user alone decides what a tool may do (allow, ask, deny). A server's own claims (for example `readOnlyHint`) are shown
//! but never grant anything, and a tool whose definition changes goes back to "ask".

use crate::client::{McpClient, RemoteTool, ServerInfo, ToolResult};
use crate::error::McpError;
use crate::secrets::{env_key, SecretStore};
use crate::store::{now_ms, AuditRow, ConnectorRow, Store, ToolRow};
use crate::transport::{
    check_http_url, HttpTransport, Shared, StdioConfig, StdioTransport, TokenProvider, Transport,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const MAX_CONNECTORS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Stdio,
    Http,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Stdio => "stdio",
            Kind::Http => "http",
        }
    }
    fn parse(s: &str) -> Kind {
        if s == "http" {
            Kind::Http
        } else {
            Kind::Stdio
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorConfig {
    pub id: String,
    pub name: String,
    pub kind: Kind,
    pub command: Option<String>,
    pub args: Vec<String>,
    /// Only the names: the values are in the secret store.
    pub env_names: Vec<String>,
    pub url: Option<String>,
    pub enabled: bool,
    pub created_at: i64,
}

impl From<ConnectorRow> for ConnectorConfig {
    fn from(r: ConnectorRow) -> Self {
        ConnectorConfig {
            id: r.id,
            name: r.name,
            kind: Kind::parse(&r.kind),
            command: r.command,
            args: r.args,
            env_names: r.env_names,
            url: r.url,
            enabled: r.enabled,
            created_at: r.created_at,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NewConnector {
    pub name: String,
    pub kind: Kind,
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Variable name -> value. The values go to the secret store.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub url: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorPatch {
    pub name: Option<String>,
    pub enabled: Option<bool>,
    pub args: Option<Vec<String>>,
    /// Replaces the set of variables. A name with an empty value keeps the value already stored for it.
    pub env: Option<BTreeMap<String, String>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ConnState {
    Disconnected,
    Connecting,
    Ready,
    NeedsAuth,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnStatus {
    pub state: ConnState,
    pub error: Option<String>,
    pub server: Option<ServerInfo>,
    pub tool_count: usize,
    /// Some tool's definition changed since the user last chose its permission.
    pub tools_changed: bool,
    /// For `needs_auth`: where the server describes its sign-in (from `WWW-Authenticate`).
    pub resource_metadata: Option<String>,
}

impl ConnStatus {
    fn idle() -> Self {
        ConnStatus {
            state: ConnState::Disconnected,
            error: None,
            server: None,
            tool_count: 0,
            tools_changed: false,
            resource_metadata: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorView {
    pub config: ConnectorConfig,
    pub status: ConnStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolView {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub input_schema: Value,
    /// What the server claims; informational only.
    pub read_only_hint: Option<bool>,
    pub permission: String,
    pub changed: bool,
    /// False when the server no longer lists the tool.
    pub available: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    Error,
    NeedsApproval,
    Denied,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CallOutcome {
    pub outcome: Outcome,
    pub content: Vec<Value>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
    pub truncated: bool,
}

impl CallOutcome {
    /// For the caller that turns `McpError::Denied` / `NeedsApproval` into a normal answer.
    pub fn without_result(outcome: Outcome) -> Self {
        CallOutcome { outcome, content: vec![], structured_content: None, is_error: false, truncated: false }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditEntry {
    pub id: i64,
    pub connector_id: String,
    pub tool: String,
    pub args_json: String,
    pub outcome: String,
    pub duration_ms: i64,
    pub result_chars: i64,
    pub created_at: i64,
}

impl From<AuditRow> for AuditEntry {
    fn from(r: AuditRow) -> Self {
        AuditEntry {
            id: r.id,
            connector_id: r.connector_id,
            tool: r.tool,
            args_json: r.args_json,
            outcome: r.outcome,
            duration_ms: r.duration_ms,
            result_chars: r.result_chars,
            created_at: r.created_at,
        }
    }
}

pub struct Manager {
    store: Arc<Store>,
    secrets: Arc<dyn SecretStore>,
    scratch_root: PathBuf,
    live: tokio::sync::Mutex<HashMap<String, Arc<McpClient>>>,
    status: Mutex<HashMap<String, ConnStatus>>,
    connect_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    tokens: Mutex<HashMap<String, Arc<dyn TokenProvider>>>,
    last_stderr: Mutex<HashMap<String, String>>,
}

fn valid_env_name(n: &str) -> bool {
    let mut chars = n.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && n.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn check_text(label: &str, s: &str, max: usize) -> Result<(), McpError> {
    if s.contains('\0') || s.contains('\n') || s.contains('\r') {
        return Err(McpError::Invalid(format!("{label} must be one line of plain text")));
    }
    if s.chars().count() > max {
        return Err(McpError::Invalid(format!("{label} is longer than {max} characters")));
    }
    Ok(())
}

fn check_name(name: &str) -> Result<String, McpError> {
    let name = name.trim();
    let n = name.chars().count();
    if n == 0 || n > 60 {
        return Err(McpError::Invalid("the name must be 1 to 60 characters".into()));
    }
    check_text("the name", name, 60)?;
    Ok(name.to_string())
}

fn check_args(args: &[String]) -> Result<(), McpError> {
    if args.len() > 64 {
        return Err(McpError::Invalid("at most 64 arguments".into()));
    }
    for a in args {
        if a.contains('\0') || a.chars().count() > 2000 {
            return Err(McpError::Invalid("an argument is too long or contains a NUL".into()));
        }
    }
    Ok(())
}

fn check_env(env: &BTreeMap<String, String>, allow_empty: bool) -> Result<(), McpError> {
    if env.len() > 32 {
        return Err(McpError::Invalid("at most 32 environment variables".into()));
    }
    for (k, v) in env {
        if !valid_env_name(k) {
            return Err(McpError::Invalid(format!("{k} is not a valid variable name")));
        }
        if v.contains('\0') || v.chars().count() > 2000 {
            return Err(McpError::Invalid(format!("the value of {k} is too long or contains a NUL")));
        }
        if v.is_empty() && !allow_empty {
            return Err(McpError::Invalid(format!("{k} needs a value")));
        }
    }
    Ok(())
}

fn text_chars(content: &[Value]) -> i64 {
    content
        .iter()
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .map(|t| t.chars().count() as i64)
        .sum()
}

impl Manager {
    pub fn new(db_path: &Path, secrets: Arc<dyn SecretStore>, scratch_root: &Path) -> Result<Arc<Self>, McpError> {
        Ok(Arc::new(Self {
            store: Arc::new(Store::open(db_path)?),
            secrets,
            scratch_root: scratch_root.to_path_buf(),
            live: tokio::sync::Mutex::new(HashMap::new()),
            status: Mutex::new(HashMap::new()),
            connect_locks: Mutex::new(HashMap::new()),
            tokens: Mutex::new(HashMap::new()),
            last_stderr: Mutex::new(HashMap::new()),
        }))
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    /// The hook the sign-in code will use: a bearer token for one HTTP connector.
    pub fn set_token_provider(&self, connector_id: &str, provider: Arc<dyn TokenProvider>) {
        self.tokens.lock().expect("tokens").insert(connector_id.to_string(), provider);
    }

    // ---- connectors ----

    pub fn add(&self, input: NewConnector) -> Result<ConnectorConfig, McpError> {
        if self.store.count_connectors()? >= MAX_CONNECTORS {
            return Err(McpError::Invalid(format!("at most {MAX_CONNECTORS} connectors")));
        }
        let name = check_name(&input.name)?;
        let (command, url) = match input.kind {
            Kind::Stdio => {
                let c = input.command.as_deref().unwrap_or("").trim();
                if c.is_empty() {
                    return Err(McpError::Invalid("a command is needed".into()));
                }
                check_text("the command", c, 500)?;
                check_args(&input.args)?;
                check_env(&input.env, false)?;
                (Some(c.to_string()), None)
            }
            Kind::Http => {
                let url = check_http_url(input.url.as_deref().unwrap_or(""))?;
                if !input.env.is_empty() {
                    return Err(McpError::Invalid("an http connector has no environment variables".into()));
                }
                (None, Some(url.to_string()))
            }
        };
        let id = uuid::Uuid::new_v4().to_string();
        // secrets first: if the credential store refuses, nothing half-exists in the database
        let mut stored: Vec<String> = Vec::new();
        for (k, v) in &input.env {
            if let Err(e) = self.secrets.set(&env_key(&id, k), v) {
                for done in &stored {
                    let _ = self.secrets.delete(&env_key(&id, done));
                }
                return Err(e);
            }
            stored.push(k.clone());
        }
        let row = ConnectorRow {
            id,
            name,
            kind: input.kind.as_str().into(),
            command,
            args: if input.kind == Kind::Stdio { input.args } else { vec![] },
            env_names: input.env.keys().cloned().collect(),
            url,
            enabled: true,
            created_at: now_ms(),
        };
        if let Err(e) = self.store.insert_connector(&row) {
            for k in &row.env_names {
                let _ = self.secrets.delete(&env_key(&row.id, k));
            }
            return Err(e);
        }
        Ok(row.into())
    }

    pub async fn update(&self, id: &str, patch: ConnectorPatch) -> Result<ConnectorConfig, McpError> {
        let mut row = self
            .store
            .get_connector(id)?
            .ok_or_else(|| McpError::NotFound(format!("connector {id}")))?;
        if let Some(name) = &patch.name {
            row.name = check_name(name)?;
        }
        if let Some(args) = &patch.args {
            if row.kind != "stdio" {
                return Err(McpError::Invalid("only a stdio connector has arguments".into()));
            }
            check_args(args)?;
            row.args = args.clone();
        }
        if let Some(env) = &patch.env {
            if row.kind != "stdio" {
                return Err(McpError::Invalid("only a stdio connector has environment variables".into()));
            }
            check_env(env, true)?;
            for (k, v) in env {
                if v.is_empty() && self.secrets.get(&env_key(id, k))?.is_none() {
                    return Err(McpError::Invalid(format!("{k} needs a value")));
                }
            }
            for (k, v) in env {
                if !v.is_empty() {
                    self.secrets.set(&env_key(id, k), v)?;
                }
            }
            for old in &row.env_names {
                if !env.contains_key(old) {
                    self.secrets.delete(&env_key(id, old))?;
                }
            }
            row.env_names = env.keys().cloned().collect();
        }
        if let Some(enabled) = patch.enabled {
            row.enabled = enabled;
        }
        self.store.update_connector(&row)?;
        let was_connected = self.live.lock().await.contains_key(id);
        if !row.enabled || (was_connected && (patch.args.is_some() || patch.env.is_some())) {
            // a changed command line or environment only applies to a new process
            self.disconnect(id).await;
        }
        Ok(row.into())
    }

    pub async fn remove(&self, id: &str) -> Result<(), McpError> {
        let row = self.store.get_connector(id)?;
        self.disconnect(id).await;
        if let Some(row) = row {
            for k in &row.env_names {
                self.secrets.delete(&env_key(id, k))?;
            }
        }
        self.store.delete_connector(id)?;
        self.status.lock().expect("status").remove(id);
        self.last_stderr.lock().expect("stderr").remove(id);
        self.tokens.lock().expect("tokens").remove(id);
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<ConnectorView>, McpError> {
        let rows = self.store.list_connectors()?;
        let status = self.status.lock().expect("status");
        Ok(rows
            .into_iter()
            .map(|r| {
                let st = status.get(&r.id).cloned().unwrap_or_else(ConnStatus::idle);
                ConnectorView { config: r.into(), status: st }
            })
            .collect())
    }

    pub fn status_of(&self, id: &str) -> ConnStatus {
        self.status
            .lock()
            .expect("status")
            .get(id)
            .cloned()
            .unwrap_or_else(ConnStatus::idle)
    }

    fn set_status(&self, id: &str, st: ConnStatus) {
        self.status.lock().expect("status").insert(id.to_string(), st);
    }

    // ---- connections ----

    fn lock_for(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(
            self.connect_locks
                .lock()
                .expect("locks")
                .entry(id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(()))),
        )
    }

    async fn open(&self, row: &ConnectorRow) -> Result<(Arc<McpClient>, Vec<RemoteTool>), McpError> {
        let shared = Shared::new();
        let transport: Arc<dyn Transport> = match row.kind.as_str() {
            "stdio" => {
                let mut env = Vec::new();
                for name in &row.env_names {
                    let v = self
                        .secrets
                        .get(&env_key(&row.id, name))?
                        .ok_or_else(|| McpError::Invalid(format!("the value of {name} is missing from the credential store")))?;
                    env.push((name.clone(), v));
                }
                Arc::new(StdioTransport::spawn(
                    StdioConfig {
                        command: row.command.clone().unwrap_or_default(),
                        args: row.args.clone(),
                        env,
                        cwd: self.scratch_root.join(&row.id),
                    },
                    Arc::clone(&shared),
                )?)
            }
            _ => {
                let token = self.tokens.lock().expect("tokens").get(&row.id).cloned();
                Arc::new(HttpTransport::new(row.url.as_deref().unwrap_or(""), token, Arc::clone(&shared))?)
            }
        };
        let client = match McpClient::connect(Arc::clone(&transport), shared).await {
            Ok(c) => c,
            Err(e) => {
                self.last_stderr
                    .lock()
                    .expect("stderr")
                    .insert(row.id.clone(), transport.diagnostics());
                transport.close().await;
                return Err(e);
            }
        };
        match client.list_tools().await {
            Ok(tools) => Ok((Arc::new(client), tools)),
            Err(e) => {
                self.last_stderr
                    .lock()
                    .expect("stderr")
                    .insert(row.id.clone(), client.diagnostics());
                client.close().await;
                Err(e)
            }
        }
    }

    fn ready_status(&self, id: &str, client: &McpClient, tool_count: usize) -> Result<ConnStatus, McpError> {
        let tools = self.store.tools(id)?;
        Ok(ConnStatus {
            state: ConnState::Ready,
            error: None,
            server: Some(client.server_info()),
            tool_count,
            tools_changed: tools.iter().any(|t| t.changed),
            resource_metadata: None,
        })
    }

    /// Starts the server (or reaches it), shakes hands and reads its tool list. Safe to call again: it starts over.
    pub async fn connect(&self, id: &str) -> Result<ConnStatus, McpError> {
        let lock = self.lock_for(id);
        let _guard = lock.lock().await;
        let row = self
            .store
            .get_connector(id)?
            .ok_or_else(|| McpError::NotFound(format!("connector {id}")))?;
        if !row.enabled {
            return Err(McpError::Invalid("this connector is turned off".into()));
        }
        self.drop_live(id).await;
        self.set_status(id, ConnStatus { state: ConnState::Connecting, ..ConnStatus::idle() });
        match self.open(&row).await {
            Ok((client, tools)) => {
                self.store.sync_tools(id, &tools)?;
                let status = self.ready_status(id, &client, tools.len())?;
                self.live.lock().await.insert(id.to_string(), client);
                self.set_status(id, status.clone());
                Ok(status)
            }
            Err(e) => {
                let status = match &e {
                    McpError::AuthRequired { resource_metadata } => ConnStatus {
                        state: ConnState::NeedsAuth,
                        error: Some("this server needs you to sign in".into()),
                        resource_metadata: resource_metadata.clone(),
                        ..ConnStatus::idle()
                    },
                    other => ConnStatus {
                        state: ConnState::Error,
                        error: Some(other.to_string()),
                        ..ConnStatus::idle()
                    },
                };
                self.set_status(id, status.clone());
                Err(e)
            }
        }
    }

    async fn drop_live(&self, id: &str) {
        let client = self.live.lock().await.remove(id);
        if let Some(c) = client {
            self.last_stderr.lock().expect("stderr").insert(id.to_string(), c.diagnostics());
            c.close().await;
        }
    }

    pub async fn disconnect(&self, id: &str) {
        self.drop_live(id).await;
        let mut st = self.status_of(id);
        if st.state != ConnState::Disconnected {
            st = ConnStatus { tool_count: st.tool_count, tools_changed: st.tools_changed, ..ConnStatus::idle() };
            self.set_status(id, st);
        }
    }

    /// The live client, connecting first when there is none (reconnect on demand).
    async fn ensure_ready(&self, id: &str) -> Result<Arc<McpClient>, McpError> {
        if let Some(c) = self.live.lock().await.get(id).cloned() {
            return Ok(c);
        }
        self.connect(id).await?;
        self.live
            .lock()
            .await
            .get(id)
            .cloned()
            .ok_or_else(|| McpError::Transport("the connection was lost".into()))
    }

    /// Reads the tool list again (the server said it changed) and applies the permission rules.
    async fn refresh_tools(&self, id: &str, client: &McpClient) -> Result<(), McpError> {
        let tools = client.list_tools().await?;
        self.store.sync_tools(id, &tools)?;
        let st = self.ready_status(id, client, tools.len())?;
        self.set_status(id, st);
        Ok(())
    }

    // ---- tools ----

    pub fn tools(&self, id: &str) -> Result<Vec<ToolView>, McpError> {
        if self.store.get_connector(id)?.is_none() {
            return Err(McpError::NotFound(format!("connector {id}")));
        }
        Ok(self.store.tools(id)?.into_iter().map(tool_view).collect())
    }

    pub fn set_permission(&self, id: &str, tool: &str, permission: &str) -> Result<(), McpError> {
        self.store.set_permission(id, tool, permission)?;
        let mut st = self.status_of(id);
        st.tools_changed = self.store.tools(id)?.iter().any(|t| t.changed);
        if st.state == ConnState::Ready {
            self.set_status(id, st);
        }
        Ok(())
    }

    /// Calls a tool if the user's permission allows it.
    ///
    /// - "deny": `Err(Denied)`, nothing is sent.
    /// - "ask" and not `approved`: `Err(NeedsApproval)`, nothing is sent.
    /// - otherwise the call is made. If the server's tool list changed first and this tool's definition changed with it, the
    ///   call is NOT made: its permission is back at "ask", so `Err(NeedsApproval)` (what the user approved is not what is there now).
    pub async fn call(
        &self,
        id: &str,
        tool: &str,
        arguments: Value,
        approved: bool,
        timeout: Option<Duration>,
    ) -> Result<CallOutcome, McpError> {
        let args_text = arguments.to_string();
        let audit = |outcome: &str, ms: i64, chars: i64| {
            let _ = self.store.audit_add(id, tool, &args_text, outcome, ms, chars);
        };
        if self.store.get_connector(id)?.is_none() {
            return Err(McpError::NotFound(format!("connector {id}")));
        }
        let row = self
            .store
            .tool(id, tool)?
            .ok_or_else(|| McpError::NotFound(format!("tool {tool}")))?;
        if !row.available {
            return Err(McpError::NotFound(format!("tool {tool} (the server no longer offers it)")));
        }
        match (row.permission.as_str(), approved) {
            ("deny", _) => {
                audit("denied", 0, 0);
                return Err(McpError::Denied);
            }
            ("ask", false) => {
                audit("needs_approval", 0, 0);
                return Err(McpError::NeedsApproval);
            }
            _ => {}
        }

        let client = self.ensure_ready(id).await?;
        if client.tools_stale() {
            self.refresh_tools(id, &client).await?;
            let now = self
                .store
                .tool(id, tool)?
                .filter(|t| t.available)
                .ok_or_else(|| McpError::NotFound(format!("tool {tool} (the server no longer offers it)")))?;
            if now.fingerprint != row.fingerprint {
                audit("needs_approval", 0, 0);
                return Err(McpError::NeedsApproval);
            }
        }

        let started = Instant::now();
        let result: Result<ToolResult, McpError> = client.call_tool(tool, arguments, timeout).await;
        let ms = started.elapsed().as_millis() as i64;
        match result {
            Ok(r) => {
                audit(if r.is_error { "error" } else { "ok" }, ms, text_chars(&r.content));
                Ok(CallOutcome {
                    outcome: if r.is_error { Outcome::Error } else { Outcome::Ok },
                    content: r.content,
                    structured_content: r.structured_content,
                    is_error: r.is_error,
                    truncated: r.truncated,
                })
            }
            Err(e) => {
                audit("error", ms, 0);
                if matches!(e, McpError::Closed(_)) {
                    // the server is gone: forget it, so the next call connects again (this call is not repeated: it may have had effects)
                    self.drop_live(id).await;
                    self.set_status(
                        id,
                        ConnStatus { state: ConnState::Error, error: Some(e.to_string()), ..ConnStatus::idle() },
                    );
                }
                Err(e)
            }
        }
    }

    // ---- diagnostics ----

    pub fn audit(&self, id: Option<&str>, limit: usize) -> Result<Vec<AuditEntry>, McpError> {
        Ok(self.store.audit_list(id, limit)?.into_iter().map(AuditEntry::from).collect())
    }

    pub async fn stderr(&self, id: &str) -> String {
        if let Some(c) = self.live.lock().await.get(id) {
            return c.diagnostics();
        }
        self.last_stderr.lock().expect("stderr").get(id).cloned().unwrap_or_default()
    }

    /// Ends every connection (stops child processes). Call when the app exits.
    pub async fn close_all(&self) {
        let ids: Vec<String> = self.live.lock().await.keys().cloned().collect();
        for id in ids {
            self.disconnect(&id).await;
        }
    }
}

fn tool_view(t: ToolRow) -> ToolView {
    let m = &t.meta;
    ToolView {
        name: t.name,
        title: m.get("title").and_then(Value::as_str).map(str::to_string),
        description: m.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
        input_schema: m.get("inputSchema").cloned().unwrap_or(Value::Null),
        read_only_hint: m.get("readOnlyHint").and_then(Value::as_bool),
        permission: t.permission,
        changed: t.changed,
        available: t.available,
    }
}
