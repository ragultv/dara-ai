//! Connectors, their connections, and the rules about what may be called.
//!
//! The user alone decides what a tool may do (allow, ask, deny). A server's own claims (for example `readOnlyHint`) are shown
//! but never grant anything, and a tool whose definition changes goes back to "ask".

use crate::client::{McpClient, RemoteTool, ServerInfo, ToolResult};
use crate::error::McpError;
use crate::oauth::{self, ClientSource, OAuthTokens, SignInParams, DEFAULT_SIGN_IN_TIMEOUT};
use crate::secrets::{self, env_key, SecretStore};
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
    pub oauth: OAuthInfo,
}

/// The sign-in settings of an http connector. The secret itself is never shown, only that there is one.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthInfo {
    pub client_id: Option<String>,
    pub scope: Option<String>,
    pub has_secret: bool,
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
            oauth: OAuthInfo { client_id: r.oauth_client_id, scope: r.oauth_scope, has_secret: false },
        }
    }
}

/// What the user may type for sign-in: a client id (and secret) when the server cannot register Dara by itself, and a scope.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OAuthInput {
    pub client_id: Option<String>,
    /// In a patch, an empty string keeps the secret already stored.
    pub client_secret: Option<String>,
    pub scope: Option<String>,
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
    /// http connectors only
    #[serde(default)]
    pub oauth: Option<OAuthInput>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorPatch {
    pub name: Option<String>,
    pub enabled: Option<bool>,
    pub args: Option<Vec<String>>,
    /// Replaces the set of variables. A name with an empty value keeps the value already stored for it.
    pub env: Option<BTreeMap<String, String>>,
    /// http connectors only. `clientId: ""` forgets the client id, its secret and the tokens; `scope: ""` clears the scope.
    pub oauth: Option<OAuthInput>,
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
    /// Tokens are stored for this connector (an http connector the user signed in to).
    pub signed_in: bool,
    /// The server said the token is not enough (403 `insufficient_scope`): the scope it asked for. Set together with `needs_auth`;
    /// the user decides whether to sign in again with it.
    pub needed_scope: Option<String>,
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
            signed_in: false,
            needed_scope: None,
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
    oauth: Mutex<HashMap<String, Arc<OAuthTokens>>>,
    sign_ins: Mutex<HashMap<String, Arc<tokio::sync::Notify>>>,
    sign_in_timeout: Mutex<Duration>,
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

fn more_access_text(scope: Option<&str>) -> String {
    match scope {
        Some(s) if !s.is_empty() => format!("this server needs more access ({s}); sign in again to allow it"),
        _ => "this server needs more access; sign in again to allow it".into(),
    }
}

/// Existing scope plus the one that is needed, each word once.
fn union_scopes(granted: Option<&str>, needed: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    for w in granted.unwrap_or("").split_whitespace().chain(needed.split_whitespace()) {
        if !out.contains(&w) {
            out.push(w);
        }
    }
    out.join(" ")
}

const TOKEN_KEYS: [&str; 5] = ["authorization", "access_token", "refresh_token", "id_token", "client_secret"];

/// Text with every known secret and every `Bearer ...` removed.
fn scrub_text(s: &str, known: &[String]) -> String {
    let mut out = s.to_string();
    for k in known.iter().filter(|k| k.len() >= 8) {
        out = out.replace(k.as_str(), "[removed]");
    }
    // "Bearer <token>" in any case, whatever the token is
    let lower = out.to_ascii_lowercase();
    let mut result = String::with_capacity(out.len());
    let mut i = 0;
    while let Some(found) = lower[i..].find("bearer ") {
        let at = i + found;
        let start = at + "bearer ".len();
        let end = out[start..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || "._~+/=-".contains(c)))
            .map(|n| start + n)
            .unwrap_or(out.len());
        result.push_str(&out[i..start]);
        if end > start {
            result.push_str("[removed]");
        }
        i = end;
    }
    result.push_str(&out[i..]);
    result
}

fn scrub_value(v: &Value, known: &[String]) -> Value {
    match v {
        Value::String(s) => Value::String(scrub_text(s, known)),
        Value::Array(a) => Value::Array(a.iter().map(|x| scrub_value(x, known)).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, x)| {
                    let secret_key = TOKEN_KEYS.iter().any(|t| k.eq_ignore_ascii_case(t));
                    (k.clone(), if secret_key && x.is_string() { Value::String("[removed]".into()) } else { scrub_value(x, known) })
                })
                .collect(),
        ),
        other => other.clone(),
    }
}

/// What the audit log keeps of a call's arguments: no token, no `Authorization` value, no secret of this connector.
pub fn scrub_args(args: &Value, known: &[String]) -> String {
    scrub_value(args, known).to_string()
}

/// The sign-in settings from a form: `(client id, client secret, scope)`. An empty value is no value.
fn check_oauth(kind: Kind, o: Option<&OAuthInput>) -> Result<(Option<String>, Option<String>, Option<String>), McpError> {
    let Some(o) = o else { return Ok((None, None, None)) };
    if kind != Kind::Http {
        return Err(McpError::Invalid("only a server address has sign-in settings".into()));
    }
    let one_line = |label: &str, v: &Option<String>, max: usize| -> Result<Option<String>, McpError> {
        match v.as_deref().map(str::trim) {
            Some(s) if !s.is_empty() => {
                check_text(label, s, max)?;
                Ok(Some(s.to_string()))
            }
            _ => Ok(None),
        }
    };
    let id = one_line("the client ID", &o.client_id, 300)?;
    let scope = one_line("the scope", &o.scope, 500)?;
    let secret = match o.client_secret.as_deref() {
        Some(s) if !s.is_empty() => {
            check_text("the client secret", s, 1500)?;
            Some(s.to_string())
        }
        _ => None,
    };
    if secret.is_some() && id.is_none() {
        return Err(McpError::Invalid("a client secret needs a client ID".into()));
    }
    Ok((id, secret, scope))
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
            oauth: Mutex::new(HashMap::new()),
            sign_ins: Mutex::new(HashMap::new()),
            sign_in_timeout: Mutex::new(DEFAULT_SIGN_IN_TIMEOUT),
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
        let (oauth_client_id, oauth_secret, oauth_scope) = check_oauth(input.kind, input.oauth.as_ref())?;
        let id = uuid::Uuid::new_v4().to_string();
        if let Some(sec) = &oauth_secret {
            self.secrets.set(&secrets::client_secret_key(&id), sec)?;
        }
        // secrets first: if the credential store refuses, nothing half-exists in the database
        let mut stored: Vec<String> = Vec::new();
        for (k, v) in &input.env {
            if let Err(e) = self.secrets.set(&env_key(&id, k), v) {
                for done in &stored {
                    let _ = self.secrets.delete(&env_key(&id, done));
                }
                let _ = self.secrets.delete(&secrets::client_secret_key(&id));
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
            oauth_client_id,
            oauth_scope,
            oauth_registered: false,
            oauth_redirect_uri: None,
        };
        if let Err(e) = self.store.insert_connector(&row) {
            for k in &row.env_names {
                let _ = self.secrets.delete(&env_key(&row.id, k));
            }
            let _ = self.secrets.delete(&secrets::client_secret_key(&row.id));
            return Err(e);
        }
        Ok(self.config_of(row))
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
        let mut forget_sign_in = false;
        if let Some(o) = &patch.oauth {
            if row.kind != "http" {
                return Err(McpError::Invalid("only a server address has sign-in settings".into()));
            }
            if let Some(cid) = &o.client_id {
                let cid = cid.trim();
                let changed = row.oauth_client_id.as_deref() != Some(cid) || row.oauth_registered;
                if cid.is_empty() {
                    forget_sign_in |= row.oauth_client_id.is_some();
                    row.oauth_client_id = None;
                } else {
                    check_text("the client ID", cid, 300)?;
                    forget_sign_in |= changed;
                    row.oauth_client_id = Some(cid.to_string());
                }
                if changed || cid.is_empty() {
                    // tokens and secret belonged to the other client
                    row.oauth_registered = false;
                    row.oauth_redirect_uri = None;
                    self.secrets.delete(&secrets::client_secret_key(id))?;
                }
            }
            if let Some(sec) = o.client_secret.as_deref().filter(|s| !s.is_empty()) {
                if row.oauth_client_id.is_none() {
                    return Err(McpError::Invalid("a client secret needs a client ID".into()));
                }
                check_text("the client secret", sec, 1500)?;
                self.secrets.set(&secrets::client_secret_key(id), sec)?;
            }
            if let Some(sc) = &o.scope {
                let sc = sc.trim();
                if sc.is_empty() {
                    row.oauth_scope = None;
                } else {
                    check_text("the scope", sc, 500)?;
                    row.oauth_scope = Some(sc.to_string());
                }
            }
        }
        self.store.update_connector(&row)?;
        if forget_sign_in {
            self.oauth_provider(id)?.clear()?;
        }
        let was_connected = self.live.lock().await.contains_key(id);
        if !row.enabled || forget_sign_in || (was_connected && (patch.args.is_some() || patch.env.is_some())) {
            // a changed command line or environment only applies to a new process
            self.disconnect(id).await;
        }
        Ok(self.config_of(row))
    }

    pub async fn remove(&self, id: &str) -> Result<(), McpError> {
        let row = self.store.get_connector(id)?;
        self.disconnect(id).await;
        if let Some(row) = row {
            for k in &row.env_names {
                self.secrets.delete(&env_key(id, k))?;
            }
            if row.kind == "http" {
                let _ = self.forget_sign_in(id, true).await; // best effort: the connector is going away either way
                self.secrets.delete(&secrets::client_secret_key(id))?;
            }
        }
        self.store.delete_connector(id)?;
        self.status.lock().expect("status").remove(id);
        self.last_stderr.lock().expect("stderr").remove(id);
        self.tokens.lock().expect("tokens").remove(id);
        self.oauth.lock().expect("oauth").remove(id);
        Ok(())
    }

    pub fn list(&self) -> Result<Vec<ConnectorView>, McpError> {
        let rows = self.store.list_connectors()?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let st = self.status_of(&r.id);
                ConnectorView { config: self.config_of(r), status: st }
            })
            .collect())
    }

    /// The row as the UI sees it: the sign-in settings include whether a client secret is stored (never the secret).
    fn config_of(&self, row: ConnectorRow) -> ConnectorConfig {
        let id = row.id.clone();
        let mut c: ConnectorConfig = row.into();
        c.oauth.has_secret = self.secrets.get(&secrets::client_secret_key(&id)).ok().flatten().is_some();
        c
    }

    pub fn status_of(&self, id: &str) -> ConnStatus {
        let mut st = self.status.lock().expect("status").get(id).cloned().unwrap_or_else(ConnStatus::idle);
        // signed in means the tokens can be read back whole (a half-written or damaged value is not a sign-in)
        st.signed_in = secrets::get_chunked(self.secrets.as_ref(), &secrets::token_key(id)).ok().flatten().is_some();
        st
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
                let token = self.provider_for(&row.id)?;
                Arc::new(HttpTransport::new(row.url.as_deref().unwrap_or(""), Some(token), Arc::clone(&shared))?)
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
            signed_in: false,
            needed_scope: None,
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
                self.set_status(id, status);
                Ok(self.status_of(id)) // with `signed_in` filled in
            }
            Err(e) => {
                let status = match &e {
                    McpError::AuthRequired { resource_metadata, .. } => ConnStatus {
                        state: ConnState::NeedsAuth,
                        error: Some("this server needs you to sign in".into()),
                        resource_metadata: resource_metadata.clone(),
                        ..ConnStatus::idle()
                    },
                    McpError::InsufficientScope { scope } => ConnStatus {
                        state: ConnState::NeedsAuth,
                        error: Some(more_access_text(scope.as_deref())),
                        needed_scope: scope.clone(),
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
        let args_text = scrub_args(&arguments, &self.known_secrets(id));
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
                if let McpError::InsufficientScope { scope } = &e {
                    // the connection is fine, this call is not allowed: the user decides whether to ask for more access
                    let mut st = self.status.lock().expect("status").get(id).cloned().unwrap_or_else(ConnStatus::idle);
                    st.state = ConnState::NeedsAuth;
                    st.error = Some(more_access_text(scope.as_deref()));
                    st.needed_scope = scope.clone();
                    self.set_status(id, st);
                }
                if let McpError::AuthRequired { resource_metadata, .. } = &e {
                    // the token was refused (and a refresh did not help): the user has to sign in again
                    self.drop_live(id).await;
                    self.set_status(
                        id,
                        ConnStatus {
                            state: ConnState::NeedsAuth,
                            error: Some("this server needs you to sign in".into()),
                            resource_metadata: resource_metadata.clone(),
                            ..ConnStatus::idle()
                        },
                    );
                }
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

    // ---- signing in ----

    /// Every secret value a connector holds (tokens, client secret, environment variable values), to keep them out of logs.
    fn known_secrets(&self, id: &str) -> Vec<String> {
        let mut v = self.oauth.lock().expect("oauth").get(id).map(|p| p.secret_values()).unwrap_or_default();
        v.extend(self.secrets.get(&secrets::client_secret_key(id)).ok().flatten());
        if let Ok(Some(row)) = self.store.get_connector(id) {
            for name in &row.env_names {
                v.extend(self.secrets.get(&env_key(id, name)).ok().flatten());
            }
        }
        v
    }

    /// The token holder of an http connector (its tokens live in the secret store; this refreshes them).
    pub fn oauth_tokens(&self, id: &str) -> Result<Arc<OAuthTokens>, McpError> {
        self.oauth_provider(id)
    }

    fn oauth_provider(&self, id: &str) -> Result<Arc<OAuthTokens>, McpError> {
        let mut m = self.oauth.lock().expect("oauth");
        if let Some(p) = m.get(id) {
            return Ok(Arc::clone(p));
        }
        let p = OAuthTokens::new(id, Arc::clone(&self.secrets))?;
        m.insert(id.to_string(), Arc::clone(&p));
        Ok(p)
    }

    /// A provider set with `set_token_provider` (tests, a custom scheme), else the OAuth one.
    fn provider_for(&self, id: &str) -> Result<Arc<dyn TokenProvider>, McpError> {
        if let Some(p) = self.tokens.lock().expect("tokens").get(id).cloned() {
            return Ok(p);
        }
        Ok(self.oauth_provider(id)? as Arc<dyn TokenProvider>)
    }

    /// How long the user has to finish in the browser (five minutes unless changed; tests shorten it).
    pub fn set_sign_in_timeout(&self, d: Duration) {
        *self.sign_in_timeout.lock().expect("timeout") = d;
    }

    /// Asks the server what it wants, with no token: the 401's pointer to its sign-in description and the scope it names.
    async fn probe_challenge(&self, url: &url::Url) -> Result<(Option<String>, Option<String>), McpError> {
        let shared = Shared::new();
        let transport = Arc::new(HttpTransport::new(url.as_str(), None, Arc::clone(&shared))?);
        match McpClient::connect(transport, shared).await {
            Err(McpError::AuthRequired { resource_metadata, scope }) => Ok((resource_metadata, scope)),
            Ok(client) => {
                client.close().await;
                Err(McpError::Invalid("this server does not ask you to sign in".into()))
            }
            Err(e) => Err(e),
        }
    }

    /// Signs in to an http connector: discovery, a client (the configured one, an earlier registration, or a new registration), the
    /// browser (`open_url` is called once with the address to open), the callback, the token exchange, then connects.
    /// One sign-in per connector at a time; `cancel_sign_in` ends it.
    pub async fn sign_in(&self, id: &str, open_url: &(dyn Fn(&str) + Send + Sync)) -> Result<ConnStatus, McpError> {
        let mut row = self
            .store
            .get_connector(id)?
            .ok_or_else(|| McpError::NotFound(format!("connector {id}")))?;
        if row.kind != "http" {
            return Err(McpError::Invalid("only a server address needs signing in".into()));
        }
        if !row.enabled {
            return Err(McpError::Invalid("this connector is turned off".into()));
        }
        let url = check_http_url(row.url.as_deref().unwrap_or(""))?;

        let cancel = Arc::new(tokio::sync::Notify::new());
        {
            let mut m = self.sign_ins.lock().expect("sign_ins");
            if m.contains_key(id) {
                return Err(McpError::Invalid("already waiting for you to sign in: finish in the browser or cancel".into()));
            }
            m.insert(id.to_string(), Arc::clone(&cancel));
        }
        struct Done<'a>(&'a Manager, &'a str);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.sign_ins.lock().expect("sign_ins").remove(self.1);
            }
        }
        let _done = Done(self, id);

        // after "needs more access" the new sign-in asks for what was granted plus what the server said it needs
        let needed = self.status.lock().expect("status").get(id).and_then(|s| s.needed_scope.clone());
        self.disconnect(id).await;
        let (resource_metadata, probed_scope) = self.probe_challenge(&url).await?;
        let challenge_scope = match needed {
            Some(n) => Some(union_scopes(self.oauth_provider(id)?.granted_scope().as_deref(), &n)),
            None => probed_scope,
        };
        let secret = self.secrets.get(&secrets::client_secret_key(id))?;
        let client = match (&row.oauth_client_id, row.oauth_registered) {
            (Some(cid), true) => ClientSource::Registered {
                client_id: cid.clone(),
                secret,
                redirect_uri: row.oauth_redirect_uri.clone().unwrap_or_default(),
            },
            (Some(cid), false) => ClientSource::Configured { client_id: cid.clone(), secret },
            (None, _) => ClientSource::None,
        };
        let timeout = *self.sign_in_timeout.lock().expect("timeout");
        let outcome = oauth::run_sign_in(
            SignInParams {
                mcp_url: url,
                resource_metadata,
                challenge_scope,
                configured_scope: row.oauth_scope.clone(),
                client,
                timeout,
            },
            open_url,
            &cancel,
        )
        .await?;

        if let Some(reg) = &outcome.registration {
            // keep what the server gave us, so the next sign-in does not have to register again
            row.oauth_client_id = Some(reg.client_id.clone());
            row.oauth_registered = true;
            row.oauth_redirect_uri = Some(reg.redirect_uri.clone());
            match &reg.client_secret {
                Some(s) => self.secrets.set(&secrets::client_secret_key(id), s)?,
                None => self.secrets.delete(&secrets::client_secret_key(id))?,
            }
            self.store.update_connector(&row)?;
        }
        self.oauth_provider(id)?.store(&outcome.tokens)?;
        // signed in: whatever the connection does next is a state to show, not a failed sign-in
        match self.connect(id).await {
            Ok(s) => Ok(s),
            Err(_) => Ok(self.status_of(id)),
        }
    }

    /// Ends a sign-in that is waiting for the browser. `true` when there was one.
    pub fn cancel_sign_in(&self, id: &str) -> bool {
        match self.sign_ins.lock().expect("sign_ins").get(id) {
            Some(n) => {
                n.notify_one();
                true
            }
            None => false,
        }
    }

    /// Tells the server the tokens are no longer wanted (best effort) and forgets them.
    async fn forget_sign_in(&self, id: &str, revoke: bool) -> Result<(), McpError> {
        let p = self.oauth_provider(id)?;
        if revoke && p.is_signed_in() {
            p.revoke().await;
        }
        p.clear()
    }

    /// Signs out: revokes at the server if it can, forgets the tokens, and disconnects. A client the server registered for Dara is
    /// forgotten too; a client id the user typed stays.
    pub async fn sign_out(&self, id: &str) -> Result<ConnStatus, McpError> {
        let mut row = self
            .store
            .get_connector(id)?
            .ok_or_else(|| McpError::NotFound(format!("connector {id}")))?;
        if row.kind != "http" {
            return Err(McpError::Invalid("only a server address has a sign-in".into()));
        }
        self.disconnect(id).await;
        self.forget_sign_in(id, true).await?;
        if row.oauth_registered {
            row.oauth_client_id = None;
            row.oauth_registered = false;
            row.oauth_redirect_uri = None;
            self.secrets.delete(&secrets::client_secret_key(id))?;
            self.store.update_connector(&row)?;
        }
        self.set_status(id, ConnStatus::idle());
        Ok(self.status_of(id))
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
