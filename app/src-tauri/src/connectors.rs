//! Tauri commands for connectors (MCP servers). All the logic is in the `dara-mcp` crate; this file only holds the state and maps
//! its errors to plain strings for the UI.

use crate::engine_settings;
use dara_mcp::manager::{AuditEntry, Outcome};
use dara_mcp::{
    CallOutcome, ConnStatus, ConnectorConfig, ConnectorPatch, ConnectorView, KeyringSecrets, Manager, McpError,
    NewConnector, ToolView,
};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

pub struct ConnectorsState {
    /// `Err` when the database could not be opened: every command then says why.
    manager: Result<Arc<Manager>, String>,
}

impl ConnectorsState {
    /// The database is `connectors.db` in the Dara data folder; secrets are in the Windows Credential Manager.
    pub fn new() -> Self {
        let dir = engine_settings::config_dir();
        let manager = std::fs::create_dir_all(&dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))
            .and_then(|_| {
                Manager::new(&dir.join("connectors.db"), Arc::new(KeyringSecrets::new("Dara")), &dir.join("connectors"))
                    .map_err(|e| e.to_string())
            });
        if let Err(e) = &manager {
            eprintln!("connectors are not available: {e}");
        }
        Self { manager }
    }

    pub fn manager(&self) -> Option<Arc<Manager>> {
        self.manager.as_ref().ok().cloned()
    }

    fn get(&self) -> Result<Arc<Manager>, String> {
        self.manager
            .as_ref()
            .map(Arc::clone)
            .map_err(|e| format!("connectors are not available: {e}"))
    }
}

impl Default for ConnectorsState {
    fn default() -> Self {
        Self::new()
    }
}

fn err(e: McpError) -> String {
    e.to_string()
}

#[tauri::command]
pub async fn connectors_list(state: tauri::State<'_, ConnectorsState>) -> Result<Vec<ConnectorView>, String> {
    state.get()?.list().map_err(err)
}

#[tauri::command]
pub async fn connector_add(
    state: tauri::State<'_, ConnectorsState>,
    input: NewConnector,
) -> Result<ConnectorConfig, String> {
    state.get()?.add(input).map_err(err)
}

#[tauri::command]
pub async fn connector_update(
    state: tauri::State<'_, ConnectorsState>,
    id: String,
    patch: ConnectorPatch,
) -> Result<ConnectorConfig, String> {
    state.get()?.update(&id, patch).await.map_err(err)
}

#[tauri::command]
pub async fn connector_remove(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<(), String> {
    state.get()?.remove(&id).await.map_err(err)
}

#[tauri::command]
pub async fn connector_connect(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<ConnStatus, String> {
    let m = state.get()?;
    match m.connect(&id).await {
        Ok(status) => Ok(status),
        // a failed connection is a state to show (error, needs sign-in), not a failed command
        Err(McpError::NotFound(w)) => Err(format!("not found: {w}")),
        Err(McpError::Invalid(w)) => Err(w),
        Err(_) => Ok(m.status_of(&id)),
    }
}

#[tauri::command]
pub async fn connector_disconnect(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<(), String> {
    state.get()?.disconnect(&id).await;
    Ok(())
}

#[tauri::command]
pub async fn connector_tools(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<Vec<ToolView>, String> {
    state.get()?.tools(&id).map_err(err)
}

#[tauri::command]
pub async fn connector_set_permission(
    state: tauri::State<'_, ConnectorsState>,
    id: String,
    tool: String,
    permission: String,
) -> Result<(), String> {
    state.get()?.set_permission(&id, &tool, &permission).map_err(err)
}

/// Calls a tool. "denied" and "needs_approval" are normal answers (the UI asks the user, then calls again with `approved: true`).
#[tauri::command]
pub async fn connector_call(
    state: tauri::State<'_, ConnectorsState>,
    id: String,
    tool: String,
    args: Option<Value>,
    approved: Option<bool>,
    timeout_ms: Option<u64>,
) -> Result<CallOutcome, String> {
    let m = state.get()?;
    let timeout = timeout_ms.map(Duration::from_millis);
    match m
        .call(&id, &tool, args.unwrap_or(Value::Null), approved.unwrap_or(false), timeout)
        .await
    {
        Ok(r) => Ok(r),
        Err(McpError::Denied) => Ok(CallOutcome::without_result(Outcome::Denied)),
        Err(McpError::NeedsApproval) => Ok(CallOutcome::without_result(Outcome::NeedsApproval)),
        Err(e) => Err(err(e)),
    }
}

#[tauri::command]
pub async fn connector_audit(
    state: tauri::State<'_, ConnectorsState>,
    id: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<AuditEntry>, String> {
    state.get()?.audit(id.as_deref(), limit.unwrap_or(100)).map_err(err)
}

#[tauri::command]
pub async fn connector_stderr(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<String, String> {
    Ok(state.get()?.stderr(&id).await)
}

/// Signs in to an http connector: opens the server's sign-in page in the system browser and waits (up to five minutes) for the answer
/// on a loopback address. Returns the status after connecting; a failed or cancelled sign-in is an error message.
#[tauri::command]
pub async fn connector_sign_in(
    app: tauri::AppHandle,
    state: tauri::State<'_, ConnectorsState>,
    id: String,
) -> Result<ConnStatus, String> {
    use tauri_plugin_opener::OpenerExt;
    let m = state.get()?;
    let open = move |url: &str| {
        // only web addresses ever go to the browser
        if url.starts_with("https://") || url.starts_with("http://") {
            if let Err(e) = app.opener().open_url(url, None::<&str>) {
                eprintln!("could not open the browser: {e}");
            }
        }
    };
    m.sign_in(&id, &open).await.map_err(err)
}

/// Ends a sign-in that is waiting for the browser. `true` when there was one.
#[tauri::command]
pub async fn connector_cancel_sign_in(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<bool, String> {
    Ok(state.get()?.cancel_sign_in(&id))
}

/// Revokes at the server when it can, forgets the tokens and disconnects.
#[tauri::command]
pub async fn connector_sign_out(state: tauri::State<'_, ConnectorsState>, id: String) -> Result<ConnStatus, String> {
    state.get()?.sign_out(&id).await.map_err(err)
}
