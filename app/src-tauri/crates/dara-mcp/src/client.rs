//! The Model Context Protocol client: handshake, listing tools, calling them.

use crate::error::McpError;
use crate::transport::{Shared, Transport};
use serde::Serialize;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Newest first. A server that answers `initialize` with an older one of these is spoken to in that version.
pub const SUPPORTED_PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

const SETUP_TIMEOUT: Duration = Duration::from_secs(20);
pub const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);
pub const MAX_CALL_TIMEOUT: Duration = Duration::from_secs(300);
/// Text in a tool result beyond this is cut off and the result is marked truncated.
pub const MAX_RESULT_TEXT_BYTES: usize = 1024 * 1024;
const MAX_PAGES: usize = 50;
const MAX_TOOLS: usize = 2000;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
    /// The protocol version both sides agreed on.
    pub protocol: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RemoteTool {
    pub name: String,
    pub title: Option<String>,
    pub description: String,
    pub input_schema: Value,
    pub annotations: Option<Value>,
    /// What the server claims. Shown to the user; never used to grant anything.
    pub read_only_hint: Option<bool>,
}

impl RemoteTool {
    /// The tool as the user saw it: a change in any of these resets its permission.
    pub fn fingerprint(&self) -> String {
        let basis = json!({
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": self.annotations,
        });
        hex::encode(Sha256::digest(canonical(&basis).as_bytes()))
    }

    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": self.annotations,
            "readOnlyHint": self.read_only_hint,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResult {
    /// Content blocks as the server sent them (text, image, resource, resource_link).
    pub content: Vec<Value>,
    pub structured_content: Option<Value>,
    pub is_error: bool,
    pub truncated: bool,
}

/// JSON with object keys sorted, so the same data always gives the same text.
pub fn canonical(v: &Value) -> String {
    fn go(v: &Value, out: &mut String) {
        match v {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String((*k).clone()).to_string());
                    out.push(':');
                    go(&map[*k], out);
                }
                out.push('}');
            }
            Value::Array(a) => {
                out.push('[');
                for (i, x) in a.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    go(x, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut s = String::new();
    go(v, &mut s);
    s
}

pub struct McpClient {
    transport: Arc<dyn Transport>,
    shared: Arc<Shared>,
    next_id: AtomicU64,
    info: Mutex<ServerInfo>,
}

impl McpClient {
    /// Connects: sends `initialize` and `notifications/initialized`.
    pub async fn connect(transport: Arc<dyn Transport>, shared: Arc<Shared>) -> Result<Self, McpError> {
        let client = Self {
            transport,
            shared,
            next_id: AtomicU64::new(1),
            info: Mutex::new(ServerInfo { name: String::new(), version: String::new(), protocol: String::new() }),
        };
        client.initialize().await?;
        Ok(client)
    }

    pub fn server_info(&self) -> ServerInfo {
        self.info.lock().expect("info").clone()
    }

    /// True after the server said its tool list changed (cleared by the next `list_tools`).
    pub fn tools_stale(&self) -> bool {
        self.shared.tools_stale.load(Ordering::SeqCst)
    }

    pub fn diagnostics(&self) -> String {
        self.transport.diagnostics()
    }

    async fn send_once(&self, method: &str, params: Value, limit: Duration) -> Result<Value, McpError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        match tokio::time::timeout(limit, self.transport.send_request(id, method, params)).await {
            Ok(r) => r,
            Err(_) => {
                // tell the server we stopped waiting
                let note = self.transport.send_notification(
                    "notifications/cancelled",
                    json!({"requestId": id, "reason": "timed out"}),
                );
                let _ = tokio::time::timeout(Duration::from_secs(2), note).await;
                Err(McpError::Timeout(limit))
            }
        }
    }

    /// A request; when an HTTP session has expired it starts a new session once and repeats the request.
    async fn rpc(&self, method: &str, params: Value, limit: Duration) -> Result<Value, McpError> {
        match self.send_once(method, params.clone(), limit).await {
            Err(McpError::SessionExpired) => {
                self.initialize().await?;
                self.send_once(method, params, limit).await
            }
            other => other,
        }
    }

    async fn initialize(&self) -> Result<(), McpError> {
        let result = self
            .send_once(
                "initialize",
                json!({
                    "protocolVersion": SUPPORTED_PROTOCOLS[0],
                    "capabilities": {},
                    "clientInfo": {"name": "Dara", "version": env!("CARGO_PKG_VERSION")},
                }),
                SETUP_TIMEOUT,
            )
            .await?;
        let protocol = result
            .get("protocolVersion")
            .and_then(Value::as_str)
            .ok_or_else(|| McpError::Protocol("initialize answered without a protocolVersion".into()))?
            .to_string();
        if !SUPPORTED_PROTOCOLS.contains(&protocol.as_str()) {
            return Err(McpError::Protocol(format!(
                "the server speaks protocol {protocol}, which this app does not support"
            )));
        }
        let server = result.get("serverInfo");
        let text = |k: &str| {
            server
                .and_then(|s| s.get(k))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        *self.info.lock().expect("info") = ServerInfo { name: text("name"), version: text("version"), protocol: protocol.clone() };
        self.transport.set_protocol_version(&protocol);
        self.transport
            .send_notification("notifications/initialized", json!({}))
            .await
    }

    pub async fn ping(&self) -> Result<(), McpError> {
        self.rpc("ping", json!({}), SETUP_TIMEOUT).await.map(|_| ())
    }

    /// All tools, following `nextCursor` across pages.
    pub async fn list_tools(&self) -> Result<Vec<RemoteTool>, McpError> {
        self.shared.tools_stale.store(false, Ordering::SeqCst);
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let params = match &cursor {
                Some(c) => json!({"cursor": c}),
                None => json!({}),
            };
            let result = self.rpc("tools/list", params, SETUP_TIMEOUT).await?;
            let page = result
                .get("tools")
                .and_then(Value::as_array)
                .ok_or_else(|| McpError::Protocol("tools/list answered without a tools array".into()))?;
            for t in page {
                tools.push(parse_tool(t)?);
                if tools.len() > MAX_TOOLS {
                    return Err(McpError::Protocol(format!("the server lists more than {MAX_TOOLS} tools")));
                }
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(c) if !c.is_empty() => cursor = Some(c.to_string()),
                _ => return Ok(tools),
            }
        }
        Err(McpError::Protocol("the tool list has too many pages".into()))
    }

    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
        timeout: Option<Duration>,
    ) -> Result<ToolResult, McpError> {
        let limit = timeout.unwrap_or(DEFAULT_CALL_TIMEOUT).min(MAX_CALL_TIMEOUT);
        let arguments = if arguments.is_null() { json!({}) } else { arguments };
        let result = self
            .rpc("tools/call", json!({"name": name, "arguments": arguments}), limit)
            .await?;
        let mut content: Vec<Value> = result
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let truncated = cap_text(&mut content, MAX_RESULT_TEXT_BYTES);
        Ok(ToolResult {
            content,
            structured_content: result.get("structuredContent").filter(|v| !v.is_null()).cloned(),
            is_error: result.get("isError").and_then(Value::as_bool).unwrap_or(false),
            truncated,
        })
    }

    pub async fn close(&self) {
        self.transport.close().await;
    }
}

fn parse_tool(t: &Value) -> Result<RemoteTool, McpError> {
    let name = t
        .get("name")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
        .ok_or_else(|| McpError::Protocol("a tool has no name".into()))?
        .to_string();
    let annotations = t.get("annotations").filter(|a| a.is_object()).cloned();
    let title = t
        .get("title")
        .and_then(Value::as_str)
        .or_else(|| annotations.as_ref().and_then(|a| a.get("title")).and_then(Value::as_str))
        .map(str::to_string);
    Ok(RemoteTool {
        name,
        title,
        description: t.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
        input_schema: t
            .get("inputSchema")
            .filter(|s| s.is_object())
            .cloned()
            .unwrap_or_else(|| Value::Object(Map::from_iter([("type".to_string(), json!("object"))]))),
        read_only_hint: annotations.as_ref().and_then(|a| a.get("readOnlyHint")).and_then(Value::as_bool),
        annotations,
    })
}

/// Cuts the text blocks so that together they hold at most `limit` bytes. Returns true when something was cut.
fn cap_text(content: &mut [Value], limit: usize) -> bool {
    let mut used = 0usize;
    let mut cut = false;
    for block in content.iter_mut() {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        let Some(text) = block.get("text").and_then(Value::as_str) else { continue };
        let room = limit.saturating_sub(used);
        if text.len() <= room {
            used += text.len();
            continue;
        }
        let mut end = room;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        let shortened = format!("{}\n[cut: the result was longer than 1 MB]", &text[..end]);
        block["text"] = Value::String(shortened);
        used = limit;
        cut = true;
    }
    cut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_ignores_key_order() {
        let a = json!({"b": 1, "a": {"y": [1, 2], "x": null}});
        let b = json!({"a": {"x": null, "y": [1, 2]}, "b": 1});
        assert_eq!(canonical(&a), canonical(&b));
    }

    #[test]
    fn a_tool_changes_its_fingerprint_when_its_description_changes() {
        let t = |d: &str| RemoteTool {
            name: "echo".into(),
            title: None,
            description: d.into(),
            input_schema: json!({"type": "object"}),
            annotations: None,
            read_only_hint: None,
        };
        assert_eq!(t("a").fingerprint(), t("a").fingerprint());
        assert_ne!(t("a").fingerprint(), t("b").fingerprint());
    }

    #[test]
    fn long_text_is_cut_on_a_character_boundary() {
        let mut c = vec![json!({"type": "text", "text": "é".repeat(20)})];
        assert!(cap_text(&mut c, 11));
        let t = c[0]["text"].as_str().unwrap();
        assert!(t.starts_with("ééééé"));
        assert!(t.contains("[cut"));
        let mut small = vec![json!({"type": "text", "text": "ok"})];
        assert!(!cap_text(&mut small, 11));
    }
}
