//! Streamable HTTP: every message is a POST to one endpoint; the answer is a JSON body or a stream of server-sent events.

use super::{check_http_url, react, Reaction, Shared, TokenProvider, Transport, MAX_BODY_BYTES};
use crate::error::McpError;
use crate::rpc;
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, ACCEPT, AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use reqwest::{redirect, Client, Response, StatusCode};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use url::Url;

pub struct HttpTransport {
    client: Client,
    url: Url,
    session: Mutex<Option<String>>,
    protocol: Mutex<Option<String>>,
    token: Option<Arc<dyn TokenProvider>>,
    shared: Arc<Shared>,
}

/// Splits a server-sent-event stream into the `data` of each event. Comments (`:`) and other fields are ignored.
#[derive(Default)]
pub(crate) struct SseParser {
    buf: Vec<u8>,
    data: Vec<String>,
}

impl SseParser {
    pub(crate) fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let raw: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&raw);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                if !self.data.is_empty() {
                    events.push(self.data.join("\n"));
                    self.data.clear();
                }
            } else if line.starts_with(':') {
                // a comment (keep-alive)
            } else if let Some(rest) = line.strip_prefix("data:") {
                self.data.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
            }
        }
        events
    }
}

/// One parameter (`name="value"` or `name=value`) of a `WWW-Authenticate: Bearer ...` challenge.
pub(crate) fn auth_param(value: &str, name: &str) -> Option<String> {
    let bytes = value.as_bytes();
    let mut from = 0;
    while let Some(found) = value[from..].find(name) {
        let at = from + found;
        from = at + name.len();
        let before_ok = at == 0 || matches!(bytes[at - 1], b' ' | b',' | b'\t');
        if !before_ok || bytes.get(from) != Some(&b'=') {
            continue;
        }
        let rest = &value[from + 1..];
        return if let Some(quoted) = rest.strip_prefix('"') {
            quoted.split('"').next().map(str::to_string)
        } else {
            rest.split([',', ' ']).next().map(str::to_string)
        };
    }
    None
}

fn resource_metadata(headers: &HeaderMap) -> Option<String> {
    auth_param(headers.get(WWW_AUTHENTICATE)?.to_str().ok()?, "resource_metadata")
}

fn challenge_scope(headers: &HeaderMap) -> Option<String> {
    auth_param(headers.get(WWW_AUTHENTICATE)?.to_str().ok()?, "scope")
}

impl HttpTransport {
    pub fn new(
        url: &str,
        token: Option<Arc<dyn TokenProvider>>,
        shared: Arc<Shared>,
    ) -> Result<Self, McpError> {
        let url = check_http_url(url)?;
        let client = Client::builder()
            .redirect(redirect::Policy::none()) // a redirect could send the token somewhere else
            .connect_timeout(Duration::from_secs(10))
            .user_agent(concat!("Dara/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| McpError::Transport(e.to_string()))?;
        Ok(Self {
            client,
            url,
            session: Mutex::new(None),
            protocol: Mutex::new(None),
            token,
            shared,
        })
    }

    fn session(&self) -> Option<String> {
        self.session.lock().ok().and_then(|s| s.clone())
    }

    async fn post(&self, body: &Value) -> Result<Response, McpError> {
        let mut rb = self
            .client
            .post(self.url.clone())
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .body(body.to_string());
        if let Some(p) = self.protocol.lock().ok().and_then(|p| p.clone()) {
            rb = rb.header("MCP-Protocol-Version", p);
        }
        if let Some(s) = self.session() {
            rb = rb.header("Mcp-Session-Id", s);
        }
        if let Some(provider) = &self.token {
            if let Some(token) = provider.bearer().await {
                rb = rb.header(AUTHORIZATION, format!("Bearer {token}"));
            }
        }
        rb.send()
            .await
            .map_err(|e| McpError::Transport(format!("could not reach the server: {e}")))
    }

    /// Posts and checks the status. A 401 with a token provider gets ONE retry after the provider refreshes its token; a second 401 is
    /// reported as it is (the connector then needs signing in again).
    async fn post_checked(&self, body: &Value) -> Result<Response, McpError> {
        let resp = self.post(body).await?;
        match self.check(resp).await {
            Err(first @ McpError::AuthRequired { .. }) => {
                let Some(provider) = &self.token else { return Err(first) };
                if !provider.refresh().await {
                    return Err(first);
                }
                let resp = self.post(body).await?;
                self.check(resp).await
            }
            other => other,
        }
    }

    /// Turns a bad status into the right error. `Ok` for 2xx.
    async fn check(&self, resp: Response) -> Result<Response, McpError> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        if status == StatusCode::UNAUTHORIZED {
            return Err(McpError::AuthRequired {
                resource_metadata: resource_metadata(resp.headers()),
                scope: challenge_scope(resp.headers()),
            });
        }
        if status == StatusCode::FORBIDDEN {
            let www = resp.headers().get(WWW_AUTHENTICATE).and_then(|v| v.to_str().ok()).map(str::to_string);
            if www.as_deref().and_then(|w| auth_param(w, "error")).as_deref() == Some("insufficient_scope") {
                return Err(McpError::InsufficientScope { scope: www.as_deref().and_then(|w| auth_param(w, "scope")) });
            }
        }
        if status == StatusCode::NOT_FOUND && self.session().is_some() {
            if let Ok(mut s) = self.session.lock() {
                *s = None;
            }
            return Err(McpError::SessionExpired);
        }
        if status.is_redirection() {
            let to = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("somewhere else")
                .to_string();
            return Err(McpError::Transport(format!(
                "the server redirected to {to} (HTTP {}); redirects are not followed",
                status.as_u16()
            )));
        }
        let www = resp.headers().get(WWW_AUTHENTICATE).and_then(|v| v.to_str().ok()).map(str::to_string);
        let body = resp.text().await.unwrap_or_default();
        // Google answers a refused request with the normal result in the body (a tool list under a 403), which says nothing about
        // the refusal; only an error object or non-JSON text is worth showing. The challenge header goes first: it holds no secret.
        let says_why = match serde_json::from_str::<Value>(&body) {
            Ok(v) => v.get("error").is_some().then(|| v["error"].to_string()),
            Err(_) => Some(body.chars().take(1500).collect::<String>()),
        };
        let mut msg = format!("HTTP {}", status.as_u16());
        if let Some(w) = www {
            msg.push_str(&format!(" [WWW-Authenticate: {w}]"));
        }
        if let Some(t) = says_why.filter(|t| !t.trim().is_empty()) {
            msg.push_str(&format!(": {}", t.trim()));
        }
        Err(McpError::Transport(msg))
    }

    async fn read_body(resp: Response) -> Result<Vec<u8>, McpError> {
        let mut stream = resp.bytes_stream();
        let mut out = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| McpError::Transport(format!("could not read the answer: {e}")))?;
            out.extend_from_slice(&chunk);
            if out.len() > MAX_BODY_BYTES {
                return Err(McpError::Protocol("the answer is larger than 4 MB".into()));
            }
        }
        Ok(out)
    }

    /// Handles one parsed JSON message of an answer: returns our response when this is it.
    async fn handle(&self, value: &Value, id: u64) -> Option<Result<Value, McpError>> {
        let items: Vec<&Value> = match value {
            Value::Array(a) => a.iter().collect(),
            v => vec![v],
        };
        for item in items {
            match react(item, &self.shared) {
                Reaction::Response { id: rid, result } if rid == id => return Some(result),
                Reaction::Response { .. } | Reaction::Nothing => {}
                Reaction::Reply(reply) => {
                    let _ = self.post(&reply).await; // "method not found" for a request we cannot serve
                }
            }
        }
        None
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send_request(&self, id: u64, method: &str, params: Value) -> Result<Value, McpError> {
        let resp = self.post_checked(&rpc::request(id, method, params)).await?;
        if method == "initialize" {
            if let Some(sid) = resp.headers().get("mcp-session-id").and_then(|v| v.to_str().ok()) {
                if let Ok(mut s) = self.session.lock() {
                    *s = Some(sid.to_string());
                }
            }
        }
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_ascii_lowercase();

        if content_type.starts_with("text/event-stream") {
            let mut parser = SseParser::default();
            let mut total = 0usize;
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk =
                    chunk.map_err(|e| McpError::Closed(format!("the event stream broke: {e}")))?;
                total += chunk.len();
                if total > MAX_BODY_BYTES {
                    return Err(McpError::Protocol("the answer is larger than 4 MB".into()));
                }
                for data in parser.feed(&chunk) {
                    let Ok(value) = serde_json::from_str::<Value>(&data) else { continue };
                    if let Some(result) = self.handle(&value, id).await {
                        return result;
                    }
                }
            }
            Err(McpError::Closed("the event stream ended without an answer".into()))
        } else {
            let body = Self::read_body(resp).await?;
            if body.is_empty() {
                return Err(McpError::Protocol("the server sent no answer".into()));
            }
            let value: Value = serde_json::from_slice(&body)
                .map_err(|e| McpError::Protocol(format!("the answer is not JSON: {e}")))?;
            self.handle(&value, id)
                .await
                .unwrap_or_else(|| Err(McpError::Protocol("the answer does not match the request".into())))
        }
    }

    async fn send_notification(&self, method: &str, params: Value) -> Result<(), McpError> {
        self.post_checked(&rpc::notification(method, params)).await.map(|_| ())
    }

    fn set_protocol_version(&self, version: &str) {
        if let Ok(mut p) = self.protocol.lock() {
            *p = Some(version.to_string());
        }
    }

    async fn close(&self) {
        let Some(session) = self.session.lock().ok().and_then(|mut s| s.take()) else { return };
        let mut rb = self
            .client
            .delete(self.url.clone())
            .header("Mcp-Session-Id", session)
            .timeout(Duration::from_secs(3));
        if let Some(p) = self.protocol.lock().ok().and_then(|p| p.clone()) {
            rb = rb.header("MCP-Protocol-Version", p);
        }
        if let Some(provider) = &self.token {
            if let Some(token) = provider.bearer().await {
                rb = rb.header(AUTHORIZATION, format!("Bearer {token}"));
            }
        }
        let _ = rb.send().await; // the server may refuse (405): that is allowed
    }

    fn diagnostics(&self) -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_events_comments_and_split_chunks() {
        let mut p = SseParser::default();
        assert!(p.feed(b": keep-alive\n\nda").is_empty());
        let ev = p.feed(b"ta: {\"a\":1}\n\ndata: x\ndata: y\r\n\r\n");
        assert_eq!(ev, vec!["{\"a\":1}".to_string(), "x\ny".to_string()]);
    }

    #[test]
    fn reads_the_resource_metadata_url() {
        let mut h = HeaderMap::new();
        h.insert(
            WWW_AUTHENTICATE,
            "Bearer realm=\"mcp\", resource_metadata=\"https://x.io/.well-known/oauth-protected-resource\""
                .parse()
                .unwrap(),
        );
        assert_eq!(
            resource_metadata(&h).as_deref(),
            Some("https://x.io/.well-known/oauth-protected-resource")
        );
        assert_eq!(resource_metadata(&HeaderMap::new()), None);
    }

    #[test]
    fn reads_scope_and_ignores_look_alikes() {
        let v = "Bearer realm=\"mcp\", error=\"insufficient_scope\", scope=\"mail.read calendar.read\", resource_metadata=\"https://x.io/m\"";
        assert_eq!(auth_param(v, "scope").as_deref(), Some("mail.read calendar.read"));
        assert_eq!(auth_param(v, "resource_metadata").as_deref(), Some("https://x.io/m"));
        assert_eq!(auth_param("Bearer x_scope=\"no\"", "scope"), None);
        assert_eq!(auth_param("Bearer scope=read", "scope").as_deref(), Some("read"));
    }
}
