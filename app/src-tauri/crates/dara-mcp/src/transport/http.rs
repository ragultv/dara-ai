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

fn resource_metadata(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(WWW_AUTHENTICATE)?.to_str().ok()?;
    let start = value.find("resource_metadata=")? + "resource_metadata=".len();
    let rest = &value[start..];
    if let Some(quoted) = rest.strip_prefix('"') {
        quoted.split('"').next().map(str::to_string)
    } else {
        rest.split([',', ' ']).next().map(str::to_string)
    }
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

    /// Turns a bad status into the right error. `Ok` for 2xx.
    async fn check(&self, resp: Response) -> Result<Response, McpError> {
        let status = resp.status();
        if status.is_success() {
            return Ok(resp);
        }
        if status == StatusCode::UNAUTHORIZED {
            return Err(McpError::AuthRequired { resource_metadata: resource_metadata(resp.headers()) });
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
        let body = resp.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(200).collect();
        Err(McpError::Transport(format!("HTTP {}: {}", status.as_u16(), snippet.trim())))
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
        let resp = self.post(&rpc::request(id, method, params)).await?;
        let resp = self.check(resp).await?;
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
        let resp = self.post(&rpc::notification(method, params)).await?;
        self.check(resp).await.map(|_| ())
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
}
