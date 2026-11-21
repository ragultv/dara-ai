//! A small Streamable-HTTP MCP server for tests (JSON or SSE answers, sessions, a 401 mode, a session-expired mode, a redirect mode).
#![allow(dead_code)]

pub mod auth;

use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    /// lower-case names
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Recorded {
    pub fn rpc_method(&self) -> Option<String> {
        serde_json::from_str::<Value>(&self.body)
            .ok()
            .and_then(|v| v["method"].as_str().map(str::to_string))
    }
}

#[derive(Default)]
pub struct MockState {
    pub requests: Mutex<Vec<Recorded>>,
    /// answer with `text/event-stream` (a notification, a comment, then the response)
    pub sse: AtomicBool,
    pub unauthorized: AtomicBool,
    pub redirect: AtomicBool,
    /// the next request that carries a session id gets a 404 (and that session is forgotten)
    pub expire_next: AtomicBool,
    pub sessions_started: AtomicUsize,
    pub protocol: Mutex<Option<String>>,
}

impl MockState {
    pub fn recorded(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
    pub fn count(&self, rpc_method: &str) -> usize {
        self.recorded().iter().filter(|r| r.rpc_method().as_deref() == Some(rpc_method)).count()
    }
}

pub struct MockHttp {
    pub addr: SocketAddr,
    pub state: Arc<MockState>,
}

impl MockHttp {
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.addr.port())
    }
}

fn find(buf: &[u8], pat: &[u8]) -> Option<usize> {
    buf.windows(pat.len()).position(|w| w == pat)
}

async fn read_request(sock: &mut tokio::net::TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(p) = find(&buf, b"\r\n\r\n") {
            break p;
        }
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let first = lines.next()?;
    let mut parts = first.split(' ');
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut headers = HashMap::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len: usize = headers.get("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut body = buf[head_end + 4..].to_vec();
    while body.len() < len {
        let n = sock.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n]);
    }
    Some(Recorded { method, path, headers, body: String::from_utf8_lossy(&body).to_string() })
}

async fn respond(sock: &mut tokio::net::TcpStream, status: &str, headers: &[(&str, String)], body: &str, sse: bool) {
    let mut out = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    if !sse {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    out.push_str(body);
    let _ = sock.write_all(out.as_bytes()).await;
    let _ = sock.shutdown().await;
}

fn tools() -> Value {
    json!([{
        "name": "echo",
        "description": "Echo the text",
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
        "annotations": {"readOnlyHint": true}
    }])
}

async fn handle(mut sock: tokio::net::TcpStream, state: Arc<MockState>, current: Arc<Mutex<Option<String>>>) {
    let Some(req) = read_request(&mut sock).await else { return };
    state.requests.lock().unwrap().push(req.clone());

    if state.redirect.load(Ordering::SeqCst) {
        respond(&mut sock, "307 Temporary Redirect", &[("Location", "/elsewhere".into())], "", false).await;
        return;
    }
    if state.unauthorized.load(Ordering::SeqCst) {
        let challenge = format!(
            "Bearer realm=\"mcp\", resource_metadata=\"http://127.0.0.1:{}/.well-known/oauth-protected-resource\"",
            sock.local_addr().map(|a| a.port()).unwrap_or(0)
        );
        respond(&mut sock, "401 Unauthorized", &[("WWW-Authenticate", challenge)], "", false).await;
        return;
    }
    if req.method == "DELETE" {
        respond(&mut sock, "200 OK", &[], "", false).await;
        return;
    }
    let msg: Value = serde_json::from_str(&req.body).unwrap_or(Value::Null);
    let method = msg["method"].as_str().unwrap_or("").to_string();
    let id = msg.get("id").cloned();

    // a notification, or the client's answer to something we asked: accepted, no body
    if id.is_none() || method.is_empty() {
        respond(&mut sock, "202 Accepted", &[], "", false).await;
        return;
    }

    let mut extra: Vec<(&str, String)> = Vec::new();
    if method == "initialize" {
        let n = state.sessions_started.fetch_add(1, Ordering::SeqCst) + 1;
        let sid = format!("sess-{n}");
        *current.lock().unwrap() = Some(sid.clone());
        extra.push(("Mcp-Session-Id", sid));
    } else {
        let given = req.headers.get("mcp-session-id").cloned();
        if state.expire_next.swap(false, Ordering::SeqCst) && given.is_some() {
            *current.lock().unwrap() = None;
            respond(&mut sock, "404 Not Found", &[], "", false).await;
            return;
        }
        let ok = given.is_some() && given == *current.lock().unwrap();
        if !ok {
            respond(&mut sock, if given.is_some() { "404 Not Found" } else { "400 Bad Request" }, &[], "", false).await;
            return;
        }
    }

    let result = match method.as_str() {
        "initialize" => {
            let asked = msg["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18").to_string();
            let version = state.protocol.lock().unwrap().clone().unwrap_or(asked);
            json!({"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "mock-http", "version": "1.0"}})
        }
        "tools/list" => json!({"tools": tools()}),
        "tools/call" => json!({"content": [{"type": "text", "text": msg["params"]["arguments"]["text"].as_str().unwrap_or("")}]}),
        "ping" => json!({}),
        _ => {
            let err = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}});
            extra.push(("Content-Type", "application/json".into()));
            respond(&mut sock, "200 OK", &extra, &err.to_string(), false).await;
            return;
        }
    };
    let response = json!({"jsonrpc": "2.0", "id": id, "result": result});
    if state.sse.load(Ordering::SeqCst) {
        extra.push(("Content-Type", "text/event-stream".into()));
        let note = json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"});
        let body = format!(": keep-alive\n\ndata: {note}\n\nid: 7\ndata: {response}\n\n");
        respond(&mut sock, "200 OK", &extra, &body, true).await;
    } else {
        extra.push(("Content-Type", "application/json".into()));
        respond(&mut sock, "200 OK", &extra, &response.to_string(), false).await;
    }
}

pub async fn start_http() -> MockHttp {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(MockState::default());
    let current = Arc::new(Mutex::new(None));
    let s = Arc::clone(&state);
    tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = listener.accept().await else { break };
            tokio::spawn(handle(sock, Arc::clone(&s), Arc::clone(&current)));
        }
    });
    MockHttp { addr, state }
}
