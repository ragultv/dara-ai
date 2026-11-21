//! How messages get to a server and back: stdio (a child process) or Streamable HTTP.

mod http;
mod stdio;

pub use http::HttpTransport;
pub use stdio::{StdioConfig, StdioTransport};

use crate::error::McpError;
use crate::rpc::{self, Incoming};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use url::Url;

/// A bearer token for an HTTP server (the OAuth code is the implementation: `oauth::OAuthTokens`).
#[async_trait]
pub trait TokenProvider: Send + Sync {
    async fn bearer(&self) -> Option<String>;
    /// The server refused the token (401). Gets a new one if that is possible; `true` when the next `bearer()` differs, so one retry
    /// is worth making. Concurrent callers share one refresh.
    async fn refresh(&self) -> bool {
        false
    }
}

/// What the server told us on the side (shared by the transport and the client).
#[derive(Default)]
pub struct Shared {
    /// `notifications/tools/list_changed` arrived: the cached tool list is out of date.
    pub tools_stale: AtomicBool,
}

impl Shared {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait]
pub trait Transport: Send + Sync {
    /// Sends a request and waits for its answer (the `result`). The caller applies the timeout by dropping this future.
    async fn send_request(&self, id: u64, method: &str, params: Value) -> Result<Value, McpError>;
    async fn send_notification(&self, method: &str, params: Value) -> Result<(), McpError>;
    /// The protocol version agreed during `initialize` (HTTP sends it as a header).
    fn set_protocol_version(&self, version: &str);
    /// Ends the session and stops the child process, if any.
    async fn close(&self);
    /// What the server wrote to stderr (stdio) or an empty string.
    fn diagnostics(&self) -> String;
}

/// What to do about one message from the server. Requests we cannot serve get "method not found".
pub(crate) enum Reaction {
    /// An answer to one of our requests.
    Response { id: u64, result: Result<Value, McpError> },
    /// Send this reply back to the server.
    Reply(Value),
    Nothing,
}

pub(crate) fn react(value: &Value, shared: &Shared) -> Reaction {
    match rpc::parse_incoming(value) {
        Ok(Incoming::Response { id, result }) => match rpc::id_as_u64(&id) {
            Some(id) => Reaction::Response { id, result },
            None => Reaction::Nothing,
        },
        Ok(Incoming::Request { id, method }) => Reaction::Reply(rpc::method_not_found(&id, &method)),
        Ok(Incoming::Notification { method }) => {
            if method == "notifications/tools/list_changed" {
                shared.tools_stale.store(true, Ordering::SeqCst);
            }
            Reaction::Nothing
        }
        Err(_) => Reaction::Nothing,
    }
}

pub(crate) fn is_loopback_host(url: &Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(d)) => d.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

/// The rule for an HTTP connector: https, or http only to this computer. No credentials in the address.
pub fn check_http_url(raw: &str) -> Result<Url, McpError> {
    let url = Url::parse(raw.trim()).map_err(|e| McpError::Invalid(format!("not a valid address: {e}")))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err(McpError::Invalid("an address must not contain a user name or password".into()));
    }
    match url.scheme() {
        "https" => Ok(url),
        "http" if is_loopback_host(&url) => Ok(url),
        "http" => Err(McpError::Invalid(
            "plain http is only allowed to this computer (localhost); use https".into(),
        )),
        other => Err(McpError::Invalid(format!("unsupported address scheme: {other}"))),
    }
}

/// Largest response body the HTTP transport reads.
pub const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn https_and_loopback_only() {
        assert!(check_http_url("https://mcp.example.com/mcp").is_ok());
        assert!(check_http_url("http://127.0.0.1:8080/mcp").is_ok());
        assert!(check_http_url("http://localhost/mcp").is_ok());
        assert!(check_http_url("http://[::1]:9/mcp").is_ok());
        assert!(check_http_url("http://example.com/mcp").is_err());
        assert!(check_http_url("http://192.168.1.5/mcp").is_err());
        assert!(check_http_url("ftp://localhost/").is_err());
        assert!(check_http_url("https://user:pw@example.com/").is_err());
        assert!(check_http_url("not a url").is_err());
    }
}
