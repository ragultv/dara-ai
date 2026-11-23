//! API keys for http connectors: a server that wants a key you paste, instead of a sign-in in the browser.
//!
//! The key is kept in the secret store (never in the database or a log); the database keeps only the NAME of the header it goes in.
//! The key is read from the store on every request, so changing it takes effect without reconnecting.

use crate::error::McpError;
use crate::secrets::SecretStore;
use crate::transport::TokenProvider;
use async_trait::async_trait;
use std::sync::Arc;

/// Longest key accepted (a Windows credential holds about 2.5 KB).
pub const MAX_KEY_CHARS: usize = 1500;

/// Headers a key must not be sent in: they belong to the protocol or to the connection.
const RESERVED_HEADERS: [&str; 12] = [
    "host",
    "content-type",
    "content-length",
    "accept",
    "mcp-session-id",
    "mcp-protocol-version",
    "cookie",
    "transfer-encoding",
    "connection",
    "user-agent",
    "origin",
    "upgrade",
];

/// A header name from the user: `None` for the default (`Authorization: Bearer <key>`), `Some(name)` otherwise. An empty name or
/// `Authorization` mean the default.
pub fn check_header_name(raw: Option<&str>) -> Result<Option<String>, McpError> {
    let Some(name) = raw.map(str::trim).filter(|n| !n.is_empty()) else { return Ok(None) };
    if name.eq_ignore_ascii_case("authorization") {
        return Ok(None);
    }
    let ok = name.len() <= 64 && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_');
    if !ok {
        return Err(McpError::Invalid(
            "the header name may only have letters, digits, - and _ (up to 64 characters)".into(),
        ));
    }
    if RESERVED_HEADERS.iter().any(|r| name.eq_ignore_ascii_case(r)) {
        return Err(McpError::Invalid(format!("{name} is used by the connection itself; choose another header")));
    }
    Ok(Some(name.to_string()))
}

/// The key as pasted: trimmed, one line, printable ASCII (it becomes an HTTP header value).
pub fn check_key(raw: &str) -> Result<String, McpError> {
    let key = raw.trim();
    // a pasted "Bearer xyz" is a common slip: the connector adds "Bearer " itself
    let key = if key.len() > 7 && key[..7].eq_ignore_ascii_case("bearer ") { key[7..].trim() } else { key };
    if key.is_empty() {
        return Err(McpError::Invalid("the API key is empty".into()));
    }
    if key.chars().count() > MAX_KEY_CHARS {
        return Err(McpError::Invalid(format!("the API key is longer than {MAX_KEY_CHARS} characters")));
    }
    if !key.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(McpError::Invalid(
            "the API key has a space or a character that cannot go in a header; paste just the key".into(),
        ));
    }
    Ok(key.to_string())
}

/// Sends the stored key with every request.
pub struct ApiKeyProvider {
    secrets: Arc<dyn SecretStore>,
    key: String,
    header: Option<String>,
}

impl ApiKeyProvider {
    pub fn new(secrets: Arc<dyn SecretStore>, secret_key: String, header: Option<String>) -> Self {
        Self { secrets, key: secret_key, header }
    }
}

#[async_trait]
impl TokenProvider for ApiKeyProvider {
    async fn bearer(&self) -> Option<String> {
        self.secrets.get(&self.key).ok().flatten()
    }

    fn header_name(&self) -> Option<String> {
        self.header.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_names() {
        assert_eq!(check_header_name(None).unwrap(), None);
        assert_eq!(check_header_name(Some("  ")).unwrap(), None);
        assert_eq!(check_header_name(Some("Authorization")).unwrap(), None);
        assert_eq!(check_header_name(Some(" X-API-Key ")).unwrap().as_deref(), Some("X-API-Key"));
        assert!(check_header_name(Some("Host")).is_err());
        assert!(check_header_name(Some("Mcp-Session-Id")).is_err());
        assert!(check_header_name(Some("bad name")).is_err());
        assert!(check_header_name(Some("bad:name")).is_err());
        assert!(check_header_name(Some(&"a".repeat(65))).is_err());
    }

    #[test]
    fn keys() {
        assert_eq!(check_key("  fc-123abc ").unwrap(), "fc-123abc");
        assert_eq!(check_key("Bearer fc-123abc").unwrap(), "fc-123abc");
        assert_eq!(check_key("bearer  fc-123abc").unwrap(), "fc-123abc");
        assert!(check_key("").is_err());
        assert!(check_key("   ").is_err());
        assert!(check_key("two words").is_err());
        assert!(check_key("line\nbreak").is_err());
        assert!(check_key("caf\u{e9}").is_err());
        assert!(check_key(&"k".repeat(MAX_KEY_CHARS + 1)).is_err());
    }
}
