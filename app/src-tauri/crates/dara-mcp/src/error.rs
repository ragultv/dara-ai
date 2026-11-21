use std::time::Duration;

/// Everything that can go wrong. Cloneable so one failure can be handed to every request that was waiting.
#[derive(Debug, Clone, thiserror::Error)]
pub enum McpError {
    /// The connection itself failed (could not start, could not reach, bad status).
    #[error("{0}")]
    Transport(String),
    #[error("timed out after {0:?}")]
    Timeout(Duration),
    /// The server went away (a child process exited, a stream ended) before answering.
    #[error("the server closed the connection: {0}")]
    Closed(String),
    /// The server answered with a JSON-RPC error.
    #[error("server error {code}: {message}")]
    Rpc { code: i64, message: String },
    /// HTTP 401: the server wants a token. `resource_metadata` and `scope` come from `WWW-Authenticate`, when given.
    #[error("authorization required")]
    AuthRequired { resource_metadata: Option<String>, scope: Option<String> },
    /// HTTP 403 with `error="insufficient_scope"`: the token is fine but does not allow this. Never fixed silently: the user decides
    /// whether to sign in again with `scope`.
    #[error("this server needs more access")]
    InsufficientScope { scope: Option<String> },
    /// The HTTP session id is no longer known to the server (404 with a session). Handled inside the client.
    #[error("the session expired")]
    SessionExpired,
    /// The server said something that is not valid MCP.
    #[error("{0}")]
    Protocol(String),
    /// Bad input from the caller (validation).
    #[error("{0}")]
    Invalid(String),
    /// The user set this tool to "deny".
    #[error("this tool is denied")]
    Denied,
    /// The tool is set to "ask" and this call was not approved.
    #[error("this tool needs approval")]
    NeedsApproval,
    #[error("not found: {0}")]
    NotFound(String),
    #[error("storage: {0}")]
    Storage(String),
}

impl From<rusqlite::Error> for McpError {
    fn from(e: rusqlite::Error) -> Self {
        McpError::Storage(e.to_string())
    }
}
