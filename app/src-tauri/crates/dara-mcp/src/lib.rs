//! MCP client, connector manager, permissions and audit for Dara.
//!
//! Headless on purpose (no Tauri): the desktop shell, and later a workflow engine, use it as a library.
//!
//! - [`client`]: the Model Context Protocol client (initialize, tools/list, tools/call) over a [`transport::Transport`].
//! - [`transport`]: stdio (a child process) and Streamable HTTP.
//! - [`store`]: SQLite for connectors, tool permissions and the audit log. Secrets are never stored here.
//! - [`secrets`]: where secret values live (Windows Credential Manager, or memory in tests).
//! - [`manager`]: connectors, connections, permission gating and audit, the API the shell calls.

pub mod client;
pub mod error;
pub mod manager;
pub mod rpc;
pub mod secrets;
pub mod store;
pub mod transport;

pub use client::{McpClient, RemoteTool, ServerInfo, ToolResult, SUPPORTED_PROTOCOLS};
pub use error::McpError;
pub use manager::{
    AuditEntry, CallOutcome, ConnState, ConnStatus, ConnectorConfig, ConnectorPatch, ConnectorView, Kind, Manager,
    NewConnector, Outcome, ToolView,
};
pub use secrets::{KeyringSecrets, MemorySecrets, SecretStore};
pub use transport::{
    check_http_url, HttpTransport, StdioConfig, StdioTransport, TokenProvider, Transport,
};
