mod common;

use async_trait::async_trait;
use common::start_http;
use dara_mcp::{HttpTransport, McpClient, McpError, TokenProvider};
use dara_mcp::transport::Shared;
use serde_json::json;
use std::sync::atomic::Ordering;
use std::sync::Arc;

async fn connect(url: &str, token: Option<Arc<dyn TokenProvider>>) -> Result<McpClient, McpError> {
    let shared = Shared::new();
    let t = HttpTransport::new(url, token, Arc::clone(&shared))?;
    McpClient::connect(Arc::new(t), shared).await
}

#[tokio::test]
async fn json_answers_keep_the_session_and_end_it_on_close() {
    let mock = start_http().await;
    let client = connect(&mock.url(), None).await.unwrap();
    assert_eq!(client.server_info().name, "mock-http");
    assert_eq!(client.list_tools().await.unwrap().len(), 1);
    let r = client.call_tool("echo", json!({"text": "hi"}), None).await.unwrap();
    assert_eq!(r.content[0]["text"], "hi");
    client.close().await;

    let reqs = mock.state.recorded();
    let init = &reqs[0];
    assert_eq!(init.rpc_method().as_deref(), Some("initialize"));
    assert!(!init.headers.contains_key("mcp-session-id"), "no session before initialize");
    assert!(init.headers["accept"].contains("text/event-stream") && init.headers["accept"].contains("application/json"));
    for r in reqs.iter().skip(1).filter(|r| r.method == "POST") {
        assert_eq!(r.headers.get("mcp-session-id").map(String::as_str), Some("sess-1"), "{:?}", r.rpc_method());
        assert_eq!(r.headers.get("mcp-protocol-version").map(String::as_str), Some("2025-06-18"));
    }
    let del = reqs.last().unwrap();
    assert_eq!(del.method, "DELETE");
    assert_eq!(del.headers.get("mcp-session-id").map(String::as_str), Some("sess-1"));
}

#[tokio::test]
async fn an_older_protocol_version_is_followed() {
    let mock = start_http().await;
    *mock.state.protocol.lock().unwrap() = Some("2025-03-26".into());
    let client = connect(&mock.url(), None).await.unwrap();
    assert_eq!(client.server_info().protocol, "2025-03-26");
    client.list_tools().await.unwrap();
    let last = mock.state.recorded().into_iter().last().unwrap();
    assert_eq!(last.headers.get("mcp-protocol-version").map(String::as_str), Some("2025-03-26"));

    *mock.state.protocol.lock().unwrap() = Some("1999-01-01".into());
    assert!(matches!(connect(&mock.url(), None).await, Err(McpError::Protocol(_))));
}

#[tokio::test]
async fn event_streams_with_comments_and_several_events() {
    let mock = start_http().await;
    mock.state.sse.store(true, Ordering::SeqCst);
    let client = connect(&mock.url(), None).await.unwrap();
    let r = client.call_tool("echo", json!({"text": "streamed"}), None).await.unwrap();
    assert_eq!(r.content[0]["text"], "streamed");
    // the notification event that came before the answer marked the tool list stale
    assert!(client.tools_stale());
    assert_eq!(client.list_tools().await.unwrap()[0].name, "echo");
}

#[tokio::test]
async fn a_401_asks_for_sign_in_and_names_the_metadata() {
    let mock = start_http().await;
    mock.state.unauthorized.store(true, Ordering::SeqCst);
    match connect(&mock.url(), None).await {
        Err(McpError::AuthRequired { resource_metadata }) => {
            let url = resource_metadata.expect("the challenge names the metadata");
            assert!(url.ends_with("/.well-known/oauth-protected-resource"), "{url}");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

#[tokio::test]
async fn an_expired_session_is_started_again_once() {
    let mock = start_http().await;
    let client = connect(&mock.url(), None).await.unwrap();
    mock.state.expire_next.store(true, Ordering::SeqCst);
    let r = client.call_tool("echo", json!({"text": "after"}), None).await.unwrap();
    assert_eq!(r.content[0]["text"], "after");
    assert_eq!(mock.state.count("initialize"), 2);
    assert_eq!(mock.state.sessions_started.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn plain_http_is_refused_except_to_this_computer() {
    assert!(HttpTransport::new("http://example.com/mcp", None, Shared::new()).is_err());
    assert!(HttpTransport::new("http://10.0.0.5/mcp", None, Shared::new()).is_err());
    assert!(HttpTransport::new("https://example.com/mcp", None, Shared::new()).is_ok());
    assert!(HttpTransport::new("http://127.0.0.1:9/mcp", None, Shared::new()).is_ok());
}

#[tokio::test]
async fn redirects_are_not_followed() {
    let mock = start_http().await;
    mock.state.redirect.store(true, Ordering::SeqCst);
    let err = connect(&mock.url(), None).await.err().expect("must fail");
    assert!(err.to_string().contains("redirect"), "{err}");
    let reqs = mock.state.recorded();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].path, "/mcp");
}

struct Fixed;
#[async_trait]
impl TokenProvider for Fixed {
    async fn bearer(&self) -> Option<String> {
        Some("tok-123".into())
    }
}

#[tokio::test]
async fn a_token_provider_supplies_the_bearer_header() {
    let mock = start_http().await;
    let client = connect(&mock.url(), Some(Arc::new(Fixed))).await.unwrap();
    client.list_tools().await.unwrap();
    assert!(mock
        .state
        .recorded()
        .iter()
        .filter(|r| r.method == "POST")
        .all(|r| r.headers.get("authorization").map(String::as_str) == Some("Bearer tok-123")));
}
