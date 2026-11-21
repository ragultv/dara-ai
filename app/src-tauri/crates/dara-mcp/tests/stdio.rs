use dara_mcp::transport::Shared;
use dara_mcp::{McpClient, McpError, StdioConfig, StdioTransport};
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MOCK: &str = env!("CARGO_BIN_EXE_mock_mcp");

fn config(args: &[&str], env: &[(&str, &str)], dir: &std::path::Path) -> StdioConfig {
    StdioConfig {
        command: MOCK.to_string(),
        args: args.iter().map(|s| s.to_string()).collect(),
        env: env.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        cwd: dir.to_path_buf(),
    }
}

async fn connect(args: &[&str], env: &[(&str, &str)], dir: &std::path::Path) -> Result<McpClient, McpError> {
    let shared = Shared::new();
    let t = StdioTransport::spawn(config(args, env, dir), Arc::clone(&shared))?;
    McpClient::connect(Arc::new(t), shared).await
}

#[tokio::test]
async fn handshake_pagination_and_results() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&[], &[], dir.path()).await.unwrap();
    let info = c.server_info();
    assert_eq!((info.name.as_str(), info.protocol.as_str()), ("mock-mcp", "2025-06-18"));
    let tools = c.list_tools().await.unwrap();
    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["echo", "slow", "fail", "big", "crash", "mutate", "probe", "env"], "two pages, in order");
    assert_eq!(tools[0].read_only_hint, Some(true));

    let ok = c.call_tool("echo", json!({"text": "hello"}), None).await.unwrap();
    assert!(!ok.is_error && !ok.truncated);
    assert_eq!(ok.content[0], json!({"type": "text", "text": "hello"}));

    let bad = c.call_tool("fail", json!({}), None).await.unwrap();
    assert!(bad.is_error);
    assert_eq!(bad.content[0]["text"], "boom");

    let unknown = c.call_tool("nope", json!({}), None).await.err().unwrap();
    assert!(matches!(unknown, McpError::Rpc { code: -32602, .. }), "{unknown:?}");
    c.ping().await.unwrap();
    c.close().await;
}

#[tokio::test]
async fn an_older_version_is_followed_and_an_unknown_one_refused() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&["--protocol", "2024-11-05"], &[], dir.path()).await.unwrap();
    assert_eq!(c.server_info().protocol, "2024-11-05");
    c.close().await;
    let err = connect(&["--protocol", "1999-01-01"], &[], dir.path()).await.err().unwrap();
    assert!(matches!(err, McpError::Protocol(_)), "{err:?}");
}

#[tokio::test]
async fn a_call_that_times_out_is_cancelled_on_the_server() {
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("mock.log");
    let c = connect(&[], &[("MOCK_LOG", log.to_str().unwrap())], dir.path()).await.unwrap();
    let started = Instant::now();
    let err = c
        .call_tool("slow", json!({"ms": 5000}), Some(Duration::from_millis(300)))
        .await
        .err()
        .unwrap();
    assert!(matches!(err, McpError::Timeout(_)), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(3));
    let mut seen = false;
    for _ in 0..40 {
        if std::fs::read_to_string(&log).unwrap_or_default().contains("cancelled:") {
            seen = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(seen, "the server got notifications/cancelled");
    // the connection still works afterwards
    assert!(c.call_tool("echo", json!({"text": "again"}), None).await.is_ok());
    c.close().await;
}

#[tokio::test]
async fn a_process_that_exits_mid_call_is_an_error_not_a_hang() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&[], &[], dir.path()).await.unwrap();
    let started = Instant::now();
    let err = c.call_tool("crash", json!({}), Some(Duration::from_secs(30))).await.err().unwrap();
    assert!(matches!(err, McpError::Closed(_)), "{err:?}");
    assert!(started.elapsed() < Duration::from_secs(10), "no waiting for the timeout");
    let again = c.call_tool("echo", json!({}), None).await.err().unwrap();
    assert!(matches!(again, McpError::Closed(_)), "{again:?}");
}

#[tokio::test]
async fn huge_text_is_cut_and_marked() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&[], &[], dir.path()).await.unwrap();
    let r = c.call_tool("big", json!({}), None).await.unwrap();
    assert!(r.truncated);
    let len = r.content[0]["text"].as_str().unwrap().len();
    assert!(len <= 1024 * 1024 + 100, "{len}");
    assert!(r.content[0]["text"].as_str().unwrap().contains("[cut"));
    c.close().await;
}

#[tokio::test]
async fn answers_can_arrive_out_of_order() {
    let dir = tempfile::tempdir().unwrap();
    let c = Arc::new(connect(&[], &[], dir.path()).await.unwrap());
    let slow = {
        let c = Arc::clone(&c);
        tokio::spawn(async move { c.call_tool("slow", json!({"ms": 600}), None).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let started = Instant::now();
    let echo = c.call_tool("echo", json!({"text": "quick"}), None).await.unwrap();
    assert_eq!(echo.content[0]["text"], "quick");
    assert!(started.elapsed() < Duration::from_millis(400), "the quick call did not wait for the slow one");
    let done = slow.await.unwrap().unwrap();
    assert_eq!(done.content[0]["text"], "done");
}

#[tokio::test]
async fn requests_from_the_server_are_answered_method_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&[], &[], dir.path()).await.unwrap();
    let r = c.call_tool("probe", json!({}), None).await.unwrap();
    assert_eq!(r.content[0]["text"], "code -32601");
}

#[tokio::test]
async fn a_list_changed_notification_marks_the_list_stale() {
    let dir = tempfile::tempdir().unwrap();
    let c = connect(&[], &[], dir.path()).await.unwrap();
    let before = c.list_tools().await.unwrap();
    assert!(!c.tools_stale());
    c.call_tool("mutate", json!({}), None).await.unwrap();
    assert!(c.tools_stale());
    let after = c.list_tools().await.unwrap();
    assert!(!c.tools_stale(), "reading the list clears the flag");
    assert_ne!(before[0].fingerprint(), after[0].fingerprint());
    assert!(after[0].description.contains("v2"));
}

#[tokio::test]
async fn the_child_gets_only_the_given_variables_not_the_parents() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("DARA_TEST_PARENT_ONLY", "leak");
    let c = connect(&[], &[("DARA_TEST_GIVEN", "hello")], dir.path()).await.unwrap();
    let given = c.call_tool("env", json!({"name": "DARA_TEST_GIVEN"}), None).await.unwrap();
    assert_eq!(given.content[0]["text"], "hello");
    let leaked = c.call_tool("env", json!({"name": "DARA_TEST_PARENT_ONLY"}), None).await.unwrap();
    assert_eq!(leaked.content[0]["text"], "(unset)");
    c.close().await;
}

#[tokio::test]
async fn a_command_that_does_not_exist_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = StdioConfig {
        command: "dara-no-such-command-xyz".into(),
        args: vec![],
        env: vec![],
        cwd: dir.path().to_path_buf(),
    };
    let err = StdioTransport::spawn(cfg, Shared::new()).err().unwrap();
    assert!(err.to_string().contains("could not start"), "{err}");
}
