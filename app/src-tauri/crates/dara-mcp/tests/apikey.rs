//! API keys for http connectors: sent with every request, kept only in the secret store, shown to the UI only as "there is one".

mod common;

use common::{start_http, MockHttp};
use dara_mcp::{
    ApiKeyInput, ConnState, ConnectorPatch, Kind, Manager, McpError, MemorySecrets, NewConnector, SecretStore,
};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::Arc;

const KEY: &str = "fc-test-key-0123456789";

struct Env {
    m: Arc<Manager>,
    secrets: Arc<MemorySecrets>,
    dir: tempfile::TempDir,
}

fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let secrets = Arc::new(MemorySecrets::new());
    let m = Manager::new(&dir.path().join("connectors.db"), secrets.clone(), &dir.path().join("scratch")).unwrap();
    Env { m, secrets, dir }
}

fn http(url: &str, key: Option<ApiKeyInput>) -> NewConnector {
    NewConnector {
        name: "Keyed".into(),
        kind: Kind::Http,
        command: None,
        args: vec![],
        env: BTreeMap::new(),
        url: Some(url.into()),
        oauth: None,
        api_key: key,
    }
}

fn key(value: &str, header: Option<&str>) -> ApiKeyInput {
    ApiKeyInput { value: Some(value.into()), header: header.map(str::to_string), clear: false }
}

/// The authorization-related headers of the requests the mock saw (lowercase names).
fn seen(mock: &MockHttp, header: &str) -> Vec<Option<String>> {
    mock.state.recorded().iter().filter(|r| r.method == "POST").map(|r| r.headers.get(header).cloned()).collect()
}

#[tokio::test]
async fn the_key_is_sent_as_a_bearer_token_by_default() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(KEY, None)))).unwrap();
    assert!(cfg.api_key.has_key);
    assert_eq!(cfg.api_key.header, None);
    let st = e.m.connect(&cfg.id).await.unwrap();
    assert_eq!(st.state, ConnState::Ready);
    let auth = seen(&mock, "authorization");
    assert!(!auth.is_empty());
    assert!(auth.iter().all(|a| a.as_deref() == Some(&format!("Bearer {KEY}")[..])), "{auth:?}");
}

#[tokio::test]
async fn a_custom_header_carries_the_key_as_it_is() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(KEY, Some("X-API-Key"))))).unwrap();
    assert_eq!(cfg.api_key.header.as_deref(), Some("X-API-Key"));
    assert_eq!(e.m.connect(&cfg.id).await.unwrap().state, ConnState::Ready);
    assert!(seen(&mock, "x-api-key").iter().all(|v| v.as_deref() == Some(KEY)));
    assert!(seen(&mock, "authorization").iter().all(Option::is_none));
}

#[tokio::test]
async fn a_pasted_bearer_prefix_is_not_doubled() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(&format!("Bearer {KEY}"), None)))).unwrap();
    e.m.connect(&cfg.id).await.unwrap();
    assert!(seen(&mock, "authorization").iter().all(|a| a.as_deref() == Some(&format!("Bearer {KEY}")[..])));
}

#[tokio::test]
async fn the_key_lives_only_in_the_secret_store() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(KEY, None)))).unwrap();
    e.m.connect(&cfg.id).await.unwrap();
    // in the secret store, under the connector's id
    assert!(e.secrets.keys().iter().any(|k| k.contains(&cfg.id) && k.ends_with("api_key")));
    // not in what the UI gets
    let shown = serde_json::to_string(&json!(e.m.list().unwrap().iter().map(|v| format!("{v:?}")).collect::<Vec<_>>())).unwrap();
    assert!(!shown.contains(KEY));
    // not in the database file (the WAL holds recent writes, so read every file the database wrote)
    for entry in std::fs::read_dir(e.dir.path()).unwrap().flatten() {
        if entry.file_type().unwrap().is_file() {
            let bytes = std::fs::read(entry.path()).unwrap();
            assert!(!String::from_utf8_lossy(&bytes).contains(KEY), "{:?}", entry.path());
        }
    }
}

#[tokio::test]
async fn the_key_is_kept_out_of_the_audit_log() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(KEY, None)))).unwrap();
    e.m.connect(&cfg.id).await.unwrap();
    let r = e.m.call(&cfg.id, "echo", json!({"text": KEY}), true, None).await.unwrap();
    assert_eq!(r.content[0]["text"], KEY); // the server may say anything; only what Dara records is scrubbed
    let log = e.m.audit(Some(&cfg.id), 10).unwrap();
    assert!(!log.is_empty());
    assert!(log.iter().all(|a| !a.args_json.contains(KEY)), "{log:?}");
}

#[tokio::test]
async fn changing_and_clearing_the_key_takes_effect_on_the_next_connection() {
    let e = env();
    let mock = start_http().await;
    let cfg = e.m.add(http(&mock.url(), Some(key(KEY, None)))).unwrap();
    e.m.connect(&cfg.id).await.unwrap();

    // a new key replaces the old one, and the live connection is dropped
    let upd = ConnectorPatch { api_key: Some(key("second-key", None)), ..Default::default() };
    assert!(e.m.update(&cfg.id, upd).await.unwrap().api_key.has_key);
    assert_ne!(e.m.status_of(&cfg.id).state, ConnState::Ready);
    e.m.connect(&cfg.id).await.unwrap();
    let last = seen(&mock, "authorization").into_iter().last().unwrap();
    assert_eq!(last.as_deref(), Some("Bearer second-key"));

    // changing only the header keeps the key
    let upd = ConnectorPatch { api_key: Some(ApiKeyInput { header: Some("X-Token".into()), ..Default::default() }), ..Default::default() };
    let c = e.m.update(&cfg.id, upd).await.unwrap();
    assert_eq!((c.api_key.has_key, c.api_key.header.as_deref()), (true, Some("X-Token")));
    e.m.connect(&cfg.id).await.unwrap();
    assert_eq!(seen(&mock, "x-token").into_iter().last().unwrap().as_deref(), Some("second-key"));

    // clearing removes it for good
    let upd = ConnectorPatch { api_key: Some(ApiKeyInput { clear: true, ..Default::default() }), ..Default::default() };
    let c = e.m.update(&cfg.id, upd).await.unwrap();
    assert_eq!((c.api_key.has_key, c.api_key.header), (false, None));
    assert!(e.secrets.keys().iter().all(|k| !k.ends_with("api_key")));
    e.m.connect(&cfg.id).await.unwrap();
    assert!(seen(&mock, "authorization").into_iter().last().unwrap().is_none());
}

#[tokio::test]
async fn a_refused_key_is_an_error_about_the_key_not_a_sign_in_prompt() {
    let e = env();
    let mock = start_http().await;
    mock.state.unauthorized.store(true, Ordering::SeqCst);
    let cfg = e.m.add(http(&mock.url(), Some(key("wrong-key", None)))).unwrap();
    assert!(matches!(e.m.connect(&cfg.id).await, Err(McpError::AuthRequired { .. })));
    let st = e.m.status_of(&cfg.id);
    assert_eq!(st.state, ConnState::Error);
    assert!(st.error.unwrap().contains("API key"));

    // the same server with no key is a sign-in prompt, as before
    let plain = e.m.add(http(&mock.url(), None)).unwrap();
    assert!(e.m.connect(&plain.id).await.is_err());
    assert_eq!(e.m.status_of(&plain.id).state, ConnState::NeedsAuth);
}

#[tokio::test]
async fn bad_input_is_refused() {
    let e = env();
    let bad = |k: ApiKeyInput| e.m.add(http("https://example.com/mcp", Some(k))).err().map(|x| x.to_string());
    assert!(bad(key("two words", None)).unwrap().contains("paste just the key"));
    assert!(bad(key("a\nb", None)).is_some());
    assert!(bad(key(&"k".repeat(1501), None)).unwrap().contains("longer"));
    assert!(bad(key(KEY, Some("Host"))).unwrap().contains("connection itself"));
    assert!(bad(key(KEY, Some("bad header"))).is_some());
    // a header with no key, and a key on a program, make no sense
    assert!(bad(ApiKeyInput { header: Some("X-API-Key".into()), ..Default::default() }).unwrap().contains("empty"));
    let mut stdio = http("https://example.com/mcp", Some(key(KEY, None)));
    stdio.kind = Kind::Stdio;
    stdio.url = None;
    stdio.command = Some("x".into());
    assert!(e.m.add(stdio).is_err());
    // an empty form (no key, no header) is simply no key
    assert!(!e.m.add(http("https://example.com/mcp", Some(ApiKeyInput::default()))).unwrap().api_key.has_key);
    // nothing was left behind by the refused ones
    assert!(e.secrets.keys().is_empty());
    // patching a connector that has no key with just a header is refused
    let c = e.m.add(http("https://example.com/mcp", None)).unwrap();
    let upd = ConnectorPatch { api_key: Some(ApiKeyInput { header: Some("X-API-Key".into()), ..Default::default() }), ..Default::default() };
    assert!(e.m.update(&c.id, upd).await.is_err());
}

#[tokio::test]
async fn removing_the_connector_removes_the_key() {
    let e = env();
    let cfg = e.m.add(http("https://example.com/mcp", Some(key(KEY, None)))).unwrap();
    assert!(e.secrets.get(&format!("connector/{}/api_key", cfg.id)).unwrap().is_some());
    e.m.remove(&cfg.id).await.unwrap();
    assert!(e.secrets.keys().is_empty(), "{:?}", e.secrets.keys());
}
