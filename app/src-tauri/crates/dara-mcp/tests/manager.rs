use dara_mcp::{
    ConnState, ConnectorPatch, Kind, Manager, McpError, MemorySecrets, NewConnector, Outcome, RemoteTool, SecretStore,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

const MOCK: &str = env!("CARGO_BIN_EXE_mock_mcp");

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

fn stdio(name: &str, vars: &[(&str, &str)]) -> NewConnector {
    NewConnector {
        name: name.into(),
        kind: Kind::Stdio,
        command: Some(MOCK.into()),
        args: vec![],
        env: vars.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
        url: None,
    }
}

fn text(r: &dara_mcp::CallOutcome) -> String {
    r.content[0]["text"].as_str().unwrap_or("").to_string()
}

#[tokio::test]
async fn adding_checks_its_input() {
    let e = env();
    let bad = |c: NewConnector| e.m.add(c).err().map(|x| x.to_string());
    assert!(bad(stdio("", &[])).unwrap().contains("1 to 60"));
    assert!(bad(stdio(&"n".repeat(61), &[])).unwrap().contains("1 to 60"));
    let mut c = stdio("x", &[]);
    c.command = Some("   ".into());
    assert!(bad(c).unwrap().contains("command"));
    let mut c = stdio("x", &[]);
    c.command = Some("a\nb".into());
    assert!(bad(c).unwrap().contains("one line"));
    assert!(bad(stdio("x", &[("1BAD", "v")])).unwrap().contains("variable name"));
    assert!(bad(stdio("x", &[("OK", "")])).unwrap().contains("needs a value"));
    let http = |u: &str| NewConnector {
        name: "h".into(),
        kind: Kind::Http,
        command: None,
        args: vec![],
        env: BTreeMap::new(),
        url: Some(u.into()),
    };
    assert!(bad(http("http://example.com/mcp")).unwrap().contains("https"));
    assert!(bad(http("https://u:p@example.com/mcp")).is_some());
    assert!(e.m.add(http("https://example.com/mcp")).is_ok());
    assert!(e.m.add(http("http://127.0.0.1:1/mcp")).is_ok());
    // the limit
    for i in 0..30 {
        e.m.add(stdio(&format!("c{i}"), &[])).unwrap();
    }
    assert_eq!(e.m.list().unwrap().len(), 32);
    assert!(bad(stdio("one too many", &[])).unwrap().contains("at most 32"));
}

#[tokio::test]
async fn a_server_claiming_read_only_still_starts_at_ask() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    let st = e.m.connect(&id).await.unwrap();
    assert_eq!((st.state, st.tool_count), (ConnState::Ready, 8));
    assert_eq!(st.server.unwrap().name, "mock-mcp");
    let tools = e.m.tools(&id).unwrap();
    assert_eq!(tools.len(), 8);
    let echo = tools.iter().find(|t| t.name == "echo").unwrap();
    assert_eq!(echo.read_only_hint, Some(true), "the claim is shown");
    assert!(tools.iter().all(|t| t.permission == "ask"), "but it grants nothing");
    assert_eq!(echo.input_schema["type"], "object");
    e.m.close_all().await;
}

#[tokio::test]
async fn permissions_gate_calls_and_every_attempt_is_audited() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    e.m.connect(&id).await.unwrap();
    let args = json!({"text": "hi"});

    // ask, not approved: nothing is sent
    assert!(matches!(e.m.call(&id, "echo", args.clone(), false, None).await, Err(McpError::NeedsApproval)));
    // ask, approved: called
    let r = e.m.call(&id, "echo", args.clone(), true, None).await.unwrap();
    assert_eq!((r.outcome, text(&r).as_str()), (Outcome::Ok, "hi"));
    // allow: no approval needed
    e.m.set_permission(&id, "echo", "allow").unwrap();
    assert!(e.m.call(&id, "echo", args.clone(), false, None).await.is_ok());
    // deny: even an approved call is refused
    e.m.set_permission(&id, "echo", "deny").unwrap();
    assert!(matches!(e.m.call(&id, "echo", args.clone(), true, None).await, Err(McpError::Denied)));
    // a tool that reports an error is an "error" outcome
    let f = e.m.call(&id, "fail", json!({}), true, None).await.unwrap();
    assert_eq!((f.outcome, f.is_error), (Outcome::Error, true));
    // unknown tool, unknown connector
    assert!(matches!(e.m.call(&id, "nope", json!({}), true, None).await, Err(McpError::NotFound(_))));
    assert!(matches!(e.m.call("nope", "echo", json!({}), true, None).await, Err(McpError::NotFound(_))));
    assert!(e.m.set_permission(&id, "echo", "sometimes").is_err());

    let audit = e.m.audit(Some(&id), 50).unwrap();
    let outcomes: Vec<&str> = audit.iter().rev().map(|a| a.outcome.as_str()).collect();
    assert_eq!(outcomes, ["needs_approval", "ok", "ok", "denied", "error"]);
    assert_eq!(audit.last().unwrap().tool, "echo");
    assert!(audit.iter().any(|a| a.args_json.contains("\"hi\"")));
    assert!(audit.iter().find(|a| a.outcome == "ok").unwrap().result_chars == 2);
    e.m.close_all().await;
}

#[tokio::test]
async fn a_tool_that_changes_goes_back_to_ask_and_is_flagged() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    e.m.connect(&id).await.unwrap();
    e.m.set_permission(&id, "echo", "allow").unwrap();
    e.m.set_permission(&id, "mutate", "allow").unwrap();
    // the server changes echo's description and says so
    e.m.call(&id, "mutate", json!({}), false, None).await.unwrap();
    // the next allowed call notices the change first and stops: what was allowed is not what is there now
    let r = e.m.call(&id, "echo", json!({"text": "x"}), false, None).await;
    assert!(matches!(r, Err(McpError::NeedsApproval)), "{r:?}");
    let echo = e.m.tools(&id).unwrap().into_iter().find(|t| t.name == "echo").unwrap();
    assert_eq!((echo.permission.as_str(), echo.changed), ("ask", true));
    assert!(echo.description.contains("v2"));
    assert!(e.m.status_of(&id).tools_changed);
    // an unchanged tool keeps its choice
    let mutate = e.m.tools(&id).unwrap().into_iter().find(|t| t.name == "mutate").unwrap();
    assert_eq!((mutate.permission.as_str(), mutate.changed), ("allow", false));
    // choosing again clears the flag
    e.m.set_permission(&id, "echo", "allow").unwrap();
    assert!(!e.m.tools(&id).unwrap().into_iter().find(|t| t.name == "echo").unwrap().changed);
    assert!(e.m.call(&id, "echo", json!({"text": "x"}), false, None).await.is_ok());
    e.m.close_all().await;
}

fn tool(name: &str) -> RemoteTool {
    RemoteTool {
        name: name.into(),
        title: None,
        description: "d".into(),
        input_schema: json!({"type": "object"}),
        annotations: None,
        read_only_hint: None,
    }
}

#[tokio::test]
async fn a_tool_that_disappears_keeps_the_users_choice_but_is_unavailable() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    e.m.store().sync_tools(&id, &[tool("a"), tool("b")]).unwrap();
    e.m.set_permission(&id, "b", "deny").unwrap();
    e.m.store().sync_tools(&id, &[tool("a")]).unwrap();
    let tools = e.m.tools(&id).unwrap();
    let b = tools.iter().find(|t| t.name == "b").unwrap();
    assert_eq!((b.available, b.permission.as_str()), (false, "deny"));
    assert!(matches!(e.m.call(&id, "b", json!({}), true, None).await, Err(McpError::NotFound(_))));
    // it comes back with its choice
    e.m.store().sync_tools(&id, &[tool("a"), tool("b")]).unwrap();
    let b = e.m.tools(&id).unwrap().into_iter().find(|t| t.name == "b").unwrap();
    assert_eq!((b.available, b.permission.as_str()), (true, "deny"));
}

#[tokio::test]
async fn the_audit_log_keeps_the_newest_5000_rows_and_cuts_long_arguments() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    for i in 0..5100 {
        e.m.store().audit_add(&id, "echo", &format!("{{\"n\":{i}}}"), "ok", 1, 1).unwrap();
    }
    assert_eq!(e.m.store().audit_count().unwrap(), 5000);
    let newest = e.m.audit(Some(&id), 1).unwrap();
    assert!(newest[0].args_json.contains("5099"));
    let oldest = e.m.audit(Some(&id), 1000).unwrap();
    assert_eq!(oldest.len(), 1000);
    e.m.store().audit_add(&id, "echo", &"é".repeat(9000), "ok", 1, 1).unwrap();
    let cut = &e.m.audit(Some(&id), 1).unwrap()[0].args_json;
    assert!(cut.len() <= 8 * 1024 && cut.chars().all(|c| c == 'é'));
}

#[tokio::test]
async fn secrets_live_only_in_the_secret_store() {
    let e = env();
    let secret = "s3cr3t-value-123";
    let c = e.m.add(stdio("mock", &[("API_TOKEN", secret)])).unwrap();
    assert_eq!(c.env_names, ["API_TOKEN"]);
    assert_eq!(e.secrets.keys(), [format!("connector/{}/env/API_TOKEN", c.id)]);
    assert_eq!(e.secrets.get(&e.secrets.keys()[0]).unwrap().as_deref(), Some(secret));

    // the child gets it
    e.m.connect(&c.id).await.unwrap();
    let r = e.m.call(&c.id, "env", json!({"name": "API_TOKEN"}), true, None).await.unwrap();
    assert_eq!(text(&r), secret);
    e.m.close_all().await;

    // the database files do not contain it (the audit stores arguments, not variables)
    for name in ["connectors.db", "connectors.db-wal", "connectors.db-shm"] {
        if let Ok(bytes) = std::fs::read(e.dir.path().join(name)) {
            let hay = String::from_utf8_lossy(&bytes);
            assert!(!hay.contains(secret), "{name} holds the secret");
        }
    }
    // removing the connector removes its secrets, tools and audit rows
    e.m.remove(&c.id).await.unwrap();
    assert!(e.secrets.keys().is_empty());
    assert!(e.m.list().unwrap().is_empty());
    assert!(e.m.audit(None, 10).unwrap().is_empty());
}

#[tokio::test]
async fn editing_variables_keeps_blank_values_and_drops_removed_names() {
    let e = env();
    let id = e.m.add(stdio("mock", &[("A", "1"), ("B", "2")])).unwrap().id;
    let mut env_map = BTreeMap::new();
    env_map.insert("A".to_string(), String::new()); // keep what is stored
    env_map.insert("C".to_string(), "3".to_string());
    let cfg = e.m.update(&id, ConnectorPatch { env: Some(env_map), ..Default::default() }).await.unwrap();
    assert_eq!(cfg.env_names, ["A", "C"]);
    assert_eq!(e.secrets.get(&format!("connector/{id}/env/A")).unwrap().as_deref(), Some("1"));
    assert_eq!(e.secrets.get(&format!("connector/{id}/env/B")).unwrap(), None);
    let mut bad = BTreeMap::new();
    bad.insert("NEW".to_string(), String::new());
    assert!(e.m.update(&id, ConnectorPatch { env: Some(bad), ..Default::default() }).await.is_err());
}

#[tokio::test]
async fn a_turned_off_connector_does_not_connect() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    e.m.connect(&id).await.unwrap();
    e.m.update(&id, ConnectorPatch { enabled: Some(false), ..Default::default() }).await.unwrap();
    assert_eq!(e.m.status_of(&id).state, ConnState::Disconnected);
    assert!(matches!(e.m.connect(&id).await, Err(McpError::Invalid(_))));
}

#[tokio::test]
async fn a_crash_is_reported_and_the_next_call_connects_again() {
    let e = env();
    let id = e.m.add(stdio("mock", &[])).unwrap().id;
    e.m.connect(&id).await.unwrap();
    let err = e.m.call(&id, "crash", json!({}), true, None).await.err().unwrap();
    assert!(matches!(err, McpError::Closed(_)), "{err:?}");
    assert_eq!(e.m.status_of(&id).state, ConnState::Error);
    let r = e.m.call(&id, "echo", json!({"text": "back"}), true, None).await.unwrap();
    assert_eq!(text(&r), "back");
    assert_eq!(e.m.status_of(&id).state, ConnState::Ready);
    e.m.close_all().await;
}

#[tokio::test]
async fn a_failed_connection_is_a_state_with_a_reason() {
    let e = env();
    let mut c = stdio("nope", &[]);
    c.command = Some("dara-no-such-command-xyz".into());
    let id = e.m.add(c).unwrap().id;
    assert!(e.m.connect(&id).await.is_err());
    let st = e.m.status_of(&id);
    assert_eq!(st.state, ConnState::Error);
    assert!(st.error.unwrap().contains("could not start"));
    // an unreachable http server too
    let http = e
        .m
        .add(NewConnector {
            name: "h".into(),
            kind: Kind::Http,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            url: Some("http://127.0.0.1:1/mcp".into()),
        })
        .unwrap();
    assert!(e.m.connect(&http.id).await.is_err());
    assert_eq!(e.m.status_of(&http.id).state, ConnState::Error);
}

#[tokio::test]
async fn the_database_is_versioned_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.db");
    let id = {
        let m = Manager::new(&path, Arc::new(MemorySecrets::new()), dir.path()).unwrap();
        assert_eq!(m.store().schema_version().unwrap(), 1);
        m.add(stdio("kept", &[])).unwrap().id
    };
    let m = Manager::new(&path, Arc::new(MemorySecrets::new()), dir.path()).unwrap();
    assert_eq!(m.list().unwrap()[0].config.id, id);
    let _: Value = json!(null);
}

#[cfg(windows)]
#[test]
fn the_windows_credential_store_round_trips_and_forgets() {
    use dara_mcp::KeyringSecrets;
    let store = KeyringSecrets::new("Dara-test");
    let key = format!("connector/test-{}/env/X", std::process::id());
    assert_eq!(store.get(&key).unwrap(), None);
    store.set(&key, "value-1").unwrap();
    assert_eq!(store.get(&key).unwrap().as_deref(), Some("value-1"));
    store.set(&key, "value-2").unwrap();
    assert_eq!(store.get(&key).unwrap().as_deref(), Some("value-2"));
    store.delete(&key).unwrap();
    assert_eq!(store.get(&key).unwrap(), None);
    store.delete(&key).unwrap(); // deleting what is gone is fine
}

/// The reset lives in the store, so it applies to every kind of connector, not just the stdio mock.
#[tokio::test]
async fn a_changed_tool_goes_back_to_ask_for_an_http_connector_too() {
    let e = env();
    let c = e
        .m
        .add(NewConnector {
            name: "Remote".into(),
            kind: Kind::Http,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            url: Some("https://example.com/mcp".into()),
        })
        .unwrap();
    let tool = |description: &str, title: Option<&str>| RemoteTool {
        name: "send".into(),
        title: title.map(str::to_string),
        description: description.into(),
        input_schema: json!({"type": "object"}),
        annotations: None,
        read_only_hint: None,
    };
    let store = e.m.store();
    store.sync_tools(&c.id, &[tool("Sends a message", None)]).unwrap();
    e.m.set_permission(&c.id, "send", "allow").unwrap();
    // the same tool again: the choice stays
    store.sync_tools(&c.id, &[tool("Sends a message", None)]).unwrap();
    assert_eq!(e.m.tools(&c.id).unwrap()[0].permission, "allow");
    // a new description: back to ask, flagged
    store.sync_tools(&c.id, &[tool("Sends a message and copies it to a third party", None)]).unwrap();
    let t = &e.m.tools(&c.id).unwrap()[0];
    assert_eq!((t.permission.as_str(), t.changed), ("ask", true));
    // a new title counts too: it is part of what the user saw
    e.m.set_permission(&c.id, "send", "allow").unwrap();
    store
        .sync_tools(&c.id, &[tool("Sends a message and copies it to a third party", Some("Read mail"))])
        .unwrap();
    assert_eq!(e.m.tools(&c.id).unwrap()[0].permission, "ask");
}
