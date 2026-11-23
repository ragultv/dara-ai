//! The shape of what the desktop shell sends the UI. `docs/ui/samples/connectors.json` is recorded from this crate's real output and is the
//! contract the TypeScript side is tested against (`app/tests/unit/connectors.test.ts`). If a field is added, renamed or retyped, this
//! test fails until the sample is re-recorded: `DARA_WRITE_SAMPLES=1 cargo test -p dara-mcp --test contract`.

use dara_mcp::{CallOutcome, Kind, Manager, MemorySecrets, NewConnector, Outcome};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

const MOCK: &str = env!("CARGO_BIN_EXE_mock_mcp");

fn sample_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../docs/ui/samples/connectors.json")
}

/// The keys and value types of a JSON value, with the values themselves dropped.
fn shape(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), shape(x))).collect()),
        Value::Array(a) => Value::Array(a.first().map(|x| vec![shape(x)]).unwrap_or_default()),
        Value::Null => json!("null"),
        Value::String(_) => json!("string"),
        Value::Number(_) => json!("number"),
        Value::Bool(_) => json!("boolean"),
    }
}

/// Same keys everywhere; a field that is null on one side matches anything (Option fields).
fn same(a: &Value, b: &Value, path: &str, problems: &mut Vec<String>) {
    match (a, b) {
        (Value::String(x), _) if x == "null" => {}
        (_, Value::String(y)) if y == "null" => {}
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => same(p, q, &format!("{path}.{k}"), problems),
                    _ => problems.push(format!("{path}.{k} is on one side only")),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            if let (Some(p), Some(q)) = (x.first(), y.first()) {
                same(p, q, &format!("{path}[]"), problems);
            }
        }
        _ => {
            if a != b {
                problems.push(format!("{path}: {a} vs {b}"));
            }
        }
    }
}

#[tokio::test]
async fn what_the_ui_receives_matches_the_recorded_sample() {
    let dir = tempfile::tempdir().unwrap();
    let m = Manager::new(&dir.path().join("c.db"), Arc::new(MemorySecrets::new()), &dir.path().join("scratch")).unwrap();
    let c = m
        .add(NewConnector {
            name: "Mock".into(),
            kind: Kind::Stdio,
            command: Some(MOCK.into()),
            args: vec![],
            env: BTreeMap::new(),
            url: None,
            oauth: None,
            api_key: None,
        })
        .unwrap();
    m.connect(&c.id).await.unwrap();

    let tools = m.tools(&c.id).unwrap();
    let args = json!({"text": "hi"});
    let ask = m.call(&c.id, "echo", args.clone(), false, None).await;
    assert!(ask.is_err(), "a tool on ask must not run unapproved");
    let ask = CallOutcome::without_result(Outcome::NeedsApproval);
    let ok = m.call(&c.id, "echo", args, true, None).await.unwrap();
    let audit = m.audit(Some(&c.id), 20).unwrap();
    let list = m.list().unwrap();
    m.close_all().await;

    let actual = json!({
        "connectors": list,
        "tools": tools,
        "callNeedsApproval": ask,
        "callOk": ok,
        "audit": audit,
    });
    let actual = serde_json::to_value(&actual).unwrap();

    let path = sample_path();
    if std::env::var("DARA_WRITE_SAMPLES").is_ok() {
        std::fs::write(&path, serde_json::to_string_pretty(&actual).unwrap() + "\n").unwrap();
    }
    let recorded: Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("docs/ui/samples/connectors.json is missing; record it with DARA_WRITE_SAMPLES=1")).unwrap();
    let mut problems = vec![];
    same(&shape(&actual), &shape(&recorded), "$", &mut problems);
    assert!(problems.is_empty(), "the payloads no longer match docs/ui/samples/connectors.json:\n{}", problems.join("\n"));
}
