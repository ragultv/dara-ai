//! Live check of a server that needs no sign-in: connects through Dara's own connector manager (sessions, headers and all) and lists
//! the tools it offers. Nothing is stored (secrets are kept in memory, the database is a temporary file).
//!
//!   cargo run -p dara-mcp --example probe_connect -- https://mcp.context7.com/mcp https://mcp.deepwiki.com/mcp
//!   cargo run -p dara-mcp --example probe_connect -- --json out.json <urls...>    also writes every tool (name, description, schema) as JSON
//!   DARA_PROBE_API_KEY=... cargo run -p dara-mcp --example probe_connect -- --key-header X-API-Key <url>   for a server that wants a key
use dara_mcp::{ApiKeyInput, Kind, Manager, MemorySecrets, NewConnector};
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut json_out: Option<String> = None;
    let mut key_header: Option<String> = None;
    let mut urls: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--json" => json_out = args.next(),
            "--key-header" => key_header = args.next(),
            _ => urls.push(a),
        }
    }
    let api_key = std::env::var("DARA_PROBE_API_KEY").ok().filter(|k| !k.is_empty());
    let dir = std::env::temp_dir().join(format!("dara-probe-{}", std::process::id()));
    let m = Manager::new(&dir.join("c.db"), Arc::new(MemorySecrets::new()), &dir.join("scratch")).expect("manager");
    let mut all = Vec::new();
    for u in urls {
        println!("{u}");
        let added = m.add(NewConnector {
            name: "probe".into(),
            kind: Kind::Http,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            url: Some(u.clone()),
            oauth: None,
            api_key: api_key.as_ref().map(|k| ApiKeyInput { value: Some(k.clone()), header: key_header.clone(), clear: false }),
        });
        let id = match added {
            Ok(c) => c.id,
            Err(e) => {
                println!("   could not add: {e}");
                continue;
            }
        };
        match m.connect(&id).await {
            Ok(st) => {
                let tools = m.tools(&id).unwrap_or_default();
                let server = st.server.as_ref().map(|s| format!("{} {}", s.name, s.version)).unwrap_or_default();
                println!("   {:?}: {server}, {} tools", st.state, tools.len());
                for t in &tools {
                    println!("     - {}", t.name);
                }
                all.push(json!({ "url": u, "server": server, "tools": tools }));
            }
            Err(e) => println!("   FAILED: {e} ({:?})", m.status_of(&id).state),
        }
        let _ = m.remove(&id).await;
    }
    if let Some(path) = json_out {
        std::fs::write(&path, serde_json::to_string_pretty(&all).unwrap()).expect("write json");
        println!("wrote {path}");
    }
    m.close_all().await;
    let _ = std::fs::remove_dir_all(&dir);
}
