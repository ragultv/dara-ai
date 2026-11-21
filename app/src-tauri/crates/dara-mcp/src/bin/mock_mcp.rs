//! A small MCP server for tests: newline-delimited JSON on stdio. Not shipped.
//!
//! Tools: echo, slow, fail, big, crash, mutate, probe, env. The tool list comes in two pages.
//! `--protocol <version>` forces the version it answers `initialize` with. `MOCK_LOG=<file>` collects what the client sent as
//! notifications (`cancelled:<id>`).

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type Out = Arc<Mutex<std::io::Stdout>>;

fn send(out: &Out, v: &Value) {
    let mut o = out.lock().unwrap();
    let _ = writeln!(o, "{v}");
    let _ = o.flush();
}

fn text_result(t: &str) -> Value {
    json!({"content": [{"type": "text", "text": t}]})
}

fn tool(name: &str, description: &str, read_only: Option<bool>) -> Value {
    let mut t = json!({
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
    });
    if let Some(r) = read_only {
        t["annotations"] = json!({"readOnlyHint": r});
    }
    t
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let forced = args.windows(2).find(|w| w[0] == "--protocol").map(|w| w[1].clone());
    let log_path = std::env::var("MOCK_LOG").ok();
    let out: Out = Arc::new(Mutex::new(std::io::stdout()));
    let v2 = Arc::new(AtomicBool::new(false));
    let replies: Arc<Mutex<HashMap<String, Value>>> = Arc::new(Mutex::new(HashMap::new()));

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let Ok(msg) = serde_json::from_str::<Value>(&line) else { continue };
        let method = msg.get("method").and_then(Value::as_str).map(str::to_string);
        let id = msg.get("id").cloned();
        match (method, id) {
            (Some(m), None) => {
                if m == "notifications/cancelled" {
                    if let Some(p) = &log_path {
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                            let _ = writeln!(f, "cancelled:{}", msg["params"]["requestId"]);
                        }
                    }
                }
            }
            (Some(m), Some(id)) => {
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                match m.as_str() {
                    "initialize" => {
                        let asked = params["protocolVersion"].as_str().unwrap_or("2025-06-18").to_string();
                        let version = forced.clone().unwrap_or(asked);
                        send(
                            &out,
                            &json!({"jsonrpc": "2.0", "id": id, "result": {
                                "protocolVersion": version,
                                "capabilities": {"tools": {"listChanged": true}},
                                "serverInfo": {"name": "mock-mcp", "version": "0.0.1"}
                            }}),
                        );
                    }
                    "ping" => send(&out, &json!({"jsonrpc": "2.0", "id": id, "result": {}})),
                    "tools/list" => {
                        let echo_desc = if v2.load(Ordering::SeqCst) { "Echo the text (v2)" } else { "Echo the text" };
                        let result = if params["cursor"].as_str() == Some("p2") {
                            json!({"tools": [
                                tool("fail", "Always fails", None),
                                tool("big", "A huge result", None),
                                tool("crash", "Exits the process", None),
                                tool("mutate", "Changes the tool list", Some(true)),
                                tool("probe", "Asks the client for sampling", None),
                                tool("env", "Reads an environment variable", None)
                            ]})
                        } else {
                            json!({"tools": [
                                tool("echo", echo_desc, Some(true)),
                                tool("slow", "Sleeps", None)
                            ], "nextCursor": "p2"})
                        };
                        send(&out, &json!({"jsonrpc": "2.0", "id": id, "result": result}));
                    }
                    "tools/call" => {
                        let name = params["name"].as_str().unwrap_or("").to_string();
                        let a = params["arguments"].clone();
                        let out = Arc::clone(&out);
                        let v2 = Arc::clone(&v2);
                        let replies = Arc::clone(&replies);
                        std::thread::spawn(move || {
                            let result = match name.as_str() {
                                "echo" => text_result(a["text"].as_str().unwrap_or("")),
                                "slow" => {
                                    std::thread::sleep(Duration::from_millis(a["ms"].as_u64().unwrap_or(5000)));
                                    text_result("done")
                                }
                                "fail" => json!({"content": [{"type": "text", "text": "boom"}], "isError": true}),
                                "big" => text_result(&"x".repeat(2 * 1024 * 1024)),
                                "crash" => std::process::exit(3),
                                "mutate" => {
                                    v2.store(true, Ordering::SeqCst);
                                    send(&out, &json!({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}));
                                    text_result("mutated")
                                }
                                "probe" => {
                                    send(
                                        &out,
                                        &json!({"jsonrpc": "2.0", "id": "s1", "method": "sampling/createMessage", "params": {}}),
                                    );
                                    let start = Instant::now();
                                    let mut text = "no reply".to_string();
                                    while start.elapsed() < Duration::from_secs(3) {
                                        if let Some(r) = replies.lock().unwrap().get("s1") {
                                            text = format!("code {}", r["error"]["code"]);
                                            break;
                                        }
                                        std::thread::sleep(Duration::from_millis(20));
                                    }
                                    text_result(&text)
                                }
                                "env" => text_result(
                                    &std::env::var(a["name"].as_str().unwrap_or("")).unwrap_or_else(|_| "(unset)".into()),
                                ),
                                other => {
                                    send(
                                        &out,
                                        &json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": format!("unknown tool {other}")}}),
                                    );
                                    return;
                                }
                            };
                            send(&out, &json!({"jsonrpc": "2.0", "id": id, "result": result}));
                        });
                    }
                    other => send(
                        &out,
                        &json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": format!("Method not found: {other}")}}),
                    ),
                }
            }
            (None, Some(id)) => {
                // a reply to a request the server sent (the probe)
                replies.lock().unwrap().insert(id.as_str().unwrap_or("").to_string(), msg);
            }
            (None, None) => {}
        }
    }
}
