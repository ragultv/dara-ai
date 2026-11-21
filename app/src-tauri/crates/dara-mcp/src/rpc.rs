//! JSON-RPC 2.0 messages as MCP uses them.

use crate::error::McpError;
use serde_json::{json, Value};

/// A message that arrived from the server.
#[derive(Debug, Clone)]
pub enum Incoming {
    /// An answer to one of our requests.
    Response { id: Value, result: Result<Value, McpError> },
    /// The server asks us something (sampling, roots, elicitation...).
    Request { id: Value, method: String },
    Notification { method: String },
}

/// Reads one JSON-RPC message. Anything that is not a well-formed message is a protocol error.
pub fn parse_incoming(v: &Value) -> Result<Incoming, McpError> {
    let obj = v
        .as_object()
        .ok_or_else(|| McpError::Protocol("a JSON-RPC message must be an object".into()))?;
    let method = obj.get("method").and_then(Value::as_str);
    match (method, obj.get("id")) {
        (Some(m), Some(id)) if !id.is_null() => Ok(Incoming::Request {
            id: id.clone(),
            method: m.to_string(),
        }),
        (Some(m), _) => Ok(Incoming::Notification { method: m.to_string() }),
        (None, Some(id)) => {
            let result = if let Some(err) = obj.get("error") {
                Err(McpError::Rpc {
                    code: err.get("code").and_then(Value::as_i64).unwrap_or(-32603),
                    message: err
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                        .to_string(),
                })
            } else if let Some(r) = obj.get("result") {
                Ok(r.clone())
            } else {
                Err(McpError::Protocol("a response has neither result nor error".into()))
            };
            Ok(Incoming::Response { id: id.clone(), result })
        }
        (None, None) => Err(McpError::Protocol("not a JSON-RPC message".into())),
    }
}

pub fn request(id: u64, method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})
}

pub fn notification(method: &str, params: Value) -> Value {
    json!({"jsonrpc": "2.0", "method": method, "params": params})
}

/// The answer to a server request we do not support: the client declares no capabilities, so "method not found".
pub fn method_not_found(id: &Value, method: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {"code": -32601, "message": format!("Method not found: {method}")}
    })
}

/// A numeric id from a response `id` (we only ever send numbers).
pub fn id_as_u64(id: &Value) -> Option<u64> {
    id.as_u64()
        .or_else(|| id.as_str().and_then(|s| s.parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_three_kinds() {
        assert!(matches!(
            parse_incoming(&json!({"jsonrpc":"2.0","id":1,"result":{}})),
            Ok(Incoming::Response { .. })
        ));
        assert!(matches!(
            parse_incoming(&json!({"jsonrpc":"2.0","id":"a","method":"sampling/createMessage"})),
            Ok(Incoming::Request { .. })
        ));
        assert!(matches!(
            parse_incoming(&json!({"jsonrpc":"2.0","method":"notifications/x"})),
            Ok(Incoming::Notification { .. })
        ));
        assert!(parse_incoming(&json!([1])).is_err());
        assert!(parse_incoming(&json!({"jsonrpc":"2.0","id":1})).is_err() == false);
    }

    #[test]
    fn an_error_response_keeps_its_code() {
        let m = parse_incoming(&json!({"id":2,"error":{"code":-32602,"message":"bad"}})).unwrap();
        match m {
            Incoming::Response { result: Err(McpError::Rpc { code, .. }), .. } => assert_eq!(code, -32602),
            other => panic!("{other:?}"),
        }
    }
}
