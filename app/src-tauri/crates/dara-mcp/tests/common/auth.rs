//! A mock MCP server that wants a bearer token, and the authorization server that issues it, on one loopback port. Every behaviour a test
//! needs to break is a knob. The mock also enforces what a real server enforces (PKCE S256, exact code use, redirect address, resource).
#![allow(dead_code)]

use super::{read_request, respond, Recorded};
use dara_mcp::oauth::b64url;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[derive(Clone)]
pub struct Knobs {
    /// put `resource_metadata` in the 401's WWW-Authenticate
    pub header_hint: bool,
    pub challenge_scope: Option<String>,
    pub omit_s256: bool,
    pub meta_issuer: Option<String>,
    pub token_endpoint_override: Option<String>,
    pub registration_endpoint_override: Option<String>,
    pub registration: bool,
    pub registration_rejects_redirect: bool,
    pub iss_in_response: Option<String>,
    pub iss_promised: bool,
    pub cimd_flag: bool,
    pub state_override: Option<String>,
    pub authorize_error: Option<String>,
    pub code_ttl_ms: i64,
    pub expires_in: i64,
    pub big_token: bool,
    pub rotate_refresh: bool,
    pub refresh_revoked: bool,
    pub mcp_always_401: bool,
    pub mcp_open: bool,
    /// tools/call needs a token whose scope contains this
    pub call_needs_scope: Option<String>,
    pub prm_authorization_servers: Option<Vec<String>>,
    /// the authorization response goes here instead of to the redirect address
    pub redirect_elsewhere: Option<String>,
    /// client id -> secret that must come with the token request
    pub client_secrets: HashMap<String, String>,
}

impl Default for Knobs {
    fn default() -> Self {
        Knobs {
            header_hint: true,
            challenge_scope: None,
            omit_s256: false,
            meta_issuer: None,
            token_endpoint_override: None,
            registration_endpoint_override: None,
            registration: true,
            registration_rejects_redirect: false,
            iss_in_response: None,
            iss_promised: false,
            cimd_flag: false,
            state_override: None,
            authorize_error: None,
            code_ttl_ms: 60_000,
            expires_in: 3600,
            big_token: false,
            rotate_refresh: false,
            refresh_revoked: false,
            mcp_always_401: false,
            mcp_open: false,
            call_needs_scope: None,
            prm_authorization_servers: None,
            redirect_elsewhere: None,
            client_secrets: HashMap::new(),
        }
    }
}

struct Code {
    challenge: String,
    redirect_uri: String,
    client_id: String,
    resource: String,
    scope: String,
    created_ms: i64,
    used: bool,
}

#[derive(Default)]
pub struct Log {
    pub paths: Vec<String>,
    pub mcp: Vec<Recorded>,
    pub authorizes: Vec<HashMap<String, String>>,
    pub registrations: Vec<Value>,
    pub token_requests: Vec<HashMap<String, String>>,
    pub revocations: Vec<HashMap<String, String>>,
}

#[derive(Default)]
struct Issued {
    /// access token -> (expires_at_ms, scope)
    access: HashMap<String, (i64, String)>,
    /// valid refresh token -> (client id, scope)
    refresh: HashMap<String, (String, String)>,
    last_access: String,
    last_refresh: String,
    n: usize,
}

pub struct Inner {
    pub knobs: Mutex<Knobs>,
    pub log: Mutex<Log>,
    codes: Mutex<HashMap<String, Code>>,
    issued: Mutex<Issued>,
    registered: Mutex<HashMap<String, String>>, // dcr client id -> redirect uri
    pub refresh_requests: AtomicUsize,
    port: u16,
}

pub struct MockAuth {
    pub addr: SocketAddr,
    pub st: Arc<Inner>,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

impl MockAuth {
    pub fn mcp_url(&self) -> String {
        format!("http://127.0.0.1:{}/mcp", self.addr.port())
    }
    pub fn knobs(&self, f: impl FnOnce(&mut Knobs)) {
        f(&mut self.st.knobs.lock().unwrap());
    }
    pub fn last_access(&self) -> String {
        self.st.issued.lock().unwrap().last_access.clone()
    }
    pub fn last_refresh(&self) -> String {
        self.st.issued.lock().unwrap().last_refresh.clone()
    }
    /// The server forgets every access token (as if they expired there), refresh tokens stay valid.
    pub fn forget_access_tokens(&self) {
        self.st.issued.lock().unwrap().access.clear();
    }
    pub fn refresh_requests(&self) -> usize {
        self.st.refresh_requests.load(Ordering::SeqCst)
    }
    pub fn authorizes(&self) -> Vec<HashMap<String, String>> {
        self.st.log.lock().unwrap().authorizes.clone()
    }
    pub fn registrations(&self) -> Vec<Value> {
        self.st.log.lock().unwrap().registrations.clone()
    }
    pub fn token_requests(&self) -> Vec<HashMap<String, String>> {
        self.st.log.lock().unwrap().token_requests.clone()
    }
    pub fn revocations(&self) -> Vec<HashMap<String, String>> {
        self.st.log.lock().unwrap().revocations.clone()
    }
    pub fn paths(&self) -> Vec<String> {
        self.st.log.lock().unwrap().paths.clone()
    }
    pub fn mcp_calls(&self, method: &str) -> usize {
        self.st.log.lock().unwrap().mcp.iter().filter(|r| r.rpc_method().as_deref() == Some(method)).count()
    }
}

fn query_of(path: &str) -> HashMap<String, String> {
    url::Url::parse(&format!("http://x{path}"))
        .map(|u| u.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect())
        .unwrap_or_default()
}

fn form_of(body: &str) -> HashMap<String, String> {
    url::form_urlencoded::parse(body.as_bytes()).map(|(k, v)| (k.into_owned(), v.into_owned())).collect()
}

async fn json_reply(sock: &mut tokio::net::TcpStream, status: &str, v: &Value) {
    respond(sock, status, &[("Content-Type", "application/json".into())], &v.to_string(), false).await;
}

fn issue_access(st: &Inner, scope: &str, expires_in: i64, big: bool) -> String {
    let mut i = st.issued.lock().unwrap();
    i.n += 1;
    let mut t = format!("at-{}-{}", i.n, b64url(&Sha256::digest(format!("a{}", i.n).as_bytes())));
    if big {
        t.push_str(&"x".repeat(6000));
    }
    i.access.insert(t.clone(), (now_ms() + expires_in * 1000, scope.to_string()));
    i.last_access = t.clone();
    t
}

fn issue_refresh(st: &Inner, client_id: &str, scope: &str) -> String {
    let mut i = st.issued.lock().unwrap();
    i.n += 1;
    let t = format!("rt-{}-{}", i.n, b64url(&Sha256::digest(format!("r{}", i.n).as_bytes())));
    i.refresh.insert(t.clone(), (client_id.to_string(), scope.to_string()));
    i.last_refresh = t.clone();
    t
}

fn token_json(access: &str, refresh: Option<&str>, scope: &str, expires_in: i64) -> Value {
    let mut v = json!({"access_token": access, "token_type": "Bearer", "expires_in": expires_in, "scope": scope});
    if let Some(r) = refresh {
        v["refresh_token"] = json!(r);
    }
    v
}

async fn handle(mut sock: tokio::net::TcpStream, st: Arc<Inner>) {
    let Some(req) = read_request(&mut sock).await else { return };
    let knobs = st.knobs.lock().unwrap().clone();
    let port = st.port;
    let origin = format!("http://127.0.0.1:{port}");
    let (path, query_str) = req.path.split_once('?').map(|(a, b)| (a.to_string(), format!("?{b}"))).unwrap_or((req.path.clone(), String::new()));
    st.log.lock().unwrap().paths.push(format!("{} {}", req.method, path));

    match (req.method.as_str(), path.as_str()) {
        ("GET", p) if p.starts_with("/.well-known/oauth-protected-resource") => {
            let servers = knobs.prm_authorization_servers.clone().unwrap_or_else(|| vec![origin.clone()]);
            json_reply(&mut sock, "200 OK", &json!({"resource": format!("{origin}/mcp"), "authorization_servers": servers})).await;
        }
        ("GET", "/.well-known/oauth-authorization-server") => {
            let mut m = json!({
                "issuer": knobs.meta_issuer.clone().unwrap_or_else(|| origin.clone()),
                "authorization_endpoint": format!("{origin}/authorize"),
                "token_endpoint": knobs.token_endpoint_override.clone().unwrap_or_else(|| format!("{origin}/token")),
                "revocation_endpoint": format!("{origin}/revoke"),
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
                "authorization_response_iss_parameter_supported": knobs.iss_promised,
                "client_id_metadata_document_supported": knobs.cimd_flag,
            });
            if !knobs.omit_s256 {
                m["code_challenge_methods_supported"] = json!(["plain", "S256"]);
            } else {
                m["code_challenge_methods_supported"] = json!(["plain"]);
            }
            if knobs.registration {
                m["registration_endpoint"] = json!(knobs.registration_endpoint_override.clone().unwrap_or_else(|| format!("{origin}/register")));
            }
            json_reply(&mut sock, "200 OK", &m).await;
        }
        ("POST", "/register") => {
            let body: Value = serde_json::from_str(&req.body).unwrap_or(Value::Null);
            st.log.lock().unwrap().registrations.push(body.clone());
            if knobs.registration_rejects_redirect {
                json_reply(&mut sock, "400 Bad Request", &json!({"error": "invalid_redirect_uri"})).await;
                return;
            }
            let uri = body["redirect_uris"][0].as_str().unwrap_or("").to_string();
            let id = format!("dcr-{}", st.registered.lock().unwrap().len() + 1);
            st.registered.lock().unwrap().insert(id.clone(), uri);
            json_reply(&mut sock, "201 Created", &json!({"client_id": id, "token_endpoint_auth_method": "none"})).await;
        }
        ("GET", "/authorize") => {
            let q = query_of(&format!("/authorize{query_str}"));
            st.log.lock().unwrap().authorizes.push(q.clone());
            let get = |k: &str| q.get(k).cloned().unwrap_or_default();
            let redirect_uri = get("redirect_uri");
            let mut problems: Vec<&str> = vec![];
            if get("response_type") != "code" {
                problems.push("response_type");
            }
            if get("code_challenge_method") != "S256" || get("code_challenge").is_empty() {
                problems.push("pkce");
            }
            if get("state").is_empty() || get("resource").is_empty() {
                problems.push("state or resource");
            }
            if let Some(reg) = st.registered.lock().unwrap().get(&get("client_id")) {
                if *reg != redirect_uri {
                    problems.push("redirect_uri does not match the registration");
                }
            }
            if !redirect_uri.starts_with("http://127.0.0.1:") {
                problems.push("redirect_uri is not loopback");
            }
            if !problems.is_empty() {
                respond(&mut sock, "400 Bad Request", &[], &format!("bad authorization request: {}", problems.join(", ")), false).await;
                return;
            }
            let base = knobs.redirect_elsewhere.clone().unwrap_or(redirect_uri.clone());
            let state = knobs.state_override.clone().unwrap_or_else(|| get("state"));
            let mut location = if let Some(err) = &knobs.authorize_error {
                format!("{base}?error={err}&state={state}")
            } else {
                let code = format!("code-{}", st.codes.lock().unwrap().len() + 1);
                st.codes.lock().unwrap().insert(
                    code.clone(),
                    Code {
                        challenge: get("code_challenge"),
                        redirect_uri,
                        client_id: get("client_id"),
                        resource: get("resource"),
                        scope: get("scope"),
                        created_ms: now_ms(),
                        used: false,
                    },
                );
                format!("{base}?code={code}&state={state}")
            };
            if let Some(iss) = &knobs.iss_in_response {
                location.push_str(&format!("&iss={iss}"));
            }
            respond(&mut sock, "302 Found", &[("Location", location)], "", false).await;
        }
        ("POST", "/token") => {
            let mut form = form_of(&req.body);
            if let Some(basic) = req.headers.get("authorization") {
                form.insert("_basic".into(), basic.clone());
            }
            let grant = form.get("grant_type").cloned().unwrap_or_default();
            let client_id = form.get("client_id").cloned().unwrap_or_default();
            // a confidential client must present its secret
            if let Some(secret) = knobs.client_secrets.get(&client_id) {
                if form.get("client_secret") != Some(secret) {
                    st.log.lock().unwrap().token_requests.push(form);
                    json_reply(&mut sock, "401 Unauthorized", &json!({"error": "invalid_client"})).await;
                    return;
                }
            }
            if grant == "authorization_code" {
                let verifier = form.get("code_verifier").cloned().unwrap_or_default();
                let verdict = {
                    let mut codes = st.codes.lock().unwrap();
                    match codes.get_mut(form.get("code").map(String::as_str).unwrap_or("")) {
                        None => Err("unknown code"),
                        Some(c) if c.used => Err("code already used"),
                        Some(c) if now_ms() - c.created_ms > knobs.code_ttl_ms => Err("code expired"),
                        Some(c) if form.get("redirect_uri") != Some(&c.redirect_uri) => Err("redirect_uri differs"),
                        Some(c) if c.client_id != client_id => Err("client differs"),
                        Some(c) if b64url(&Sha256::digest(verifier.as_bytes())) != c.challenge => Err("pkce verifier does not match"),
                        Some(c) if form.get("resource") != Some(&c.resource) => Err("resource differs"),
                        Some(c) => {
                            c.used = true;
                            Ok(c.scope.clone())
                        }
                    }
                };
                let mut logged = form.clone();
                logged.insert("_verdict".into(), format!("{verdict:?}"));
                st.log.lock().unwrap().token_requests.push(logged);
                match verdict {
                    Err(why) => json_reply(&mut sock, "400 Bad Request", &json!({"error": "invalid_grant", "error_description": why})).await,
                    Ok(scope) => {
                        let scope = if scope.is_empty() { "default".to_string() } else { scope };
                        let at = issue_access(&st, &scope, knobs.expires_in, knobs.big_token);
                        let rt = issue_refresh(&st, &client_id, &scope);
                        json_reply(&mut sock, "200 OK", &token_json(&at, Some(&rt), &scope, knobs.expires_in)).await;
                    }
                }
            } else if grant == "refresh_token" {
                st.refresh_requests.fetch_add(1, Ordering::SeqCst);
                st.log.lock().unwrap().token_requests.push(form.clone());
                let rt = form.get("refresh_token").cloned().unwrap_or_default();
                let found = st.issued.lock().unwrap().refresh.get(&rt).cloned();
                match found {
                    Some((cid, scope)) if !knobs.refresh_revoked && cid == client_id => {
                        let at = issue_access(&st, &scope, knobs.expires_in, knobs.big_token);
                        let new_rt = if knobs.rotate_refresh {
                            st.issued.lock().unwrap().refresh.remove(&rt);
                            Some(issue_refresh(&st, &client_id, &scope))
                        } else {
                            None
                        };
                        json_reply(&mut sock, "200 OK", &token_json(&at, new_rt.as_deref(), &scope, knobs.expires_in)).await;
                    }
                    _ => json_reply(&mut sock, "400 Bad Request", &json!({"error": "invalid_grant"})).await,
                }
            } else {
                json_reply(&mut sock, "400 Bad Request", &json!({"error": "unsupported_grant_type"})).await;
            }
        }
        ("POST", "/revoke") => {
            let form = form_of(&req.body);
            if let Some(t) = form.get("token") {
                let mut i = st.issued.lock().unwrap();
                i.refresh.remove(t);
                i.access.remove(t);
            }
            st.log.lock().unwrap().revocations.push(form);
            respond(&mut sock, "200 OK", &[], "", false).await;
        }
        (_, "/mcp") => {
            st.log.lock().unwrap().mcp.push(req.clone());
            let challenge = {
                let mut c = String::from("Bearer realm=\"mcp\"");
                if knobs.header_hint {
                    c.push_str(&format!(", resource_metadata=\"{origin}/.well-known/oauth-protected-resource\""));
                }
                if let Some(s) = &knobs.challenge_scope {
                    c.push_str(&format!(", scope=\"{s}\""));
                }
                c
            };
            let bearer = req.headers.get("authorization").and_then(|a| a.strip_prefix("Bearer ")).map(str::to_string);
            let scope = bearer.as_deref().and_then(|b| {
                st.issued.lock().unwrap().access.get(b).and_then(|(exp, sc)| (*exp > now_ms()).then(|| sc.clone()))
            });
            if !knobs.mcp_open && (knobs.mcp_always_401 || scope.is_none()) {
                respond(&mut sock, "401 Unauthorized", &[("WWW-Authenticate", challenge)], "", false).await;
                return;
            }
            if req.method == "DELETE" {
                respond(&mut sock, "200 OK", &[], "", false).await;
                return;
            }
            let msg: Value = serde_json::from_str(&req.body).unwrap_or(Value::Null);
            let method = msg["method"].as_str().unwrap_or("").to_string();
            let id = msg.get("id").cloned();
            if id.is_none() || method.is_empty() {
                respond(&mut sock, "202 Accepted", &[], "", false).await;
                return;
            }
            if method == "tools/call" {
                if let Some(need) = &knobs.call_needs_scope {
                    let have = scope.clone().unwrap_or_default();
                    if !have.split_whitespace().any(|w| w == need) {
                        let h = format!("Bearer error=\"insufficient_scope\", scope=\"{need}\"");
                        respond(&mut sock, "403 Forbidden", &[("WWW-Authenticate", h)], "", false).await;
                        return;
                    }
                }
            }
            let result = match method.as_str() {
                "initialize" => {
                    json!({"protocolVersion": msg["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"), "capabilities": {"tools": {}}, "serverInfo": {"name": "mock-auth", "version": "1.0"}})
                }
                "tools/list" => json!({"tools": [{"name": "echo", "description": "Echo the text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}}]}),
                "tools/call" => json!({"content": [{"type": "text", "text": msg["params"]["arguments"]["text"].as_str().unwrap_or("")}]}),
                _ => json!({}),
            };
            json_reply(&mut sock, "200 OK", &json!({"jsonrpc": "2.0", "id": id, "result": result})).await;
        }
        _ => respond(&mut sock, "404 Not Found", &[], "", false).await,
    }
}

pub async fn start(configure: impl FnOnce(&mut Knobs)) -> MockAuth {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut knobs = Knobs::default();
    configure(&mut knobs);
    let st = Arc::new(Inner {
        knobs: Mutex::new(knobs),
        log: Mutex::new(Log::default()),
        codes: Mutex::new(HashMap::new()),
        issued: Mutex::new(Issued::default()),
        registered: Mutex::new(HashMap::new()),
        refresh_requests: AtomicUsize::new(0),
        port: addr.port(),
    });
    let s = Arc::clone(&st);
    tokio::spawn(async move {
        loop {
            let Ok((sock, _)) = listener.accept().await else { break };
            tokio::spawn(handle(sock, Arc::clone(&s)));
        }
    });
    MockAuth { addr, st }
}

/// What a user's browser does with the address Dara opens: GET it, and follow the redirect to Dara's loopback address.
pub async fn browse(url: &str) -> Option<String> {
    let c = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().ok()?;
    let first = c.get(url).send().await.ok()?;
    let loc = first.headers().get("location")?.to_str().ok()?.to_string();
    let second = c.get(&loc).send().await.ok()?;
    Some(second.text().await.unwrap_or_default())
}

/// The `open_url` of a user who finishes in the browser.
pub fn browser() -> impl Fn(&str) + Send + Sync {
    |u: &str| {
        let u = u.to_string();
        tokio::spawn(async move {
            let _ = browse(&u).await;
        });
    }
}

/// The `open_url` of a user who never comes back.
pub fn no_browser() -> impl Fn(&str) + Send + Sync {
    |_: &str| {}
}
