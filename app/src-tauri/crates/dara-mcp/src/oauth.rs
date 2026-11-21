//! OAuth sign-in for Streamable-HTTP MCP servers.
//!
//! Follows the MCP authorization spec (2025-06-18) and, for the parts it points to: protected resource metadata (RFC 9728), authorization
//! server metadata (RFC 8414 / OpenID discovery), dynamic client registration (RFC 7591), PKCE (RFC 7636), resource indicators (RFC 8707),
//! native apps with a loopback redirect (RFC 8252), the `iss` check (RFC 9207) and revocation (RFC 7009).
//!
//! Tokens only ever live in the [`SecretStore`] (in pieces, see `secrets::put_chunked`). Nothing here logs or returns a token to the UI.
//!
//! Every address a server hands us (resource metadata, authorization server, token, registration, revocation) is untrusted: it must be
//! https, must not resolve to a private, link-local or unique-local address, redirects are never followed, and the connection is made
//! to the addresses that were checked (so a second DNS answer cannot change where it goes). Loopback is allowed only when the MCP server
//! itself is on this computer.

use crate::error::McpError;
use crate::secrets::{self, SecretStore};
use crate::store::now_ms;
use crate::transport::{check_http_url, is_loopback_host, TokenProvider};
use async_trait::async_trait;
use futures_util::StreamExt;
use rand::RngCore;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use reqwest::{redirect, Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Notify;
use url::Url;

/// How long the user has to finish in the browser.
pub const DEFAULT_SIGN_IN_TIMEOUT: Duration = Duration::from_secs(300);
/// A token that expires within this long is refreshed before it is used.
const REFRESH_MARGIN_MS: i64 = 60_000;
const MAX_META_BYTES: usize = 1024 * 1024;
/// A taken or refused loopback port is retried this many times with another port.
const PORT_TRIES: usize = 3;

// ---- the network rules ----

/// What a discovered address may point at.
#[derive(Debug, Clone, Copy)]
pub struct Net {
    /// true only when the MCP server is itself on this computer (a local server the user added)
    pub allow_loopback: bool,
}

impl Net {
    pub fn for_server(mcp: &Url) -> Self {
        Net { allow_loopback: is_loopback_host(mcp) }
    }
    pub fn for_resource(resource: &str) -> Self {
        Net { allow_loopback: Url::parse(resource).map(|u| is_loopback_host(&u)).unwrap_or(false) }
    }
}

/// Addresses a hostile server must not be able to send Dara to: private, link-local, unique-local, unspecified, carrier-grade NAT,
/// multicast, and loopback unless it is allowed.
pub fn is_forbidden_ip(ip: IpAddr, allow_loopback: bool) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            (v4.is_loopback() && !allow_loopback)
                || (!v4.is_loopback()
                    && (v4.is_private()
                        || v4.is_link_local() // 169.254.0.0/16, which includes the cloud metadata address
                        || v4.is_unspecified()
                        || v4.is_broadcast()
                        || v4.is_multicast()
                        || o[0] == 0
                        || (o[0] == 100 && (o[1] & 0xC0) == 64))) // 100.64.0.0/10
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_forbidden_ip(IpAddr::V4(v4), allow_loopback);
            }
            let s = v6.segments();
            (v6.is_loopback() && !allow_loopback)
                || (!v6.is_loopback()
                    && (v6.is_unspecified()
                        || v6.is_multicast()
                        || (s[0] & 0xfe00) == 0xfc00 // fc00::/7
                        || (s[0] & 0xffc0) == 0xfe80)) // fe80::/10
        }
    }
}

fn refuse_address(url: &Url) -> McpError {
    McpError::Invalid(format!(
        "the address {} points to a private or internal computer, so it was not used",
        url.host_str().unwrap_or("?")
    ))
}

/// Checks an address a server gave us and returns a client that connects only to the addresses that were checked.
pub async fn guarded_client(url: &Url, net: Net) -> Result<Client, McpError> {
    check_http_url(url.as_str()).map_err(|e| McpError::Invalid(format!("an address the server gave is not acceptable: {e}")))?;
    let mut b = Client::builder()
        .redirect(redirect::Policy::none()) // a redirect could send a code, a token or a secret somewhere else
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(20))
        .user_agent(concat!("Dara/", env!("CARGO_PKG_VERSION")));
    match url.host() {
        Some(url::Host::Ipv4(ip)) => {
            if is_forbidden_ip(IpAddr::V4(ip), net.allow_loopback) {
                return Err(refuse_address(url));
            }
        }
        Some(url::Host::Ipv6(ip)) => {
            if is_forbidden_ip(IpAddr::V6(ip), net.allow_loopback) {
                return Err(refuse_address(url));
            }
        }
        Some(url::Host::Domain(d)) => {
            let port = url.port_or_known_default().unwrap_or(443);
            let addrs: Vec<SocketAddr> = tokio::net::lookup_host((d, port))
                .await
                .map_err(|e| McpError::Transport(format!("could not find {d}: {e}")))?
                .collect();
            if addrs.is_empty() {
                return Err(McpError::Transport(format!("could not find {d}")));
            }
            if addrs.iter().any(|a| is_forbidden_ip(a.ip(), net.allow_loopback)) {
                return Err(refuse_address(url));
            }
            b = b.resolve_to_addrs(d, &addrs);
        }
        None => return Err(McpError::Invalid("an address the server gave has no host".into())),
    }
    b.build().map_err(|e| McpError::Transport(e.to_string()))
}

/// An address that is an IP literal is judged at once; a name is judged when it is resolved, at the moment of connecting.
fn literal_ok(url: &Url, net: Net) -> bool {
    match url.host() {
        Some(url::Host::Ipv4(ip)) => !is_forbidden_ip(IpAddr::V4(ip), net.allow_loopback),
        Some(url::Host::Ipv6(ip)) => !is_forbidden_ip(IpAddr::V6(ip), net.allow_loopback),
        _ => true,
    }
}

async fn read_capped(resp: Response) -> Result<Vec<u8>, McpError> {
    let mut stream = resp.bytes_stream();
    let mut out = Vec::new();
    while let Some(chunk) = stream.next().await {
        out.extend_from_slice(&chunk.map_err(|e| McpError::Transport(format!("could not read the answer: {e}")))?);
        if out.len() > MAX_META_BYTES {
            return Err(McpError::Protocol("a sign-in answer is larger than 1 MB".into()));
        }
    }
    Ok(out)
}

/// Plain text that is safe to show: no control characters, cut short.
fn clean(s: &str) -> String {
    s.chars().filter(|c| !c.is_control()).take(200).collect()
}

fn endpoint(raw: &str, what: &str) -> Result<Url, McpError> {
    check_http_url(raw).map_err(|e| McpError::Invalid(format!("the server's {what} is not acceptable: {e}")))
}

/// Base64url without padding (RFC 4648 section 5).
pub fn b64url(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        if c.len() > 1 {
            out.push(A[(n >> 6) as usize & 63] as char);
        }
        if c.len() > 2 {
            out.push(A[n as usize & 63] as char);
        }
    }
    out
}

fn random_b64url(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    rand::rngs::OsRng.fill_bytes(&mut buf);
    b64url(&buf)
}

/// A PKCE verifier (64 characters) and its S256 challenge.
pub fn pkce_pair() -> (String, String) {
    let verifier = random_b64url(48);
    let challenge = b64url(&Sha256::digest(verifier.as_bytes()));
    (verifier, challenge)
}

// ---- discovery ----

#[derive(Debug, Clone)]
pub struct AuthServer {
    pub issuer: String,
    pub authorization_endpoint: Url,
    pub token_endpoint: Url,
    pub registration_endpoint: Option<Url>,
    pub revocation_endpoint: Option<Url>,
    pub token_auth_methods: Vec<String>,
    /// The server accepts a client ID that is the https address of a client metadata document. Dara cannot host one for a desktop
    /// app, so this is only detected (it makes the "needs a client ID" message more helpful).
    pub cimd_supported: bool,
    /// The server says every authorization response carries `iss` (RFC 9207), so a missing one is refused.
    pub iss_required: bool,
}

#[derive(Debug, Clone)]
pub struct Discovered {
    /// The canonical address of the MCP server: the `resource` of every authorization and token request.
    pub resource: String,
    pub server: AuthServer,
}

async fn get_json(url: &Url, net: Net) -> Result<Value, McpError> {
    let client = guarded_client(url, net).await?;
    let resp = client
        .get(url.clone())
        .header(ACCEPT, "application/json")
        .send()
        .await
        .map_err(|e| McpError::Transport(format!("could not reach {}: {e}", url.host_str().unwrap_or("the server"))))?;
    if !resp.status().is_success() {
        return Err(McpError::Transport(format!("{} answered HTTP {}", url.path(), resp.status().as_u16())));
    }
    let body = read_capped(resp).await?;
    serde_json::from_slice(&body).map_err(|e| McpError::Protocol(format!("{} is not JSON: {e}", url.path())))
}

fn origin_of(u: &Url) -> String {
    let mut o = format!("{}://{}", u.scheme(), u.host_str().unwrap_or(""));
    if let Some(p) = u.port() {
        o.push_str(&format!(":{p}"));
    }
    o
}

/// The address used as `resource`: the MCP server's own address without query or fragment.
fn canonical_resource(u: &Url) -> String {
    let mut c = u.clone();
    c.set_query(None);
    c.set_fragment(None);
    c.to_string()
}

fn same_resource(prm: &Url, mcp: &Url) -> bool {
    origin_of(prm) == origin_of(mcp) && mcp.path().trim_end_matches('/').starts_with(prm.path().trim_end_matches('/'))
}

/// Tries each candidate; a policy refusal (a private address, not https) ends the search with that error, anything else moves on.
async fn first_json(candidates: Vec<Url>, net: Net) -> Result<Option<Value>, McpError> {
    for c in candidates {
        match get_json(&c, net).await {
            Ok(v) if v.is_object() => return Ok(Some(v)),
            Err(e @ McpError::Invalid(_)) => return Err(e),
            _ => {}
        }
    }
    Ok(None)
}

/// Fetches the protected resource metadata: from the 401's `resource_metadata` when given, else the well-known addresses.
async fn protected_resource(mcp: &Url, hint: Option<&str>, net: Net) -> Result<Option<Value>, McpError> {
    let mut candidates: Vec<Url> = Vec::new();
    if let Some(h) = hint {
        candidates.push(endpoint(h, "resource metadata address")?);
    }
    let origin = origin_of(mcp);
    let path = mcp.path().trim_end_matches('/');
    if !path.is_empty() {
        if let Ok(u) = Url::parse(&format!("{origin}/.well-known/oauth-protected-resource{path}")) {
            candidates.push(u);
        }
    }
    if let Ok(u) = Url::parse(&format!("{origin}/.well-known/oauth-protected-resource")) {
        candidates.push(u);
    }
    first_json(candidates, net).await
}

async fn auth_server_metadata(issuer: &Url, net: Net) -> Result<AuthServer, McpError> {
    let origin = origin_of(issuer);
    let path = issuer.path().trim_end_matches('/');
    let mut raw: Vec<String> = Vec::new();
    if path.is_empty() {
        raw.push(format!("{origin}/.well-known/oauth-authorization-server"));
        raw.push(format!("{origin}/.well-known/openid-configuration"));
    } else {
        raw.push(format!("{origin}/.well-known/oauth-authorization-server{path}"));
        raw.push(format!("{origin}/.well-known/openid-configuration{path}"));
        raw.push(format!("{origin}{path}/.well-known/openid-configuration"));
    }
    let candidates: Vec<Url> = raw.iter().filter_map(|c| Url::parse(c).ok()).collect();
    let m = first_json(candidates, net).await?.ok_or_else(|| {
        McpError::Transport(format!("could not find how to sign in at {origin} (no authorization server metadata)"))
    })?;

    // the issuer must be the server we asked (a defence against a server pointing at someone else's metadata); a template such as
    // Microsoft's {tenantid} is the one known exception
    let claimed = m["issuer"].as_str().unwrap_or("");
    if claimed.trim_end_matches('/') != issuer.as_str().trim_end_matches('/') && !claimed.contains('{') {
        return Err(McpError::Invalid(format!(
            "the sign-in server's metadata names a different issuer ({}), so it was not trusted",
            clean(claimed)
        )));
    }
    let methods = m["code_challenge_methods_supported"].as_array();
    if !methods.map(|a| a.iter().any(|x| x == "S256")).unwrap_or(false) {
        return Err(McpError::Invalid(
            "the sign-in server does not support PKCE (S256), so Dara will not sign in to it".into(),
        ));
    }
    let need = |k: &str| -> Result<Url, McpError> {
        let u = endpoint(m[k].as_str().ok_or_else(|| McpError::Protocol(format!("the sign-in server has no {k}")))?, k)?;
        literal_ok(&u, net).then_some(u.clone()).ok_or_else(|| refuse_address(&u))
    };
    // an optional address that is not acceptable is simply not used
    let optional = |k: &str| m[k].as_str().and_then(|s| check_http_url(s).ok()).filter(|u| literal_ok(u, net));
    Ok(AuthServer {
        issuer: claimed.to_string(),
        authorization_endpoint: need("authorization_endpoint")?,
        token_endpoint: need("token_endpoint")?,
        registration_endpoint: optional("registration_endpoint"),
        revocation_endpoint: optional("revocation_endpoint"),
        token_auth_methods: m["token_endpoint_auth_methods_supported"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        cimd_supported: m["client_id_metadata_document_supported"].as_bool().unwrap_or(false),
        iss_required: m["authorization_response_iss_parameter_supported"].as_bool().unwrap_or(false),
    })
}

/// Finds how to sign in to an MCP server: its protected resource metadata, then its authorization server. A server with no resource
/// metadata is treated as its own authorization server (the older 2025-03-26 behaviour).
pub async fn discover(mcp_url: &Url, resource_metadata: Option<&str>) -> Result<Discovered, McpError> {
    let net = Net::for_server(mcp_url);
    let prm = protected_resource(mcp_url, resource_metadata, net).await?;
    let (resource, issuer) = match &prm {
        Some(v) => {
            let resource = match v["resource"].as_str() {
                Some(r) => {
                    let u = endpoint(r, "resource address")?;
                    if !same_resource(&u, mcp_url) {
                        return Err(McpError::Invalid(
                            "the server's resource metadata names a different server, so it was not trusted".into(),
                        ));
                    }
                    r.to_string()
                }
                None => canonical_resource(mcp_url),
            };
            let first = v["authorization_servers"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("the server's resource metadata lists no authorization server".into()))?;
            (resource, endpoint(first, "authorization server address")?)
        }
        None => (canonical_resource(mcp_url), Url::parse(&origin_of(mcp_url)).map_err(|e| McpError::Invalid(e.to_string()))?),
    };
    let server = auth_server_metadata(&issuer, net).await?;
    Ok(Discovered { resource, server })
}

// ---- tokens ----

/// What is kept for a signed-in connector (in the secret store, as one value in pieces).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredTokens {
    pub access_token: String,
    pub refresh_token: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub scope: Option<String>,
    pub client_id: String,
    pub token_endpoint: String,
    pub revocation_endpoint: Option<String>,
    pub resource: String,
    /// How the client proves itself at the token endpoint: "none" (public), "post" or "basic" (with a secret).
    pub auth_method: String,
}

#[derive(Debug)]
pub enum TokenError {
    /// The server said no (invalid_grant, invalid_client, ...): trying again will not help.
    Rejected { error: String, description: String },
    /// Could not reach it, or it failed (5xx, 429), or a policy refused the address: the tokens are still the user's.
    Transient(String),
}

impl TokenError {
    fn into_mcp(self) -> McpError {
        match self {
            TokenError::Rejected { error, description } => McpError::Invalid(format!(
                "the server refused the sign-in: {error}{}",
                if description.is_empty() { String::new() } else { format!(" ({description})") }
            )),
            TokenError::Transient(m) => McpError::Transport(m),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClientAuth {
    pub client_id: String,
    pub secret: Option<String>,
    pub method: String,
}

impl ClientAuth {
    /// "none" without a secret; with one, `client_secret_post` unless the server only offers `client_secret_basic`.
    pub fn new(client_id: &str, secret: Option<String>, supported: &[String]) -> Self {
        let method = match &secret {
            None => "none",
            Some(_) if supported.is_empty() || supported.iter().any(|m| m == "client_secret_post") => "post",
            Some(_) if supported.iter().any(|m| m == "client_secret_basic") => "basic",
            Some(_) => "post",
        };
        Self { client_id: client_id.to_string(), secret, method: method.to_string() }
    }
}

async fn token_request(
    endpoint: &Url,
    mut form: Vec<(&'static str, String)>,
    auth: &ClientAuth,
    net: Net,
) -> Result<Value, TokenError> {
    let client = guarded_client(endpoint, net).await.map_err(|e| match e {
        // a refused address is the server's fault, not a reason to forget the tokens
        McpError::Invalid(m) => TokenError::Transient(m),
        other => TokenError::Transient(other.to_string()),
    })?;
    let mut rb = client.post(endpoint.clone()).header(ACCEPT, "application/json");
    match (auth.method.as_str(), &auth.secret) {
        ("basic", Some(secret)) => rb = rb.basic_auth(&auth.client_id, Some(secret)),
        ("post", Some(secret)) => {
            form.push(("client_id", auth.client_id.clone()));
            form.push(("client_secret", secret.clone()));
        }
        _ => form.push(("client_id", auth.client_id.clone())),
    }
    // the body is built first: the serializer must not live across the await
    let body: String = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(form.iter().map(|(k, v)| (*k, v.as_str())))
        .finish();
    let resp = rb
        .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|e| TokenError::Transient(format!("could not reach the sign-in server: {e}")))?;
    let status = resp.status();
    let body = read_capped(resp).await.map_err(|e| TokenError::Transient(e.to_string()))?;
    let json: Option<Value> = serde_json::from_slice(&body).ok();
    if status.is_success() {
        return json.ok_or_else(|| TokenError::Transient("the sign-in server's answer is not JSON".into()));
    }
    let error = json.as_ref().and_then(|j| j["error"].as_str()).map(clean).unwrap_or_default();
    let description = json.as_ref().and_then(|j| j["error_description"].as_str()).map(clean).unwrap_or_default();
    let transient_error = matches!(error.as_str(), "temporarily_unavailable" | "server_error");
    if status.is_server_error() || status.as_u16() == 429 || transient_error {
        return Err(TokenError::Transient(format!("the sign-in server answered HTTP {}", status.as_u16())));
    }
    Err(TokenError::Rejected { error: if error.is_empty() { format!("HTTP {}", status.as_u16()) } else { error }, description })
}

fn tokens_from_response(
    v: &Value,
    previous: Option<&StoredTokens>,
    client_id: &str,
    token_endpoint: &Url,
    revocation: Option<&Url>,
    resource: &str,
    auth_method: &str,
) -> Result<StoredTokens, McpError> {
    let access = v["access_token"].as_str().filter(|s| !s.is_empty()).ok_or_else(|| McpError::Protocol("the sign-in server sent no access token".into()))?;
    if let Some(t) = v["token_type"].as_str() {
        if !t.eq_ignore_ascii_case("bearer") {
            return Err(McpError::Protocol(format!("the sign-in server sent a {} token, not a bearer token", clean(t))));
        }
    }
    let expires_in = v["expires_in"].as_i64().or_else(|| v["expires_in"].as_str().and_then(|s| s.parse().ok()));
    Ok(StoredTokens {
        access_token: access.to_string(),
        // a refresh may or may not rotate the refresh token: keep the old one when no new one comes
        refresh_token: v["refresh_token"].as_str().map(str::to_string).or_else(|| previous.and_then(|p| p.refresh_token.clone())),
        expires_at_ms: expires_in.map(|s| now_ms() + s * 1000),
        scope: v["scope"].as_str().map(str::to_string).or_else(|| previous.and_then(|p| p.scope.clone())),
        client_id: client_id.to_string(),
        token_endpoint: token_endpoint.to_string(),
        revocation_endpoint: revocation.map(Url::to_string).or_else(|| previous.and_then(|p| p.revocation_endpoint.clone())),
        resource: resource.to_string(),
        auth_method: auth_method.to_string(),
    })
}

/// The bearer token of one connector: read from the secret store, refreshed when it is about to expire (one refresh at a time).
pub struct OAuthTokens {
    connector_id: String,
    secrets: Arc<dyn SecretStore>,
    cache: Mutex<Option<StoredTokens>>,
    lock: tokio::sync::Mutex<()>,
    refreshes: AtomicUsize,
}

enum Refreshed {
    Done(StoredTokens),
    /// the server rejected the refresh token: the tokens were cleared
    Rejected,
    /// a network or server failure: the tokens were kept
    Transient,
}

impl OAuthTokens {
    pub fn new(connector_id: &str, secrets: Arc<dyn SecretStore>) -> Result<Arc<Self>, McpError> {
        Ok(Arc::new(Self {
            connector_id: connector_id.to_string(),
            secrets,
            cache: Mutex::new(None),
            lock: tokio::sync::Mutex::new(()),
            refreshes: AtomicUsize::new(0),
        }))
    }

    /// How many refresh requests went to the server (for tests).
    pub fn refresh_count(&self) -> usize {
        self.refreshes.load(Ordering::SeqCst)
    }

    pub fn is_signed_in(&self) -> bool {
        secrets::get_chunked(self.secrets.as_ref(), &secrets::token_key(&self.connector_id)).ok().flatten().is_some()
    }

    /// Forgets what is cached, so the next use reads the secret store again.
    pub fn reload(&self) {
        *self.cache.lock().expect("tokens") = None;
    }

    fn current(&self) -> Option<StoredTokens> {
        if let Some(t) = self.cache.lock().expect("tokens").clone() {
            return Some(t);
        }
        let raw = secrets::get_chunked(self.secrets.as_ref(), &secrets::token_key(&self.connector_id)).ok().flatten()?;
        let t: StoredTokens = serde_json::from_str(&raw).ok()?;
        *self.cache.lock().expect("tokens") = Some(t.clone());
        Some(t)
    }

    /// The scope the stored tokens were granted, when there are tokens.
    pub fn granted_scope(&self) -> Option<String> {
        self.current().and_then(|t| t.scope)
    }

    /// Every secret value this connector holds (tokens, client secret). The audit log scrubs these from what it keeps.
    pub fn secret_values(&self) -> Vec<String> {
        let mut v = Vec::new();
        if let Some(t) = self.current() {
            v.push(t.access_token);
            v.extend(t.refresh_token);
        }
        v.extend(self.client_secret());
        v
    }

    /// Keeps the tokens (in the secret store, in pieces).
    pub fn store(&self, t: &StoredTokens) -> Result<(), McpError> {
        let raw = serde_json::to_string(t).map_err(|e| McpError::Storage(e.to_string()))?;
        secrets::put_chunked(self.secrets.as_ref(), &secrets::token_key(&self.connector_id), &raw)?;
        *self.cache.lock().expect("tokens") = Some(t.clone());
        Ok(())
    }

    /// Forgets the tokens.
    pub fn clear(&self) -> Result<(), McpError> {
        *self.cache.lock().expect("tokens") = None;
        secrets::delete_chunked(self.secrets.as_ref(), &secrets::token_key(&self.connector_id))
    }

    fn client_secret(&self) -> Option<String> {
        self.secrets.get(&secrets::client_secret_key(&self.connector_id)).ok().flatten()
    }

    fn expiring(t: &StoredTokens) -> bool {
        t.expires_at_ms.map(|e| e - now_ms() <= REFRESH_MARGIN_MS).unwrap_or(false)
    }

    /// Callers hold `self.lock`. Exactly one request goes out; a rejected refresh token ends the sign-in (no retry, no loop).
    async fn do_refresh(&self, t: &StoredTokens) -> Refreshed {
        let Some(rt) = &t.refresh_token else {
            let _ = self.clear();
            return Refreshed::Rejected;
        };
        let Ok(endpoint) = Url::parse(&t.token_endpoint) else {
            let _ = self.clear();
            return Refreshed::Rejected;
        };
        let auth = ClientAuth {
            client_id: t.client_id.clone(),
            secret: self.client_secret().filter(|_| t.auth_method != "none"),
            method: t.auth_method.clone(),
        };
        self.refreshes.fetch_add(1, Ordering::SeqCst);
        let form = vec![("grant_type", "refresh_token".to_string()), ("refresh_token", rt.clone()), ("resource", t.resource.clone())];
        match token_request(&endpoint, form, &auth, Net::for_resource(&t.resource)).await {
            Ok(v) => {
                let rev = t.revocation_endpoint.as_deref().and_then(|s| Url::parse(s).ok());
                match tokens_from_response(&v, Some(t), &t.client_id, &endpoint, rev.as_ref(), &t.resource, &t.auth_method) {
                    Ok(n) => {
                        // a rotated refresh token is saved BEFORE the new access token is used: the old one no longer works
                        if self.store(&n).is_err() {
                            *self.cache.lock().expect("tokens") = Some(n.clone()); // keep this session going; the store failed
                        }
                        Refreshed::Done(n)
                    }
                    Err(_) => Refreshed::Transient,
                }
            }
            Err(TokenError::Rejected { .. }) => {
                let _ = self.clear(); // the refresh token is no good: the user has to sign in again
                Refreshed::Rejected
            }
            Err(TokenError::Transient(_)) => Refreshed::Transient,
        }
    }

    /// Best effort (RFC 7009): tell the server the tokens are no longer wanted. Failure changes nothing.
    pub async fn revoke(&self) {
        let Some(t) = self.current() else { return };
        let Some(ep) = t.revocation_endpoint.as_deref().and_then(|s| Url::parse(s).ok()) else { return };
        let (token, hint) = match &t.refresh_token {
            Some(r) => (r.clone(), "refresh_token"),
            None => (t.access_token.clone(), "access_token"),
        };
        let auth = ClientAuth { client_id: t.client_id.clone(), secret: self.client_secret().filter(|_| t.auth_method != "none"), method: t.auth_method.clone() };
        let form = vec![("token", token), ("token_type_hint", hint.to_string())];
        let _ = tokio::time::timeout(Duration::from_secs(5), token_request(&ep, form, &auth, Net::for_resource(&t.resource))).await;
    }
}

#[async_trait]
impl TokenProvider for OAuthTokens {
    async fn bearer(&self) -> Option<String> {
        let t = self.current()?;
        if !Self::expiring(&t) {
            return Some(t.access_token);
        }
        let _one_at_a_time = self.lock.lock().await;
        let t = self.current()?; // another caller may have refreshed (or the tokens were cleared) while this one waited
        if !Self::expiring(&t) {
            return Some(t.access_token);
        }
        match self.do_refresh(&t).await {
            Refreshed::Done(n) => Some(n.access_token),
            Refreshed::Rejected => None,
            // could not reach the server: the old token may still be accepted
            Refreshed::Transient => Some(t.access_token),
        }
    }

    async fn refresh(&self) -> bool {
        let before = self.current().map(|t| t.access_token);
        let _one_at_a_time = self.lock.lock().await;
        let Some(t) = self.current() else { return false };
        if Some(&t.access_token) != before.as_ref() {
            return true; // someone else refreshed while this one waited
        }
        matches!(self.do_refresh(&t).await, Refreshed::Done(_))
    }
}

// ---- the sign-in itself ----

/// Where the client identity comes from. A client id the user typed (which may also be the https address of a client metadata
/// document they host) skips registration.
#[derive(Debug, Clone)]
pub enum ClientSource {
    /// The user gave a client id (and maybe a secret).
    Configured { client_id: String, secret: Option<String> },
    /// A client the server registered for us earlier, with the redirect address it was registered for.
    Registered { client_id: String, secret: Option<String>, redirect_uri: String },
    /// Nothing yet: register dynamically.
    None,
}

#[derive(Debug, Clone)]
pub struct NewRegistration {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub redirect_uri: String,
}

pub struct SignInParams {
    pub mcp_url: Url,
    /// from the 401 challenge
    pub resource_metadata: Option<String>,
    pub challenge_scope: Option<String>,
    pub configured_scope: Option<String>,
    pub client: ClientSource,
    pub timeout: Duration,
}

pub struct SignInOutcome {
    pub tokens: StoredTokens,
    /// set when this sign-in registered a new client
    pub registration: Option<NewRegistration>,
}

enum RegisterError {
    /// The server refused the redirect address (`invalid_redirect_uri`): another port may be accepted.
    Redirect,
    Other(McpError),
}

async fn register_client(
    ep: &Url,
    redirect_uri: &str,
    scope: Option<&str>,
    net: Net,
) -> Result<(String, Option<String>), RegisterError> {
    let mut body = json!({
        "client_name": "Dara",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
        "application_type": "native",
    });
    if let Some(s) = scope {
        body["scope"] = json!(s);
    }
    let client = guarded_client(ep, net).await.map_err(RegisterError::Other)?;
    let resp = client
        .post(ep.clone())
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await
        .map_err(|e| RegisterError::Other(McpError::Transport(format!("could not register with the sign-in server: {e}"))))?;
    let status = resp.status();
    let bytes = read_capped(resp).await.map_err(RegisterError::Other)?;
    let v: Option<Value> = serde_json::from_slice(&bytes).ok();
    if !status.is_success() {
        if v.as_ref().and_then(|j| j["error"].as_str()) == Some("invalid_redirect_uri") {
            return Err(RegisterError::Redirect);
        }
        return Err(RegisterError::Other(McpError::Invalid(format!(
            "the sign-in server would not register Dara (HTTP {})",
            status.as_u16()
        ))));
    }
    let v = v.ok_or_else(|| RegisterError::Other(McpError::Protocol("the registration answer is not JSON".into())))?;
    let id = v["client_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| RegisterError::Other(McpError::Protocol("the registration answer has no client_id".into())))?;
    Ok((id.to_string(), v["client_secret"].as_str().map(str::to_string)))
}

/// Binds the loopback listener: the port the client was registered with if it is free, else any free port (a few tries). Only 127.0.0.1.
pub async fn bind_loopback(preferred: Option<u16>) -> Result<TcpListener, McpError> {
    if let Some(p) = preferred {
        if let Ok(l) = TcpListener::bind(("127.0.0.1", p)).await {
            return Ok(l);
        }
    }
    let mut last = String::new();
    for _ in 0..PORT_TRIES {
        match TcpListener::bind(("127.0.0.1", 0)).await {
            Ok(l) => return Ok(l),
            Err(e) => last = e.to_string(),
        }
    }
    Err(McpError::Transport(format!("could not listen for the sign-in answer on this computer: {last}")))
}

const PAGE_OK: &str = "<!doctype html><meta charset=\"utf-8\"><title>Dara</title><body style=\"font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;padding:0 1rem\"><h1>You are signed in</h1><p>You can close this window and go back to Dara.</p></body>";
const PAGE_NO: &str = "<!doctype html><meta charset=\"utf-8\"><title>Dara</title><body style=\"font-family:system-ui,sans-serif;max-width:32rem;margin:4rem auto;padding:0 1rem\"><h1>Sign-in was not completed</h1><p>You can close this window and try again in Dara.</p></body>";

async fn answer(sock: &mut TcpStream, status: &str, body: &str) {
    let out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = sock.write_all(out.as_bytes()).await;
    let _ = sock.shutdown().await;
}

enum Callback {
    Code(String),
    Failed(String),
    /// not the answer (a wrong path, a wrong state, no code): keep waiting
    Ignore,
}

/// What the callback must say about who answered (RFC 9207).
struct IssuerCheck<'a> {
    issuer: &'a str,
    required: bool,
}

impl IssuerCheck<'_> {
    fn verify(&self, iss: Option<&str>) -> Result<(), String> {
        match iss {
            Some(i) => {
                // an issuer template (Microsoft's {tenantid}) cannot be compared
                if self.issuer.contains('{') || i.trim_end_matches('/') == self.issuer.trim_end_matches('/') {
                    Ok(())
                } else {
                    Err("the sign-in answer says it came from a different server than the one Dara asked, so it was rejected".into())
                }
            }
            None if self.required => Err("the sign-in answer did not say which server it came from, which this server promises to do, so it was rejected".into()),
            None => Ok(()),
        }
    }
}

fn redirect_refused(redirect_uri: &str) -> String {
    format!(
        "the sign-in server did not accept Dara's local redirect address ({redirect_uri}). It probably matches redirect addresses exactly and has no rule for a changing port (RFC 8252, section 7.3). Register Dara as a desktop client with the server and enter that client ID in the connector's sign-in settings."
    )
}

async fn read_callback(sock: &mut TcpStream, port: u16, state: &str, iss: &IssuerCheck<'_>) -> Callback {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 2048];
    let read = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match sock.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") || buf.len() > 8192 {
                        return;
                    }
                }
            }
        }
    })
    .await;
    if read.is_err() {
        return Callback::Ignore;
    }
    let head = String::from_utf8_lossy(&buf).to_string();
    let mut lines = head.split("\r\n");
    let first = lines.next().unwrap_or("");
    let mut parts = first.split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    // only our own page may call back: a request for another host name (DNS rebinding) is turned away
    let host_ok = lines
        .filter_map(|l| l.split_once(':'))
        .any(|(k, v)| k.eq_ignore_ascii_case("host") && v.trim() == format!("127.0.0.1:{port}"));
    if method != "GET" || !host_ok {
        answer(sock, "400 Bad Request", PAGE_NO).await;
        return Callback::Ignore;
    }
    let Ok(url) = Url::parse(&format!("http://127.0.0.1:{port}{target}")) else {
        answer(sock, "400 Bad Request", PAGE_NO).await;
        return Callback::Ignore;
    };
    if url.path() != "/callback" {
        answer(sock, "404 Not Found", PAGE_NO).await;
        return Callback::Ignore;
    }
    let q: std::collections::HashMap<String, String> = url.query_pairs().map(|(k, v)| (k.into_owned(), v.into_owned())).collect();
    if q.get("state").map(String::as_str) != Some(state) {
        answer(sock, "400 Bad Request", PAGE_NO).await;
        return Callback::Ignore;
    }
    if let Err(m) = iss.verify(q.get("iss").map(String::as_str)) {
        answer(sock, "400 Bad Request", PAGE_NO).await;
        return Callback::Failed(m);
    }
    if let Some(err) = q.get("error") {
        answer(sock, "200 OK", PAGE_NO).await;
        let desc = q.get("error_description").map(String::as_str).unwrap_or("");
        if err == "invalid_redirect_uri" || desc.to_ascii_lowercase().contains("redirect") {
            return Callback::Failed(redirect_refused(&format!("http://127.0.0.1:{port}/callback")));
        }
        let d = if desc.is_empty() { String::new() } else { format!(" ({})", clean(desc)) };
        return Callback::Failed(format!("the server refused the sign-in: {}{d}", clean(err)));
    }
    match q.get("code").filter(|c| !c.is_empty()) {
        Some(code) => {
            answer(sock, "200 OK", PAGE_OK).await;
            Callback::Code(code.clone())
        }
        None => {
            answer(sock, "400 Bad Request", PAGE_NO).await;
            Callback::Ignore
        }
    }
}

/// Waits for the one valid callback. A wrong state or path is turned away and the wait goes on, until the time is up or it is cancelled.
async fn wait_for_code(
    listener: TcpListener,
    state: &str,
    iss: &IssuerCheck<'_>,
    timeout: Duration,
    cancel: &Notify,
) -> Result<String, McpError> {
    let port = listener.local_addr().map_err(|e| McpError::Transport(e.to_string()))?.port();
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            _ = &mut deadline => {
                let wait = if timeout.as_secs() >= 60 { format!("{} minutes", timeout.as_secs() / 60) } else { format!("{} seconds", timeout.as_secs().max(1)) };
                return Err(McpError::Invalid(format!("sign-in timed out: nothing came back from the browser in {wait}")));
            }
            _ = cancel.notified() => return Err(McpError::Invalid("sign-in was cancelled".into())),
            accepted = listener.accept() => {
                let Ok((mut sock, _)) = accepted else { continue };
                match read_callback(&mut sock, port, state, iss).await {
                    Callback::Code(c) => return Ok(c),
                    Callback::Failed(m) => return Err(McpError::Invalid(m)),
                    Callback::Ignore => {}
                }
            }
        }
    }
}

/// Discovery, registration if needed, the browser, the callback and the token exchange. `open_url` is called once with the address to
/// open in the user's browser.
pub async fn run_sign_in(
    p: SignInParams,
    open_url: &(dyn Fn(&str) + Send + Sync),
    cancel: &Notify,
) -> Result<SignInOutcome, McpError> {
    let discovered = discover(&p.mcp_url, p.resource_metadata.as_deref()).await?;
    let server = &discovered.server;
    let net = Net::for_server(&p.mcp_url);

    let mut preferred_port = match &p.client {
        ClientSource::Registered { redirect_uri, .. } => Url::parse(redirect_uri).ok().and_then(|u| u.port()),
        _ => None,
    };
    let scope = p.challenge_scope.clone().filter(|s| !s.is_empty()).or_else(|| p.configured_scope.clone().filter(|s| !s.is_empty()));

    // a listener on a port, and who is signing in (a configured or an earlier client, else a new registration). A server that refuses
    // the redirect address of a new registration gets a few other ports before it is given up on.
    let mut tries = 0;
    let (listener, redirect_uri, client_id, secret, registration) = loop {
        let listener = bind_loopback(preferred_port).await?;
        let port = listener.local_addr().map_err(|e| McpError::Transport(e.to_string()))?.port();
        let redirect_uri = format!("http://127.0.0.1:{port}/callback");
        match &p.client {
            ClientSource::Configured { client_id, secret } => break (listener, redirect_uri, client_id.clone(), secret.clone(), None),
            ClientSource::Registered { client_id, secret, redirect_uri: old } if *old == redirect_uri => {
                break (listener, redirect_uri, client_id.clone(), secret.clone(), None)
            }
            _ => {}
        }
        let Some(ep) = server.registration_endpoint.as_ref() else {
            return Err(McpError::Invalid(if server.cimd_supported {
                "this server asks for a client ID that is the address of a client metadata document, which Dara cannot host for a desktop app. Add a client ID in the connector's sign-in settings (an https address of a document you host works as a client ID).".into()
            } else {
                "this server needs a client ID; add one in the connector's sign-in settings".into()
            }));
        };
        match register_client(ep, &redirect_uri, scope.as_deref(), net).await {
            Ok((id, sec)) => {
                let reg = NewRegistration { client_id: id.clone(), client_secret: sec.clone(), redirect_uri: redirect_uri.clone() };
                break (listener, redirect_uri, id, sec, Some(reg));
            }
            Err(RegisterError::Redirect) => {
                tries += 1;
                if tries >= PORT_TRIES {
                    return Err(McpError::Invalid(redirect_refused(&redirect_uri)));
                }
                preferred_port = None; // another port
            }
            Err(RegisterError::Other(e)) => return Err(e),
        }
    };

    let (verifier, challenge) = pkce_pair();
    let state = random_b64url(32);
    let mut auth_url = server.authorization_endpoint.clone();
    {
        let mut q = auth_url.query_pairs_mut();
        q.append_pair("response_type", "code")
            .append_pair("client_id", &client_id)
            .append_pair("redirect_uri", &redirect_uri)
            .append_pair("state", &state)
            .append_pair("code_challenge", &challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("resource", &discovered.resource);
        if let Some(s) = &scope {
            q.append_pair("scope", s);
        }
    }
    open_url(auth_url.as_str());
    let issuer = IssuerCheck { issuer: &server.issuer, required: server.iss_required };
    let code = wait_for_code(listener, &state, &issuer, p.timeout, cancel).await?;

    let auth = ClientAuth::new(&client_id, secret, &server.token_auth_methods);
    let form = vec![
        ("grant_type", "authorization_code".to_string()),
        ("code", code),
        ("redirect_uri", redirect_uri),
        ("code_verifier", verifier),
        ("resource", discovered.resource.clone()),
    ];
    let v = token_request(&server.token_endpoint, form, &auth, net).await.map_err(TokenError::into_mcp)?;
    let mut tokens = tokens_from_response(&v, None, &client_id, &server.token_endpoint, server.revocation_endpoint.as_ref(), &discovered.resource, &auth.method)?;
    if tokens.scope.is_none() {
        tokens.scope = scope;
    }
    Ok(SignInOutcome { tokens, registration })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64url_matches_the_rfc_vectors() {
        assert_eq!(b64url(b""), "");
        assert_eq!(b64url(b"f"), "Zg");
        assert_eq!(b64url(b"fo"), "Zm8");
        assert_eq!(b64url(b"foo"), "Zm9v");
        assert_eq!(b64url(&[0xfb, 0xff]), "-_8");
    }

    #[test]
    fn the_pkce_challenge_is_s256_of_the_verifier() {
        // the example of RFC 7636 appendix B
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(b64url(&Sha256::digest(verifier.as_bytes())), "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
        let (v, c) = pkce_pair();
        assert_eq!(v.len(), 64);
        assert_eq!(b64url(&Sha256::digest(v.as_bytes())), c);
    }

    #[test]
    fn client_authentication_follows_what_the_server_offers() {
        let m = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(ClientAuth::new("c", None, &m(&["client_secret_post"])).method, "none");
        assert_eq!(ClientAuth::new("c", Some("s".into()), &[]).method, "post");
        assert_eq!(ClientAuth::new("c", Some("s".into()), &m(&["client_secret_basic"])).method, "basic");
        assert_eq!(ClientAuth::new("c", Some("s".into()), &m(&["client_secret_basic", "client_secret_post"])).method, "post");
    }

    #[test]
    fn private_and_internal_addresses_are_refused() {
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        for bad in [
            "10.0.0.5", "172.16.3.4", "192.168.1.1", "169.254.169.254", "0.0.0.0", "100.64.0.1", "224.0.0.1", "fc00::1", "fd12:3456::1",
            "fe80::1", "::", "::ffff:10.0.0.1", "::ffff:169.254.169.254",
        ] {
            assert!(is_forbidden_ip(ip(bad), true), "{bad} must be refused");
        }
        for good in ["93.184.216.34", "8.8.8.8", "2606:4700::1111"] {
            assert!(!is_forbidden_ip(ip(good), false), "{good} is a public address");
        }
        // loopback only when the MCP server itself is on this computer
        assert!(is_forbidden_ip(ip("127.0.0.1"), false));
        assert!(is_forbidden_ip(ip("::1"), false));
        assert!(is_forbidden_ip(ip("::ffff:127.0.0.1"), false));
        assert!(!is_forbidden_ip(ip("127.0.0.1"), true));
        assert!(!is_forbidden_ip(ip("::1"), true));
    }

    #[tokio::test]
    async fn the_guard_checks_literals_and_names() {
        let no = Net { allow_loopback: false };
        let yes = Net { allow_loopback: true };
        let u = |s: &str| Url::parse(s).unwrap();
        assert!(guarded_client(&u("https://10.0.0.5/token"), no).await.is_err());
        assert!(guarded_client(&u("https://192.168.1.2/token"), yes).await.is_err());
        assert!(guarded_client(&u("https://[fd00::1]/token"), yes).await.is_err());
        assert!(guarded_client(&u("http://example.com/token"), yes).await.is_err()); // not https
        assert!(guarded_client(&u("http://127.0.0.1:9/token"), no).await.is_err()); // loopback not allowed
        assert!(guarded_client(&u("http://127.0.0.1:9/token"), yes).await.is_ok());
        // a name that resolves to this computer is the same as the address
        assert!(guarded_client(&u("http://localhost:9/token"), no).await.is_err());
        assert!(guarded_client(&u("http://localhost:9/token"), yes).await.is_ok());
    }

    #[test]
    fn the_issuer_check_follows_rfc_9207() {
        let c = IssuerCheck { issuer: "https://as.example", required: false };
        assert!(c.verify(Some("https://as.example")).is_ok());
        assert!(c.verify(Some("https://as.example/")).is_ok());
        assert!(c.verify(Some("https://evil.example")).is_err());
        assert!(c.verify(None).is_ok());
        let strict = IssuerCheck { issuer: "https://as.example", required: true };
        assert!(strict.verify(None).is_err());
        let tenant = IssuerCheck { issuer: "https://login.example/{tenantid}/v2.0", required: true };
        assert!(tenant.verify(Some("https://login.example/abc/v2.0")).is_ok());
    }
}
