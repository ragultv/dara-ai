//! OAuth sign-in, token handling and the negative cases, against a mock server that enforces what a real one does.

mod common;

use common::auth::{self, browse, browser, no_browser, Knobs, MockAuth};
use dara_mcp::manager::scrub_args;
use dara_mcp::oauth::StoredTokens;
use dara_mcp::{
    secrets, ConnState, ConnectorPatch, Kind, Manager, McpError, MemorySecrets, NewConnector, OAuthInput, TokenProvider,
};
use futures_util::future::join_all;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

struct Rig {
    auth: MockAuth,
    m: Arc<Manager>,
    secrets: Arc<MemorySecrets>,
    dir: tempfile::TempDir,
    id: String,
}

async fn rig(configure: impl FnOnce(&mut Knobs), oauth: Option<OAuthInput>) -> Rig {
    let auth = auth::start(configure).await;
    let dir = tempfile::tempdir().unwrap();
    let secrets = Arc::new(MemorySecrets::new());
    let m = Manager::new(&dir.path().join("c.db"), secrets.clone(), &dir.path().join("scratch")).unwrap();
    let c = m
        .add(NewConnector {
            name: "Auth".into(),
            kind: Kind::Http,
            command: None,
            args: vec![],
            env: BTreeMap::new(),
            url: Some(auth.mcp_url()),
            oauth,
        })
        .unwrap();
    m.set_sign_in_timeout(Duration::from_secs(10));
    Rig { auth, m, secrets, dir, id: c.id }
}

impl Rig {
    async fn sign_in(&self) -> Result<dara_mcp::ConnStatus, McpError> {
        let b = browser();
        self.m.sign_in(&self.id, &b).await
    }
    fn stored(&self) -> StoredTokens {
        let raw = secrets::get_chunked(self.secrets.as_ref(), &secrets::token_key(&self.id)).unwrap().expect("tokens");
        serde_json::from_str(&raw).unwrap()
    }
    /// Makes the stored access token look expired.
    fn expire(&self) {
        let mut t = self.stored();
        t.expires_at_ms = Some(0);
        secrets::put_chunked(self.secrets.as_ref(), &secrets::token_key(&self.id), &serde_json::to_string(&t).unwrap()).unwrap();
        self.m.oauth_tokens(&self.id).unwrap().reload();
    }
    async fn call_echo(&self, text: &str) -> Result<dara_mcp::CallOutcome, McpError> {
        self.m.call(&self.id, "echo", json!({"text": text}), true, None).await
    }
    fn db_has(&self, needle: &str) -> bool {
        ["c.db", "c.db-wal", "c.db-shm"].iter().any(|f| {
            std::fs::read(self.dir.path().join(f))
                .map(|b| b.windows(needle.len()).any(|w| w == needle.as_bytes()))
                .unwrap_or(false)
        })
    }
}

fn err_text(r: Result<dara_mcp::ConnStatus, McpError>) -> String {
    r.expect_err("the sign-in should have failed").to_string()
}

// ---- discovery and who signs in ----

#[tokio::test]
async fn discovery_uses_the_pointer_in_the_401() {
    let r = rig(|_| {}, None).await;
    let st = r.sign_in().await.unwrap();
    assert_eq!((st.state, st.signed_in), (ConnState::Ready, true));
    let paths = r.auth.paths();
    assert!(paths.contains(&"GET /.well-known/oauth-protected-resource".to_string()), "{paths:?}");
    assert!(paths.contains(&"GET /.well-known/oauth-authorization-server".to_string()));
}

#[tokio::test]
async fn discovery_falls_back_to_the_well_known_address_of_the_server() {
    let r = rig(|k| k.header_hint = false, None).await;
    assert!(r.sign_in().await.unwrap().signed_in);
    // RFC 9728: the well-known address with the server's path added comes first
    assert!(r.auth.paths().contains(&"GET /.well-known/oauth-protected-resource/mcp".to_string()));
}

#[tokio::test]
async fn a_new_client_is_registered_once_and_reused() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    let reg = &r.auth.registrations()[0];
    assert_eq!(reg["token_endpoint_auth_method"], "none");
    assert_eq!(reg["grant_types"], json!(["authorization_code", "refresh_token"]));
    assert!(reg["redirect_uris"][0].as_str().unwrap().starts_with("http://127.0.0.1:"));
    assert_eq!(r.m.list().unwrap()[0].config.oauth.client_id.as_deref(), Some("dcr-1"));

    r.sign_in().await.unwrap(); // again: the same client, no second registration
    assert_eq!(r.auth.registrations().len(), 1);
    assert_eq!(r.auth.authorizes()[1]["client_id"], "dcr-1");
}

#[tokio::test]
async fn a_configured_client_id_with_a_secret_skips_registration() {
    let oauth = OAuthInput {
        client_id: Some("my-client".into()),
        client_secret: Some("s3cret-value-1234".into()),
        scope: Some("mail.read".into()),
    };
    let r = rig(|k| { k.client_secrets.insert("my-client".into(), "s3cret-value-1234".into()); }, Some(oauth)).await;
    assert!(r.sign_in().await.unwrap().signed_in);
    assert!(r.auth.registrations().is_empty());
    let a = &r.auth.authorizes()[0];
    assert_eq!((a["client_id"].as_str(), a["scope"].as_str()), ("my-client", "mail.read"));
    assert_eq!(r.auth.token_requests()[0]["client_secret"], "s3cret-value-1234");
    let cfg = r.m.list().unwrap()[0].config.clone();
    assert!(cfg.oauth.has_secret);
    assert!(!r.db_has("s3cret-value-1234"), "the client secret must only be in the credential store");
    assert!(!serde_json::to_string(&cfg).unwrap().contains("s3cret-value-1234"));
}

#[tokio::test]
async fn a_server_that_cannot_register_dara_says_a_client_id_is_needed() {
    let r = rig(|k| k.registration = false, None).await;
    assert!(err_text(r.sign_in().await).contains("needs a client ID"));
    // detected, not supported: with the document flag the message says what could be done
    let r = rig(|k| { k.registration = false; k.cimd_flag = true; }, None).await;
    assert!(err_text(r.sign_in().await).contains("metadata document"));
    // and registration still wins when it is offered
    let r = rig(|k| k.cimd_flag = true, None).await;
    assert!(r.sign_in().await.unwrap().signed_in);
    assert_eq!(r.auth.registrations().len(), 1);
}

#[tokio::test]
async fn a_server_that_needs_no_sign_in_says_so() {
    let r = rig(|k| k.mcp_open = true, None).await;
    assert!(err_text(r.sign_in().await).contains("does not ask you to sign in"));
}

// ---- the authorization request and the callback ----

#[tokio::test]
async fn pkce_state_resource_and_the_loopback_redirect_are_sent_and_checked() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    let a = &r.auth.authorizes()[0];
    assert_eq!(a["response_type"], "code");
    assert_eq!(a["code_challenge_method"], "S256");
    assert!(a["state"].len() >= 32);
    assert_eq!(a["resource"], r.auth.mcp_url());
    assert!(a["redirect_uri"].starts_with("http://127.0.0.1:") && a["redirect_uri"].ends_with("/callback"));
    let t = &r.auth.token_requests()[0];
    assert_eq!(t["code_verifier"].len(), 64);
    assert_eq!(t["resource"], r.auth.mcp_url());
    assert!(t["_verdict"].starts_with("Ok"), "the mock verified S256 of the verifier against the challenge: {}", t["_verdict"]);
}

#[tokio::test]
async fn the_mock_really_rejects_a_wrong_verifier() {
    // proves the check above has teeth: the same flow with a wrong verifier is refused
    let r = rig(|_| {}, None).await;
    let c = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).build().unwrap();
    let base = format!("http://127.0.0.1:{}", r.auth.addr.port());
    let auth_url = format!(
        "{base}/authorize?response_type=code&client_id=x&redirect_uri=http%3A%2F%2F127.0.0.1%3A5%2Fcallback&state=s&code_challenge=abc&code_challenge_method=S256&resource={}",
        r.auth.mcp_url()
    );
    let loc = c.get(&auth_url).send().await.unwrap().headers()["location"].to_str().unwrap().to_string();
    let code = url::Url::parse(&loc).unwrap().query_pairs().find(|(k, _)| k == "code").unwrap().1.to_string();
    let res = c
        .post(format!("{base}/token"))
        .form(&[("grant_type", "authorization_code"), ("code", code.as_str()), ("code_verifier", "wrong"), ("client_id", "x"), ("redirect_uri", "http://127.0.0.1:5/callback"), ("resource", r.auth.mcp_url().as_str())])
        .send()
        .await
        .unwrap();
    assert_eq!(res.status().as_u16(), 400);
}

#[tokio::test]
async fn a_wrong_state_is_turned_away_and_nothing_is_stored() {
    let r = rig(|k| k.state_override = Some("not-the-state".into()), None).await;
    r.m.set_sign_in_timeout(Duration::from_millis(1500));
    assert!(err_text(r.sign_in().await).contains("timed out"));
    assert!(r.auth.token_requests().is_empty(), "no code may be exchanged after a wrong state");
    assert!(!r.m.status_of(&r.id).signed_in);
}

#[tokio::test]
async fn an_error_from_the_server_is_shown_and_a_refused_redirect_is_explained() {
    let r = rig(|k| k.authorize_error = Some("access_denied".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("access_denied"));
    let r = rig(|k| k.authorize_error = Some("invalid_redirect_uri".into()), None).await;
    let e = err_text(r.sign_in().await);
    assert!(e.contains("redirect address") && e.contains("client ID"), "{e}");
}

#[tokio::test]
async fn a_registration_that_refuses_the_redirect_is_retried_then_explained() {
    let r = rig(|k| k.registration_rejects_redirect = true, None).await;
    let e = err_text(r.sign_in().await);
    assert!(e.contains("redirect address") && e.contains("matches redirect addresses exactly"), "{e}");
    assert_eq!(r.auth.registrations().len(), 3, "another port is tried a few times");
    let ports: std::collections::HashSet<_> = r.auth.registrations().iter().map(|g| g["redirect_uris"][0].as_str().unwrap().to_string()).collect();
    assert!(ports.len() > 1, "each try uses a different port");
}

#[tokio::test]
async fn the_issuer_in_the_answer_is_checked() {
    let r = rig(|k| k.iss_in_response = Some("http://evil.example".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("different server"));
    assert!(r.auth.token_requests().is_empty());
    // promised but missing
    let r = rig(|k| k.iss_promised = true, None).await;
    assert!(err_text(r.sign_in().await).contains("did not say which server"));
    // present and right (the server's own address)
    let r = rig(|k| { k.iss_promised = true; }, None).await;
    let port = r.auth.addr.port();
    r.auth.knobs(|k| k.iss_in_response = Some(format!("http://127.0.0.1:{port}")));
    assert!(r.sign_in().await.unwrap().signed_in);
}

#[tokio::test]
async fn an_expired_authorization_code_is_refused() {
    let r = rig(|k| k.code_ttl_ms = -1, None).await;
    let e = err_text(r.sign_in().await);
    assert!(e.contains("invalid_grant"), "{e}");
    assert!(!r.m.status_of(&r.id).signed_in);
}

#[tokio::test]
async fn an_answer_sent_to_another_host_never_reaches_dara() {
    let r = rig(|k| k.redirect_elsewhere = Some("http://never.invalid/callback".into()), None).await;
    r.m.set_sign_in_timeout(Duration::from_millis(1500));
    assert!(err_text(r.sign_in().await).contains("timed out"));
    // what Dara sent was always its own loopback address
    assert!(r.auth.authorizes()[0]["redirect_uri"].starts_with("http://127.0.0.1:"));
}

#[tokio::test]
async fn a_taken_loopback_port_gets_another_and_a_new_registration() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    let old = r.m.store().get_connector(&r.id).unwrap().unwrap().oauth_redirect_uri.unwrap();
    let port: u16 = url::Url::parse(&old).unwrap().port().unwrap();
    let _held = tokio::net::TcpListener::bind(("127.0.0.1", port)).await.expect("the port is free again");
    r.sign_in().await.unwrap();
    let new = r.m.store().get_connector(&r.id).unwrap().unwrap().oauth_redirect_uri.unwrap();
    assert_ne!(new, old);
    assert_eq!(r.auth.registrations().len(), 2, "the client was registered again for the new address");
    // a free preferred port is reused
    drop(_held);
    let l = dara_mcp::oauth::bind_loopback(Some(port)).await.unwrap();
    assert_eq!(l.local_addr().unwrap().port(), port);
}

#[tokio::test]
async fn a_browser_that_never_returns_times_out_and_a_sign_in_can_be_cancelled() {
    let r = rig(|_| {}, None).await;
    r.m.set_sign_in_timeout(Duration::from_millis(800));
    let b = no_browser();
    assert!(r.m.sign_in(&r.id, &b).await.unwrap_err().to_string().contains("timed out"));

    r.m.set_sign_in_timeout(Duration::from_secs(30));
    let (m, id) = (r.m.clone(), r.id.clone());
    let task = tokio::spawn(async move {
        let b = no_browser();
        m.sign_in(&id, &b).await
    });
    loop {
        if r.m.cancel_sign_in(&r.id) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    }
    assert!(task.await.unwrap().unwrap_err().to_string().contains("cancelled"));
    assert!(!r.m.cancel_sign_in(&r.id), "nothing is waiting any more");
    r.m.set_sign_in_timeout(Duration::from_secs(10));
    assert!(r.sign_in().await.unwrap().signed_in, "a new sign-in works after a cancel");
}

// ---- what a server may point us at ----

#[tokio::test]
async fn private_and_non_https_servers_are_refused() {
    let r = rig(|k| k.prm_authorization_servers = Some(vec!["https://10.0.0.5".into()]), None).await;
    assert!(err_text(r.sign_in().await).contains("private or internal"));
    let r = rig(|k| k.prm_authorization_servers = Some(vec!["http://example.com".into()]), None).await;
    assert!(err_text(r.sign_in().await).contains("not acceptable"));
    // a token address inside the metadata
    let r = rig(|k| k.token_endpoint_override = Some("https://169.254.169.254/token".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("private or internal"));
    let r = rig(|k| k.token_endpoint_override = Some("http://example.com/token".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("not acceptable"));
    // a registration address that is not acceptable is not used, so a client id is needed
    let r = rig(|k| k.registration_endpoint_override = Some("https://192.168.0.9/register".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("needs a client ID"));
}

#[tokio::test]
async fn an_authorization_server_without_s256_or_with_another_issuer_is_refused() {
    let r = rig(|k| k.omit_s256 = true, None).await;
    assert!(err_text(r.sign_in().await).contains("PKCE"));
    assert!(r.auth.authorizes().is_empty());
    let r = rig(|k| k.meta_issuer = Some("https://other.example".into()), None).await;
    assert!(err_text(r.sign_in().await).contains("different issuer"));
}

// ---- tokens ----

#[tokio::test]
async fn a_long_token_is_kept_in_pieces_and_read_back_whole() {
    let r = rig(|k| k.big_token = true, None).await;
    assert!(r.sign_in().await.unwrap().signed_in);
    let issued = r.auth.last_access();
    assert!(issued.len() > 6000);
    let keys = r.secrets.keys();
    let pieces: Vec<_> = keys.iter().filter(|k| k.starts_with(&format!("{}/", secrets::token_key(&r.id)))).collect();
    assert!(pieces.len() >= 5, "{keys:?}");
    {
        use dara_mcp::SecretStore;
        assert!(pieces.iter().all(|k| r.secrets.get(k).unwrap().unwrap().chars().count() <= secrets::CHUNK_CHARS));
    }
    let p = r.m.oauth_tokens(&r.id).unwrap();
    assert_eq!(p.bearer().await.as_deref(), Some(issued.as_str()));
    // a missing piece reads as no token at all, never as half a token
    use dara_mcp::SecretStore;
    r.secrets.delete(&format!("{}/1", secrets::token_key(&r.id))).unwrap();
    p.reload();
    assert_eq!(p.bearer().await, None);
    assert!(!r.m.status_of(&r.id).signed_in);
}

#[tokio::test]
async fn an_expiring_token_is_refreshed_before_use_by_one_request() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    let before = r.stored().access_token;
    r.expire();
    let p = r.m.oauth_tokens(&r.id).unwrap();
    let all = join_all((0..8).map(|_| p.bearer())).await;
    let fresh = r.auth.last_access();
    assert_ne!(fresh, before);
    assert!(all.iter().all(|t| t.as_deref() == Some(fresh.as_str())));
    assert_eq!(r.auth.refresh_requests(), 1, "eight callers, one refresh");
    assert_eq!(p.refresh_count(), 1);
    assert!(r.call_echo("hi").await.is_ok());
}

#[tokio::test]
async fn a_rotated_refresh_token_is_saved_and_the_old_one_is_not_used_again() {
    let r = rig(|k| k.rotate_refresh = true, None).await;
    r.sign_in().await.unwrap();
    let first = r.stored().refresh_token.unwrap();
    r.expire();
    let p = r.m.oauth_tokens(&r.id).unwrap();
    assert!(p.bearer().await.is_some());
    let second = r.stored().refresh_token.unwrap();
    assert_ne!(first, second);
    assert_eq!(second, r.auth.last_refresh(), "the new refresh token is what is stored");
    r.expire();
    assert!(p.bearer().await.is_some(), "the second refresh used the NEW token (the mock refuses the old one)");
    assert_eq!(r.auth.refresh_requests(), 2);
}

#[tokio::test]
async fn a_revoked_refresh_token_ends_the_sign_in_without_a_retry_loop() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    r.auth.knobs(|k| k.refresh_revoked = true);
    r.expire();
    let e = r.call_echo("hi").await.unwrap_err();
    assert!(matches!(e, McpError::AuthRequired { .. }), "{e}");
    let st = r.m.status_of(&r.id);
    assert_eq!((st.state, st.signed_in), (ConnState::NeedsAuth, false));
    assert!(!r.secrets.keys().iter().any(|k| k.contains("/oauth/token")), "the tokens are gone");
    assert_eq!(r.auth.refresh_requests(), 1);
    for _ in 0..3 {
        assert!(r.call_echo("again").await.is_err());
    }
    assert_eq!(r.auth.refresh_requests(), 1, "no loop: the dead refresh token is not tried again");
}

#[tokio::test]
async fn a_401_in_the_middle_gets_one_refresh_and_one_retry() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    r.auth.forget_access_tokens(); // the server no longer knows the access token; the refresh token is fine
    let out = r.call_echo("hello").await.unwrap();
    assert_eq!(out.content[0]["text"], "hello");
    assert_eq!(r.auth.refresh_requests(), 1);
    assert_eq!(r.auth.mcp_calls("tools/call"), 2, "the refused call, then the retry");

    // a second 401 is the end: one refresh, one retry, then the user has to sign in
    r.auth.knobs(|k| k.mcp_always_401 = true);
    let before = r.auth.refresh_requests();
    assert!(matches!(r.call_echo("x").await.unwrap_err(), McpError::AuthRequired { .. }));
    assert_eq!(r.auth.refresh_requests(), before + 1);
    assert_eq!(r.m.status_of(&r.id).state, ConnState::NeedsAuth);
}

#[tokio::test]
async fn insufficient_scope_is_reported_and_never_fixed_silently() {
    let r = rig(|k| { k.challenge_scope = Some("calendar.read".into()); k.call_needs_scope = Some("calendar.write".into()); }, None).await;
    r.sign_in().await.unwrap();
    assert_eq!(r.auth.authorizes()[0]["scope"], "calendar.read");
    let e = r.call_echo("hi").await.unwrap_err();
    assert!(matches!(&e, McpError::InsufficientScope { scope } if scope.as_deref() == Some("calendar.write")), "{e}");
    let st = r.m.status_of(&r.id);
    assert_eq!((st.state, st.needed_scope.as_deref()), (ConnState::NeedsAuth, Some("calendar.write")));
    assert!(st.error.unwrap().contains("calendar.write"));
    assert_eq!(r.auth.authorizes().len(), 1, "nothing re-authorized by itself");
    assert_eq!(r.auth.token_requests().len(), 1);

    // the user chooses to: the new sign-in asks for what was granted plus what is needed
    r.sign_in().await.unwrap();
    let scope = r.auth.authorizes()[1]["scope"].clone();
    assert!(scope.split(' ').any(|w| w == "calendar.read") && scope.split(' ').any(|w| w == "calendar.write"), "{scope}");
    assert!(r.call_echo("hi").await.is_ok());
    assert_eq!(r.m.status_of(&r.id).needed_scope, None);
}

#[tokio::test]
async fn signing_out_revokes_forgets_and_clears_a_registered_client() {
    let r = rig(|_| {}, None).await;
    r.sign_in().await.unwrap();
    let rt = r.stored().refresh_token.unwrap();
    let st = r.m.sign_out(&r.id).await.unwrap();
    assert_eq!((st.state, st.signed_in), (ConnState::Disconnected, false));
    let rev = &r.auth.revocations()[0];
    assert_eq!((rev["token"].as_str(), rev["token_type_hint"].as_str()), (rt.as_str(), "refresh_token"));
    assert!(!r.secrets.keys().iter().any(|k| k.contains("/oauth/")), "no token and no client secret is left");
    assert_eq!(r.m.list().unwrap()[0].config.oauth.client_id, None, "a client the server registered is forgotten");
}

#[tokio::test]
async fn changing_the_client_id_signs_the_connector_out() {
    let oauth = OAuthInput { client_id: Some("one".into()), client_secret: None, scope: None };
    let r = rig(|_| {}, Some(oauth)).await;
    r.sign_in().await.unwrap();
    assert!(r.m.status_of(&r.id).signed_in);
    let patch = ConnectorPatch { oauth: Some(OAuthInput { client_id: Some("two".into()), client_secret: Some("shh-secret-9".into()), scope: None }), ..Default::default() };
    let c = r.m.update(&r.id, patch).await.unwrap();
    assert_eq!((c.oauth.client_id.as_deref(), c.oauth.has_secret), (Some("two"), true));
    assert!(!r.m.status_of(&r.id).signed_in, "tokens of the other client are not kept");
    // an empty secret keeps the stored one; an empty client id forgets everything
    let keep = ConnectorPatch { oauth: Some(OAuthInput { client_id: None, client_secret: Some(String::new()), scope: Some("a b".into()) }), ..Default::default() };
    assert!(r.m.update(&r.id, keep).await.unwrap().oauth.has_secret);
    let forget = ConnectorPatch { oauth: Some(OAuthInput { client_id: Some(String::new()), client_secret: None, scope: None }), ..Default::default() };
    let c = r.m.update(&r.id, forget).await.unwrap();
    assert_eq!((c.oauth.client_id, c.oauth.has_secret), (None, false));
    // sign-in settings belong to a server address only
    let dir = tempfile::tempdir().unwrap();
    let m = Manager::new(&dir.path().join("x.db"), Arc::new(MemorySecrets::new()), dir.path()).unwrap();
    let stdio = NewConnector { name: "p".into(), kind: Kind::Stdio, command: Some("x".into()), args: vec![], env: BTreeMap::new(), url: None, oauth: Some(OAuthInput { client_id: Some("c".into()), ..Default::default() }) };
    assert!(m.add(stdio).is_err());
}

// ---- nothing secret leaks ----

#[tokio::test]
async fn tokens_never_reach_the_database_the_audit_log_or_anything_returned() {
    let r = rig(|k| { k.rotate_refresh = true; k.client_secrets.insert("conf".into(), "client-secret-xyz-123".into()); }, Some(OAuthInput { client_id: Some("conf".into()), client_secret: Some("client-secret-xyz-123".into()), scope: None })).await;
    r.sign_in().await.unwrap();
    let (at, rt) = (r.stored().access_token, r.stored().refresh_token.unwrap());
    // a call whose arguments carry the secrets (as a careless tool call might)
    let args = json!({"text": format!("Authorization: Bearer {at}"), "access_token": at, "note": format!("rt={rt} and Bearer abc.def-123"), "n": 3});
    r.m.call(&r.id, "echo", args, true, None).await.unwrap();

    let audit = serde_json::to_string(&r.m.audit(Some(&r.id), 50).unwrap()).unwrap();
    let list = serde_json::to_string(&r.m.list().unwrap()).unwrap();
    let tools = serde_json::to_string(&r.m.tools(&r.id).unwrap()).unwrap();
    let status = serde_json::to_string(&r.m.status_of(&r.id)).unwrap();
    for secret in [&at, &rt, "client-secret-xyz-123", "abc.def-123"] {
        for (what, text) in [("audit", &audit), ("list", &list), ("tools", &tools), ("status", &status)] {
            assert!(!text.contains(secret), "{what} leaks {secret}");
        }
        assert!(!r.db_has(secret), "the database holds {secret}");
    }
    let args = r.m.audit(Some(&r.id), 50).unwrap()[0].args_json.clone();
    assert!(args.contains("[removed]") && args.contains("\"n\":3"), "the rest of the arguments is kept: {args}");
}

#[test]
fn scrubbing_removes_known_secrets_bearer_values_and_token_keys() {
    let known = vec!["supersecretvalue".to_string(), "short".to_string()];
    let out = scrub_args(&json!({"a": "x supersecretvalue y", "Authorization": "Basic zzz", "refresh_token": "r", "nested": [{"b": "BEARER tok.en-1~ end"}], "keep": "short word", "n": 1}), &known);
    assert!(!out.contains("supersecretvalue") && !out.contains("zzz") && !out.contains("tok.en-1"));
    assert!(out.contains("Bearer") == false || out.contains("[removed]"));
    assert!(out.contains("short word"), "values shorter than 8 characters are not treated as secrets");
    assert!(out.contains("\"n\":1"));
}

#[tokio::test]
async fn the_browser_helper_reaches_the_callback() {
    // sanity for the mock itself: the address Dara opens redirects to a loopback address and Dara answers it
    let r = rig(|_| {}, None).await;
    let (m, id) = (r.m.clone(), r.id.clone());
    let url = Arc::new(std::sync::Mutex::new(String::new()));
    let u = url.clone();
    let task = tokio::spawn(async move {
        let open = move |s: &str| *u.lock().unwrap() = s.to_string();
        m.sign_in(&id, &open).await
    });
    let opened = loop {
        let s = url.lock().unwrap().clone();
        if !s.is_empty() {
            break s;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    };
    let body = browse(&opened).await.unwrap();
    assert!(body.contains("You are signed in"));
    assert!(task.await.unwrap().unwrap().signed_in);
}
