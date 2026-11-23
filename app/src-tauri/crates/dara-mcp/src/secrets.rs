//! Where secret values live. Never in the SQLite file and never in logs.

use crate::error::McpError;
use std::collections::HashMap;
use std::sync::Mutex;

pub trait SecretStore: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<String>, McpError>;
    fn set(&self, key: &str, value: &str) -> Result<(), McpError>;
    /// Deleting a key that does not exist is not an error.
    fn delete(&self, key: &str) -> Result<(), McpError>;
}

/// The key under which one environment variable of a stdio connector is kept.
pub fn env_key(connector_id: &str, name: &str) -> String {
    format!("connector/{connector_id}/env/{name}")
}

/// Secrets in memory: for tests (and for a headless run that must not touch the OS store).
#[derive(Default)]
pub struct MemorySecrets(Mutex<HashMap<String, String>>);

impl MemorySecrets {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn keys(&self) -> Vec<String> {
        let mut k: Vec<String> = self.0.lock().expect("secrets").keys().cloned().collect();
        k.sort();
        k
    }
}

impl SecretStore for MemorySecrets {
    fn get(&self, key: &str) -> Result<Option<String>, McpError> {
        Ok(self.0.lock().expect("secrets").get(key).cloned())
    }
    fn set(&self, key: &str, value: &str) -> Result<(), McpError> {
        self.0.lock().expect("secrets").insert(key.to_string(), value.to_string());
        Ok(())
    }
    fn delete(&self, key: &str) -> Result<(), McpError> {
        self.0.lock().expect("secrets").remove(key);
        Ok(())
    }
}

/// The operating system's credential store (Windows Credential Manager), service name "Dara".
pub struct KeyringSecrets {
    service: String,
}

impl KeyringSecrets {
    pub fn new(service: &str) -> Self {
        Self { service: service.to_string() }
    }
    fn entry(&self, key: &str) -> Result<keyring::Entry, McpError> {
        keyring::Entry::new(&self.service, key).map_err(|e| McpError::Storage(format!("credential store: {e}")))
    }
}

impl SecretStore for KeyringSecrets {
    fn get(&self, key: &str) -> Result<Option<String>, McpError> {
        match self.entry(key)?.get_password() {
            Ok(v) => Ok(Some(v)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(McpError::Storage(format!("credential store: {e}"))),
        }
    }
    fn set(&self, key: &str, value: &str) -> Result<(), McpError> {
        self.entry(key)?
            .set_password(value)
            .map_err(|e| McpError::Storage(format!("credential store: {e}")))
    }
    fn delete(&self, key: &str) -> Result<(), McpError> {
        match self.entry(key)?.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(McpError::Storage(format!("credential store: {e}"))),
        }
    }
}

/// The key of the API key a user pasted for an http connector (a server that wants a key instead of a sign-in).
pub fn api_key_key(connector_id: &str) -> String {
    format!("connector/{connector_id}/api_key")
}

/// The key of a connector's OAuth token record (kept in pieces, see [`put_chunked`]).
pub fn token_key(connector_id: &str) -> String {
    format!("connector/{connector_id}/oauth/token")
}

/// The key of a connector's OAuth client secret (a configured one, or one the server gave at registration).
pub fn client_secret_key(connector_id: &str) -> String {
    format!("connector/{connector_id}/oauth/client_secret")
}

/// A Windows credential holds about 2.5 KB and a JWT access token can be larger, so a long value is split into pieces of at most this
/// many characters, each under its own key.
pub const CHUNK_CHARS: usize = 1500;

fn chunk_key(base: &str, i: usize) -> String {
    format!("{base}/{i}")
}

/// Header format: `<pieces>:<total characters>:<first 16 hex digits of the SHA-256 of the value>`.
fn parse_header(h: &str) -> Option<(usize, usize, String)> {
    let mut it = h.splitn(3, ':');
    let n = it.next()?.parse().ok()?;
    let len = it.next()?.parse().ok()?;
    let hash = it.next()?.to_string();
    (hash.len() == 16).then_some((n, len, hash))
}

fn value_hash(value: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(value.as_bytes()))[..16].to_string()
}

/// Stores `value` in pieces: the pieces first, then the header, so a reader never sees a header without its pieces.
pub fn put_chunked(store: &dyn SecretStore, base: &str, value: &str) -> Result<(), McpError> {
    let chars: Vec<char> = value.chars().collect();
    let pieces: Vec<String> = chars.chunks(CHUNK_CHARS).map(|c| c.iter().collect()).collect();
    let old = store.get(base)?.and_then(|h| parse_header(&h)).map(|(n, _, _)| n).unwrap_or(0);
    for (i, piece) in pieces.iter().enumerate() {
        store.set(&chunk_key(base, i), piece)?;
    }
    // the header goes LAST: until it is written, a reader sees the previous value (or none), never a mixture
    store.set(base, &format!("{}:{}:{}", pieces.len(), chars.len(), value_hash(value)))?;
    for i in pieces.len()..old {
        store.delete(&chunk_key(base, i))?;
    }
    Ok(())
}

/// All or nothing: a missing piece, or a piece of the wrong size, reads as no value at all.
pub fn get_chunked(store: &dyn SecretStore, base: &str) -> Result<Option<String>, McpError> {
    let Some(header) = store.get(base)? else { return Ok(None) };
    let Some((n, total, hash)) = parse_header(&header) else { return Ok(None) };
    let mut out = String::new();
    for i in 0..n {
        match store.get(&chunk_key(base, i))? {
            Some(piece) => out.push_str(&piece),
            None => return Ok(None),
        }
    }
    Ok((out.chars().count() == total && value_hash(&out) == hash).then_some(out))
}

/// True when a value has been stored (its header exists). Cheap: it reads one entry.
pub fn has_chunked(store: &dyn SecretStore, base: &str) -> bool {
    store.get(base).ok().flatten().and_then(|h| parse_header(&h)).is_some()
}

/// Removes the header and every piece.
pub fn delete_chunked(store: &dyn SecretStore, base: &str) -> Result<(), McpError> {
    let n = store.get(base)?.and_then(|h| parse_header(&h)).map(|(n, _, _)| n).unwrap_or(0);
    store.delete(base)?;
    for i in 0..n {
        store.delete(&chunk_key(base, i))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_value_round_trips_in_pieces() {
        let s = MemorySecrets::new();
        let big: String = (0..6000).map(|i| char::from(b'a' + (i % 26) as u8)).collect();
        put_chunked(&s, "k", &big).unwrap();
        assert_eq!(get_chunked(&s, "k").unwrap().as_deref(), Some(big.as_str()));
        assert!(s.keys().iter().filter(|k| k.starts_with("k/")).count() >= 4);
        assert!(s.keys().iter().all(|k| s.get(k).unwrap().unwrap().chars().count() <= CHUNK_CHARS));
        // a shorter value leaves no stale pieces
        put_chunked(&s, "k", "short").unwrap();
        assert_eq!(s.keys(), vec!["k".to_string(), "k/0".to_string()]);
        assert_eq!(get_chunked(&s, "k").unwrap().as_deref(), Some("short"));
        delete_chunked(&s, "k").unwrap();
        assert!(s.keys().is_empty());
    }

    #[test]
    fn a_missing_or_damaged_piece_reads_as_no_value() {
        let s = MemorySecrets::new();
        let big = "x".repeat(4000);
        put_chunked(&s, "k", &big).unwrap();
        s.delete("k/1").unwrap();
        assert_eq!(get_chunked(&s, "k").unwrap(), None);
        put_chunked(&s, "k", &big).unwrap();
        s.set("k/2", "short").unwrap();
        assert_eq!(get_chunked(&s, "k").unwrap(), None);
        // a piece of the right size but different content (a half-written or mixed value) is caught by the hash
        put_chunked(&s, "k", &big).unwrap();
        s.set("k/1", &"y".repeat(1500)).unwrap();
        assert_eq!(get_chunked(&s, "k").unwrap(), None);
        assert_eq!(get_chunked(&s, "absent").unwrap(), None);
        assert!(has_chunked(&s, "k"));
    }
}
