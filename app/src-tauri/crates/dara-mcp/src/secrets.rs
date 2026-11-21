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
