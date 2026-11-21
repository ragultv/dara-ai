//! A server that is a child process: one JSON message per line on its stdin and stdout.

use super::{react, Reaction, Shared, Transport};
use crate::error::McpError;
use crate::rpc;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;

/// What the child process may inherit from this app; everything else must be given on purpose by the user.
const INHERITED_ENV: [&str; 8] = [
    "PATH",
    "SystemRoot",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "APPDATA",
    "LOCALAPPDATA",
    "HOME",
];
const STDERR_RING_BYTES: usize = 64 * 1024;
/// A single JSON line larger than this ends the connection (a broken or hostile server).
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;

pub struct StdioConfig {
    pub command: String,
    pub args: Vec<String>,
    /// The user's own variables (their values come from the secret store).
    pub env: Vec<(String, String)>,
    /// A scratch folder for this connector; the process starts there.
    pub cwd: PathBuf,
}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, McpError>>>>>;

pub struct StdioTransport {
    stdin: Arc<tokio::sync::Mutex<Option<ChildStdin>>>,
    pending: Pending,
    /// Set once the process is gone; the text says why.
    closed: Arc<Mutex<Option<String>>>,
    child: tokio::sync::Mutex<Option<Child>>,
    stderr: Arc<Mutex<VecDeque<u8>>>,
}

struct PendingGuard {
    pending: Pending,
    id: u64,
}
impl Drop for PendingGuard {
    fn drop(&mut self) {
        if let Ok(mut p) = self.pending.lock() {
            p.remove(&self.id);
        }
    }
}

/// `npx` is `npx.cmd` on Windows and the OS will not find it without the extension, so look for the usual ones on PATH.
fn resolve_command(command: &str) -> PathBuf {
    let direct = PathBuf::from(command);
    let has_dir = command.contains('/') || command.contains('\\');
    if cfg!(windows) && !has_dir && Path::new(command).extension().is_none() {
        if let Some(path) = std::env::var_os("PATH") {
            for dir in std::env::split_paths(&path) {
                for ext in ["exe", "cmd", "bat", "com"] {
                    let candidate = dir.join(format!("{command}.{ext}"));
                    if candidate.is_file() {
                        return candidate;
                    }
                }
            }
        }
    }
    direct
}

impl StdioTransport {
    pub fn spawn(config: StdioConfig, shared: Arc<Shared>) -> Result<Self, McpError> {
        std::fs::create_dir_all(&config.cwd)
            .map_err(|e| McpError::Transport(format!("could not create {}: {e}", config.cwd.display())))?;
        let program = resolve_command(&config.command);
        let mut cmd = Command::new(&program);
        cmd.args(&config.args)
            .current_dir(&config.cwd)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for name in INHERITED_ENV {
            if let Some(v) = std::env::var_os(name) {
                cmd.env(name, v);
            }
        }
        for (k, v) in &config.env {
            cmd.env(k, v);
        }
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::Transport(format!("could not start {}: {e}", config.command)))?;
        let stdin = child.stdin.take().expect("piped stdin");
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut stderr_pipe = child.stderr.take().expect("piped stderr");

        let stdin = Arc::new(tokio::sync::Mutex::new(Some(stdin)));
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let closed: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let stderr_ring: Arc<Mutex<VecDeque<u8>>> = Arc::new(Mutex::new(VecDeque::new()));

        // stderr: the last 64 KB, for diagnostics
        {
            let ring = Arc::clone(&stderr_ring);
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                loop {
                    match stderr_pipe.read(&mut buf).await {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            if let Ok(mut r) = ring.lock() {
                                r.extend(&buf[..n]);
                                while r.len() > STDERR_RING_BYTES {
                                    r.pop_front();
                                }
                            }
                        }
                    }
                }
            });
        }

        // stdout: split into lines, answer pending requests, refuse what we cannot serve
        {
            let pending = Arc::clone(&pending);
            let closed = Arc::clone(&closed);
            let stdin = Arc::clone(&stdin);
            let ring = Arc::clone(&stderr_ring);
            let shared = Arc::clone(&shared);
            tokio::spawn(async move {
                let mut acc: Vec<u8> = Vec::new();
                let mut chunk = vec![0u8; 16 * 1024];
                let why = loop {
                    let n = match stdout.read(&mut chunk).await {
                        Ok(0) => break "the process ended".to_string(),
                        Ok(n) => n,
                        Err(e) => break format!("could not read from the process: {e}"),
                    };
                    acc.extend_from_slice(&chunk[..n]);
                    let mut too_long = false;
                    while let Some(pos) = acc.iter().position(|b| *b == b'\n') {
                        let line: Vec<u8> = acc.drain(..=pos).collect();
                        let text = String::from_utf8_lossy(&line);
                        let text = text.trim();
                        if text.is_empty() {
                            continue;
                        }
                        let Ok(value) = serde_json::from_str::<Value>(text) else {
                            continue; // not JSON (a server printing a banner): ignore
                        };
                        match react(&value, &shared) {
                            Reaction::Response { id, result } => {
                                let tx = pending.lock().ok().and_then(|mut p| p.remove(&id));
                                if let Some(tx) = tx {
                                    let _ = tx.send(result);
                                }
                            }
                            Reaction::Reply(reply) => {
                                if let Some(w) = stdin.lock().await.as_mut() {
                                    let mut line = reply.to_string();
                                    line.push('\n');
                                    let _ = w.write_all(line.as_bytes()).await;
                                    let _ = w.flush().await;
                                }
                            }
                            Reaction::Nothing => {}
                        }
                    }
                    if acc.len() > MAX_LINE_BYTES {
                        too_long = true;
                    }
                    if too_long {
                        break "a message from the process was larger than 8 MB".to_string();
                    }
                };
                // order matters: mark closed first, then fail what is waiting (a sender checks `closed` after registering)
                let tail = {
                    let r = ring.lock().map(|r| r.iter().copied().collect::<Vec<u8>>()).unwrap_or_default();
                    let s = String::from_utf8_lossy(&r).trim().to_string();
                    let start = s.char_indices().rev().nth(299).map(|(i, _)| i).unwrap_or(0);
                    s[start..].to_string()
                };
                let message = if tail.is_empty() { why } else { format!("{why} (last output: {tail})") };
                if let Ok(mut c) = closed.lock() {
                    *c = Some(message.clone());
                }
                let waiting: Vec<_> = pending.lock().map(|mut p| p.drain().collect()).unwrap_or_default();
                for (_, tx) in waiting {
                    let _ = tx.send(Err(McpError::Closed(message.clone())));
                }
            });
        }

        Ok(Self {
            stdin,
            pending,
            closed,
            child: tokio::sync::Mutex::new(Some(child)),
            stderr: stderr_ring,
        })
    }

    fn closed_reason(&self) -> Option<String> {
        self.closed.lock().ok().and_then(|c| c.clone())
    }

    async fn write(&self, message: &Value) -> Result<(), McpError> {
        let mut line = message.to_string();
        line.push('\n');
        let mut guard = self.stdin.lock().await;
        let w = guard
            .as_mut()
            .ok_or_else(|| McpError::Closed("the connection was closed".into()))?;
        w.write_all(line.as_bytes())
            .await
            .map_err(|e| McpError::Closed(format!("could not write to the process: {e}")))?;
        w.flush()
            .await
            .map_err(|e| McpError::Closed(format!("could not write to the process: {e}")))
    }
}

#[async_trait]
impl Transport for StdioTransport {
    async fn send_request(&self, id: u64, method: &str, params: Value) -> Result<Value, McpError> {
        if let Some(why) = self.closed_reason() {
            return Err(McpError::Closed(why));
        }
        let (tx, rx) = oneshot::channel();
        self.pending.lock().expect("pending").insert(id, tx);
        let _guard = PendingGuard { pending: Arc::clone(&self.pending), id };
        // the reader may have finished between the check and the insert: look again
        if let Some(why) = self.closed_reason() {
            return Err(McpError::Closed(why));
        }
        self.write(&rpc::request(id, method, params)).await?;
        match rx.await {
            Ok(r) => r,
            Err(_) => Err(McpError::Closed(
                self.closed_reason().unwrap_or_else(|| "the connection was closed".into()),
            )),
        }
    }

    async fn send_notification(&self, method: &str, params: Value) -> Result<(), McpError> {
        if let Some(why) = self.closed_reason() {
            return Err(McpError::Closed(why));
        }
        self.write(&rpc::notification(method, params)).await
    }

    fn set_protocol_version(&self, _version: &str) {}

    async fn close(&self) {
        if let Ok(mut c) = self.closed.lock() {
            if c.is_none() {
                *c = Some("the connection was closed".into());
            }
        }
        // closing stdin is how a well-behaved server is told to stop
        self.stdin.lock().await.take();
        if let Some(mut child) = self.child.lock().await.take() {
            if tokio::time::timeout(Duration::from_millis(800), child.wait()).await.is_err() {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        }
    }

    fn diagnostics(&self) -> String {
        let r = self
            .stderr
            .lock()
            .map(|r| r.iter().copied().collect::<Vec<u8>>())
            .unwrap_or_default();
        String::from_utf8_lossy(&r).into_owned()
    }
}
