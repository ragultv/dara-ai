use rand::RngCore;
use serde::Serialize;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Serialize, Debug)]
pub struct ServerConnection {
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

pub struct ServerManager {
    connection: Mutex<ServerConnection>,
    binary_path: Option<PathBuf>,
    child: Mutex<Option<Child>>,
    restart_count: AtomicU32,
    is_stopping: AtomicBool,
    has_failed: AtomicBool,
}

impl ServerManager {
    pub fn new() -> Arc<Self> {
        let port = Self::find_free_port();
        let api_key = Self::generate_api_key();
        let binary_path = Self::locate_binary();

        let base_url = format!("http://127.0.0.1:{port}");
        let connection = ServerConnection { base_url, api_key };

        Arc::new(Self {
            connection: Mutex::new(connection),
            binary_path,
            child: Mutex::new(None),
            restart_count: AtomicU32::new(0),
            is_stopping: AtomicBool::new(false),
            has_failed: AtomicBool::new(false),
        })
    }

    pub fn get_connection(&self) -> Result<ServerConnection, String> {
        if self.has_failed.load(Ordering::SeqCst) {
            return Err("Engine crashed repeatedly (exceeded 3 restart attempts)".to_string());
        }
        let conn = self.connection.lock().map_err(|e| e.to_string())?;
        Ok(conn.clone())
    }

    pub fn start(self: &Arc<Self>) {
        if let Err(err) = self.spawn_process() {
            eprintln!(
                "[Dara] Notice: Sidecar binary not started ({err}). Proceeding in dev/mock mode."
            );
            return;
        }

        let manager = Arc::clone(self);
        std::thread::spawn(move || {
            manager.monitor_loop();
        });
    }

    pub fn stop(&self) {
        self.is_stopping.store(true, Ordering::SeqCst);
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
            }
            *guard = None;
        }
    }

    fn spawn_process(&self) -> Result<(), String> {
        let binary = match &self.binary_path {
            Some(path) => path,
            None => return Err("dara backend engine not found".to_string()),
        };

        // Guard: Prevent spawning the running executable as a sidecar
        if let Ok(current) = std::env::current_exe() {
            if let (Ok(c1), Ok(c2)) = (binary.canonicalize(), current.canonicalize()) {
                if c1 == c2 {
                    return Err(
                        "Refusing to spawn current executable as backend sidecar (anti-recursion guard triggered)"
                            .to_string(),
                    );
                }
            }
        }

        let (port_str, key_str) = {
            let conn = self.connection.lock().map_err(|e| e.to_string())?;
            let port = conn
                .base_url
                .split(':')
                .next_back()
                .unwrap_or("11435")
                .to_string();
            (port, conn.api_key.clone())
        };

        // Note: Never log key_str
        let mut cmd = Command::new(binary);
        cmd.args([
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            &port_str,
            "--api-key",
            &key_str,
        ]);

        let child = cmd.spawn().map_err(|e| {
            format!(
                "Failed to spawn sidecar binary at {}: {}",
                binary.display(),
                e
            )
        })?;

        let mut guard = self.child.lock().map_err(|e| e.to_string())?;
        *guard = Some(child);
        Ok(())
    }

    fn monitor_loop(&self) {
        loop {
            if self.is_stopping.load(Ordering::SeqCst) {
                break;
            }

            let status_res = {
                let mut guard = match self.child.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                if let Some(ref mut child) = *guard {
                    child.try_wait()
                } else {
                    break;
                }
            };

            match status_res {
                Ok(Some(exit_status)) => {
                    if self.is_stopping.load(Ordering::SeqCst) {
                        break;
                    }

                    let count = self.restart_count.fetch_add(1, Ordering::SeqCst) + 1;
                    if count <= 3 {
                        eprintln!(
                            "[Dara] Sidecar process exited unexpectedly ({exit_status}). Restarting ({count}/3)..."
                        );
                        std::thread::sleep(Duration::from_millis(500));
                        if let Err(e) = self.spawn_process() {
                            eprintln!("[Dara] Failed to restart sidecar: {e}");
                        }
                    } else {
                        eprintln!(
                            "[Dara] Sidecar process failed {count} times. Maximum restart limit exceeded."
                        );
                        self.has_failed.store(true, Ordering::SeqCst);
                        break;
                    }
                }
                Ok(None) => {
                    // Process still running
                }
                Err(err) => {
                    eprintln!("[Dara] Error waiting on sidecar process: {err}");
                    break;
                }
            }

            std::thread::sleep(Duration::from_millis(400));
        }
    }

    fn find_free_port() -> u16 {
        // Prefer default port 11435 if available
        if let Ok(listener) = TcpListener::bind("127.0.0.1:11435") {
            if let Ok(addr) = listener.local_addr() {
                return addr.port();
            }
        }
        // Fallback to ephemeral port assigned by OS
        if let Ok(listener) = TcpListener::bind("127.0.0.1:0") {
            if let Ok(addr) = listener.local_addr() {
                return addr.port();
            }
        }
        11435
    }

    fn generate_api_key() -> String {
        let mut bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut bytes);
        let mut s = String::with_capacity(64);
        for b in bytes {
            use std::fmt::Write;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }

    fn locate_binary() -> Option<PathBuf> {
        let current_exe_canon = std::env::current_exe()
            .ok()
            .and_then(|p| p.canonicalize().ok());

        let is_self = |p: &Path| -> bool {
            if let (Some(ref cur), Ok(cand_canon)) = (&current_exe_canon, p.canonicalize()) {
                return cur == &cand_canon;
            }
            false
        };

        // 1. Explicit override via environment variable
        if let Ok(env_path) = std::env::var("DARA_SIDECAR_PATH") {
            let p = PathBuf::from(env_path);
            if p.exists() && !is_self(&p) {
                return Some(p);
            }
        }

        // 2. Standard relative paths from workspace/execution root
        let candidates = [
            "build/cuda/dara.exe",
            "../build/cuda/dara.exe",
            "../../build/cuda/dara.exe",
            "../../../build/cuda/dara.exe",
            "../../../../build/cuda/dara.exe",
        ];

        for c in candidates {
            let p = Path::new(c);
            if p.exists() && !is_self(p) {
                return Some(p.to_path_buf());
            }
        }

        // 3. Search directory ancestors of current executable for build/cuda/dara.exe
        if let Ok(current_exe) = std::env::current_exe() {
            let mut ancestor = current_exe.parent();
            while let Some(dir) = ancestor {
                let candidate = dir.join("build").join("cuda").join("dara.exe");
                if candidate.exists() && !is_self(&candidate) {
                    return Some(candidate);
                }
                ancestor = dir.parent();
            }
        }

        None
    }
}

impl Drop for ServerManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_free_port() {
        let port = ServerManager::find_free_port();
        assert!(port > 0);
    }

    #[test]
    fn test_generate_api_key() {
        let key1 = ServerManager::generate_api_key();
        let key2 = ServerManager::generate_api_key();
        assert_eq!(key1.len(), 64);
        assert_eq!(key2.len(), 64);
        assert_ne!(key1, key2);
        assert!(key1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_server_manager_connection() {
        let manager = ServerManager::new();
        let conn = manager.get_connection().expect("connection must be ok");
        assert!(conn.base_url.starts_with("http://127.0.0.1:"));
        assert_eq!(conn.api_key.len(), 64);
    }

    #[test]
    fn test_locate_binary_never_matches_current_exe() {
        if let Some(binary) = ServerManager::locate_binary() {
            if let Ok(current) = std::env::current_exe() {
                let bin_canon = binary.canonicalize().unwrap_or(binary);
                let cur_canon = current.canonicalize().unwrap_or(current);
                assert_ne!(
                    bin_canon, cur_canon,
                    "locate_binary() must never return the running executable itself"
                );
            }
        }
    }
}
