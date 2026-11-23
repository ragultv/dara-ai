use crate::engine_settings::{self, EngineSettings};
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
    /// What the running engine was started with (the port may differ from the saved one if it was busy).
    running: Mutex<EngineSettings>,
    restart_count: AtomicU32,
    is_stopping: AtomicBool,
    has_failed: AtomicBool,
    /// True while `restart_engine` swaps the child; the watchdog must not respawn then.
    is_restarting: AtomicBool,
    monitor_started: AtomicBool,
    restart_lock: Mutex<()>,
}

impl ServerManager {
    pub fn new() -> Arc<Self> {
        let mut settings = engine_settings::load_from(&engine_settings::config_path());
        // The saved port may be taken by another program: fall back to a free one for this run.
        if !engine_settings::is_port_free(settings.port, settings.allow_lan) {
            settings.port = Self::find_free_port();
        }
        let api_key = Self::generate_api_key();
        let binary_path = Self::locate_binary();

        let base_url = format!("http://127.0.0.1:{}", settings.port);
        let connection = ServerConnection { base_url, api_key };

        Arc::new(Self {
            connection: Mutex::new(connection),
            binary_path,
            child: Mutex::new(None),
            running: Mutex::new(settings),
            restart_count: AtomicU32::new(0),
            is_stopping: AtomicBool::new(false),
            has_failed: AtomicBool::new(false),
            is_restarting: AtomicBool::new(false),
            monitor_started: AtomicBool::new(false),
            restart_lock: Mutex::new(()),
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

        self.ensure_monitor();
    }

    fn ensure_monitor(self: &Arc<Self>) {
        if self.monitor_started.swap(true, Ordering::SeqCst) {
            return;
        }
        let manager = Arc::clone(self);
        std::thread::spawn(move || {
            manager.monitor_loop();
        });
    }

    /// What the engine is running with right now.
    pub fn running_settings(&self) -> EngineSettings {
        self.running.lock().map(|g| g.clone()).unwrap_or_default()
    }

    pub fn saved_settings(&self) -> EngineSettings {
        engine_settings::load_from(&engine_settings::config_path())
    }

    /// Validates and saves new engine settings. They apply at the next (re)start.
    pub fn save_settings(&self, input: &EngineSettings) -> Result<EngineSettings, String> {
        let running_port = self.running_settings().port;
        let valid = engine_settings::validate(input, running_port, |p| {
            engine_settings::is_port_free(p, input.allow_lan)
        })?;
        engine_settings::save_to(&engine_settings::config_path(), &valid)?;
        Ok(valid)
    }

    /// Stops the engine and starts it again with the saved settings (and a new key when asked).
    pub fn restart(self: &Arc<Self>, regenerate_key: bool) -> Result<ServerConnection, String> {
        let _guard = self.restart_lock.lock().map_err(|e| e.to_string())?;
        if self.is_stopping.load(Ordering::SeqCst) {
            return Err("The app is closing.".to_string());
        }
        self.is_restarting.store(true, Ordering::SeqCst);
        let result = self.restart_inner(regenerate_key);
        self.is_restarting.store(false, Ordering::SeqCst);
        result?;
        self.ensure_monitor();
        self.get_connection()
    }

    fn restart_inner(&self, regenerate_key: bool) -> Result<(), String> {
        let saved = self.saved_settings();
        let old_port = self.running_settings().port;
        // Check before stopping anything, so a bad value keeps the current engine alive.
        let valid = engine_settings::validate(&saved, old_port, |p| {
            engine_settings::is_port_free(p, saved.allow_lan)
        })?;

        self.kill_child();
        self.restart_count.store(0, Ordering::SeqCst);
        self.has_failed.store(false, Ordering::SeqCst);

        {
            let mut conn = self.connection.lock().map_err(|e| e.to_string())?;
            conn.base_url = format!("http://127.0.0.1:{}", valid.port);
            if regenerate_key {
                conn.api_key = Self::generate_api_key();
            }
        }
        {
            let mut running = self.running.lock().map_err(|e| e.to_string())?;
            *running = valid;
        }
        self.spawn_process()
    }

    fn kill_child(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait();
            }
            *guard = None;
        }
    }

    pub fn stop(&self) {
        self.is_stopping.store(true, Ordering::SeqCst);
        self.kill_child();
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

        let key = {
            let conn = self.connection.lock().map_err(|e| e.to_string())?;
            conn.api_key.clone()
        };
        let settings = self.running_settings();

        // Make sure the models folder exists on disk.
        let models_dir = settings.effective_models_dir();
        if let Err(e) = std::fs::create_dir_all(&models_dir) {
            eprintln!(
                "[Dara] Notice: Could not pre-create models directory {}: {e}",
                models_dir.display()
            );
        }

        // The key goes in the environment, not on the command line (visible in the process list).
        // Never log it.
        let mut cmd = Command::new(binary);
        cmd.args(settings.serve_args());
        cmd.env("DARA_API_KEY", &key);
        // The app is a GUI program with no console, so Windows would open a new terminal window for the console-based engine.
        // The engine runs in the background: its window is never created, and the app is the only thing the user sees.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }

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
            if self.is_restarting.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(200));
                continue;
            }

            let status_res = {
                let mut guard = match self.child.lock() {
                    Ok(g) => g,
                    Err(_) => break,
                };
                if let Some(ref mut child) = *guard {
                    child.try_wait()
                } else {
                    drop(guard);
                    std::thread::sleep(Duration::from_millis(400));
                    continue;
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

        // 2. The installed layout: the installer puts the engine (and its Flash drafters) in `engine` next to the app
        if let Ok(current_exe) = std::env::current_exe() {
            if let Some(p) = current_exe.parent().and_then(installed_engine) {
                if !is_self(&p) {
                    return Some(p);
                }
            }
        }

        // 3. Standard relative paths from workspace/execution root
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

        // 4. Search directory ancestors of current executable for build/cuda/dara.exe
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

/// `<app folder>/engine/dara.exe`, where the installer puts the engine; None when it is not there (a development run).
pub(crate) fn installed_engine(app_dir: &Path) -> Option<PathBuf> {
    let p = app_dir.join("engine").join("dara.exe");
    p.is_file().then_some(p)
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
    fn finds_the_engine_the_installer_put_next_to_the_app() {
        let dir = std::env::temp_dir().join(format!("dara-installed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(installed_engine(&dir), None);
        std::fs::create_dir_all(dir.join("engine")).unwrap();
        assert_eq!(installed_engine(&dir), None, "a folder alone is not an engine");
        std::fs::write(dir.join("engine").join("dara.exe"), b"x").unwrap();
        assert_eq!(installed_engine(&dir), Some(dir.join("engine").join("dara.exe")));
        let _ = std::fs::remove_dir_all(&dir);
    }

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
