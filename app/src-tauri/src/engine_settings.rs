//! Engine settings that must be known before the server starts (port, models folder, LAN access,
//! VRAM reserve). They live in a small JSON file in the app config folder because the server's own
//! database does not exist yet when the sidecar is spawned. Pure logic here; no Tauri types.

use serde::{Deserialize, Serialize};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

pub const DEFAULT_PORT: u16 = 11435;
pub const DEFAULT_VRAM_RESERVE_MB: u32 = 64;
const MAX_VRAM_RESERVE_MB: u32 = 8192;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineSettings {
    pub port: u16,
    /// Empty string means "the default folder" (see `default_models_dir`).
    pub models_dir: String,
    pub allow_lan: bool,
    pub vram_reserve_mb: u32,
}

impl Default for EngineSettings {
    fn default() -> Self {
        Self {
            port: DEFAULT_PORT,
            models_dir: String::new(),
            allow_lan: false,
            vram_reserve_mb: DEFAULT_VRAM_RESERVE_MB,
        }
    }
}

/// Same rule as the server: DARA_MODELS, else %LOCALAPPDATA%\Dara\models.
pub fn default_models_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("DARA_MODELS") {
        if !custom.is_empty() {
            return PathBuf::from(custom);
        }
    }
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("Dara").join("models");
    }
    PathBuf::from("models")
}

/// Folder holding `app.json`: %LOCALAPPDATA%\Dara.
pub fn config_dir() -> PathBuf {
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local).join("Dara");
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("Dara");
    }
    PathBuf::from(".")
}

pub fn config_path() -> PathBuf {
    config_dir().join("app.json")
}

impl EngineSettings {
    /// The folder the engine will use (the default when `models_dir` is empty).
    pub fn effective_models_dir(&self) -> PathBuf {
        if self.models_dir.trim().is_empty() {
            default_models_dir()
        } else {
            PathBuf::from(&self.models_dir)
        }
    }

    /// True when the two settings start the engine the same way.
    pub fn same_as(&self, other: &EngineSettings) -> bool {
        self.port == other.port
            && self.allow_lan == other.allow_lan
            && self.vram_reserve_mb == other.vram_reserve_mb
            && self.effective_models_dir() == other.effective_models_dir()
    }

    /// `--host` value: loopback unless the user opted in to LAN access.
    pub fn host(&self) -> &'static str {
        if self.allow_lan {
            "0.0.0.0"
        } else {
            "127.0.0.1"
        }
    }

    /// Command-line arguments for `dara serve`. The API key is NOT here: it goes in the
    /// DARA_API_KEY environment variable because arguments are visible in the process list.
    pub fn serve_args(&self) -> Vec<String> {
        vec![
            "serve".to_string(),
            "--host".to_string(),
            self.host().to_string(),
            "--port".to_string(),
            self.port.to_string(),
            "--models-dir".to_string(),
            self.effective_models_dir().to_string_lossy().into_owned(),
            "--vram-reserve".to_string(),
            self.vram_reserve_mb.to_string(),
        ]
    }
}

/// Tolerant parser: a missing or wrongly typed field falls back to its default, so a hand-edited or
/// older file never stops the app from starting. Out-of-range values also fall back.
pub fn parse_settings(text: &str) -> EngineSettings {
    let mut s = EngineSettings::default();
    let value: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => return s,
    };
    if let Some(p) = value.get("port").and_then(|v| v.as_u64()) {
        if (1024..=65535).contains(&p) {
            s.port = p as u16;
        }
    }
    if let Some(d) = value.get("models_dir").and_then(|v| v.as_str()) {
        s.models_dir = d.to_string();
    }
    if let Some(b) = value.get("allow_lan").and_then(|v| v.as_bool()) {
        s.allow_lan = b;
    }
    if let Some(r) = value.get("vram_reserve_mb").and_then(|v| v.as_u64()) {
        if r <= MAX_VRAM_RESERVE_MB as u64 {
            s.vram_reserve_mb = r as u32;
        }
    }
    s
}

pub fn load_from(path: &Path) -> EngineSettings {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_settings(&text),
        Err(_) => EngineSettings::default(),
    }
}

/// Writes through a temporary file and renames it, so a crash never leaves a half-written file.
pub fn save_to(path: &Path, settings: &EngineSettings) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Cannot create the settings folder: {e}"))?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("Cannot write the settings file: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("Cannot save the settings file: {e}"))?;
    Ok(())
}

pub fn is_port_free(port: u16, allow_lan: bool) -> bool {
    let host = if allow_lan { "0.0.0.0" } else { "127.0.0.1" };
    TcpListener::bind((host, port)).is_ok()
}

/// Checks a settings object coming from the UI. `running_port` is the port the engine already
/// holds (always acceptable). `port_free` is injected so tests do not touch the network.
/// Creates the models folder when it does not exist yet.
pub fn validate(
    input: &EngineSettings,
    running_port: u16,
    port_free: impl Fn(u16) -> bool,
) -> Result<EngineSettings, String> {
    if !(1024..=65535).contains(&input.port) {
        return Err("The port must be between 1024 and 65535.".to_string());
    }
    if input.port != running_port && !port_free(input.port) {
        return Err(format!(
            "Port {} is already in use by another program.",
            input.port
        ));
    }
    if input.vram_reserve_mb > MAX_VRAM_RESERVE_MB {
        return Err(format!(
            "The VRAM reserve cannot be more than {MAX_VRAM_RESERVE_MB} MB."
        ));
    }
    let mut out = input.clone();
    out.models_dir = out.models_dir.trim().to_string();
    if !out.models_dir.is_empty() {
        if out.models_dir.contains('\0') {
            return Err("The models folder path is not valid.".to_string());
        }
        let p = Path::new(&out.models_dir);
        if !p.is_absolute() {
            return Err("The models folder must be a full path, such as D:\\models.".to_string());
        }
        if p.exists() {
            if !p.is_dir() {
                return Err("The models folder path points to a file, not a folder.".to_string());
            }
        } else {
            std::fs::create_dir_all(p)
                .map_err(|e| format!("Cannot create the models folder: {e}"))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_defaults_on_garbage() {
        assert_eq!(parse_settings("not json"), EngineSettings::default());
        assert_eq!(parse_settings("{}"), EngineSettings::default());
    }

    #[test]
    fn parse_reads_valid_fields_and_ignores_bad_ones() {
        let s = parse_settings(
            r#"{"port": 12000, "models_dir": "D:\\models", "allow_lan": true, "vram_reserve_mb": 128}"#,
        );
        assert_eq!(s.port, 12000);
        assert_eq!(s.models_dir, "D:\\models");
        assert!(s.allow_lan);
        assert_eq!(s.vram_reserve_mb, 128);

        let bad = parse_settings(
            r#"{"port": 80, "models_dir": null, "allow_lan": "yes", "vram_reserve_mb": 999999}"#,
        );
        assert_eq!(bad, EngineSettings::default());
    }

    #[test]
    fn validate_port_rules() {
        let mut s = EngineSettings::default();
        s.port = 80;
        assert!(validate(&s, 11435, |_| true).is_err());
        s.port = 12000;
        assert!(validate(&s, 11435, |_| false).is_err());
        // the port the engine already holds is accepted even when it looks busy
        s.port = 11435;
        assert!(validate(&s, 11435, |_| false).is_ok());
    }

    #[test]
    fn validate_models_dir_rules() {
        let mut s = EngineSettings::default();
        s.models_dir = "relative/path".to_string();
        assert!(validate(&s, 11435, |_| true).is_err());

        let base = std::env::temp_dir().join(format!("dara_es_test_{}", std::process::id()));
        let nested = base.join("a").join("b");
        s.models_dir = nested.to_string_lossy().into_owned();
        let ok = validate(&s, 11435, |_| true).expect("creatable folder is accepted");
        assert!(nested.is_dir());
        assert_eq!(ok.models_dir, s.models_dir);

        let file = base.join("file.txt");
        std::fs::write(&file, "x").unwrap();
        s.models_dir = file.to_string_lossy().into_owned();
        assert!(validate(&s, 11435, |_| true).is_err());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn serve_args_keep_loopback_and_never_carry_the_key() {
        let mut s = EngineSettings::default();
        s.models_dir = "D:\\models".to_string();
        let args = s.serve_args();
        let host_at = args.iter().position(|a| a == "--host").unwrap();
        assert_eq!(args[host_at + 1], "127.0.0.1");
        assert!(!args.iter().any(|a| a.contains("api-key")));
        assert!(args.contains(&"D:\\models".to_string()));
        s.allow_lan = true;
        let args = s.serve_args();
        let host_at = args.iter().position(|a| a == "--host").unwrap();
        assert_eq!(args[host_at + 1], "0.0.0.0");
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("dara_es_rt_{}", std::process::id()));
        let path = dir.join("app.json");
        let mut s = EngineSettings::default();
        s.port = 12345;
        s.allow_lan = true;
        save_to(&path, &s).unwrap();
        assert_eq!(load_from(&path), s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn same_as_treats_empty_dir_as_default() {
        let a = EngineSettings::default();
        let mut b = EngineSettings::default();
        b.models_dir = default_models_dir().to_string_lossy().into_owned();
        assert!(a.same_as(&b));
        b.port = 12000;
        assert!(!a.same_as(&b));
    }
}
