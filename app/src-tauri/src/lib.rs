pub mod connectors;
pub mod engine_settings;
pub mod sidecar;

use connectors::ConnectorsState;
use engine_settings::EngineSettings;
use serde::Serialize;
use sidecar::{ServerConnection, ServerManager};
use std::sync::Arc;

#[derive(Serialize)]
pub struct EngineSettingsInfo {
    /// What is stored in app.json (applies at the next engine start).
    pub saved: EngineSettings,
    /// What the engine is running with now; `models_dir` is always the real folder.
    pub running: EngineSettings,
    pub default_models_dir: String,
    pub restart_needed: bool,
}

fn engine_settings_info(manager: &ServerManager) -> EngineSettingsInfo {
    let saved = manager.saved_settings();
    let mut running = manager.running_settings();
    running.models_dir = running
        .effective_models_dir()
        .to_string_lossy()
        .into_owned();
    EngineSettingsInfo {
        restart_needed: !saved.same_as(&running),
        default_models_dir: engine_settings::default_models_dir()
            .to_string_lossy()
            .into_owned(),
        saved,
        running,
    }
}

#[tauri::command]
fn get_server_connection(
    manager: tauri::State<'_, Arc<ServerManager>>,
) -> Result<ServerConnection, String> {
    manager.get_connection()
}

#[tauri::command]
fn get_engine_settings(manager: tauri::State<'_, Arc<ServerManager>>) -> EngineSettingsInfo {
    engine_settings_info(&manager)
}

/// Validates and stores the engine settings. Nothing restarts; the UI offers "Restart engine".
#[tauri::command]
async fn set_engine_settings(
    manager: tauri::State<'_, Arc<ServerManager>>,
    settings: EngineSettings,
) -> Result<EngineSettingsInfo, String> {
    manager.save_settings(&settings)?;
    Ok(engine_settings_info(&manager))
}

/// Stops the engine and starts it with the saved settings. With `regenerate_key` a new API key is
/// created. Returns the connection to use afterwards (the port and key can change).
#[tauri::command]
async fn restart_engine(
    manager: tauri::State<'_, Arc<ServerManager>>,
    regenerate_key: Option<bool>,
) -> Result<ServerConnection, String> {
    manager.restart(regenerate_key.unwrap_or(false))
}

#[derive(Serialize)]
pub struct DriveInfo {
    /// "C:\\"
    pub root: String,
    pub free_gb: f64,
    pub total_gb: f64,
}

#[cfg(windows)]
mod drives {
    use super::DriveInfo;
    extern "system" {
        fn GetLogicalDrives() -> u32;
        fn GetDriveTypeW(root: *const u16) -> u32;
        fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    const DRIVE_FIXED: u32 = 3;

    pub fn list() -> Vec<DriveInfo> {
        let mut out = Vec::new();
        let mask = unsafe { GetLogicalDrives() };
        for i in 0..26u32 {
            if mask & (1 << i) == 0 {
                continue;
            }
            let root = format!("{}:\\", (b'A' + i as u8) as char);
            let wide: Vec<u16> = root.encode_utf16().chain(std::iter::once(0)).collect();
            if unsafe { GetDriveTypeW(wide.as_ptr()) } != DRIVE_FIXED {
                continue;
            }
            let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
            if unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut avail, &mut total, &mut free) } != 0 {
                let gb = 1024.0 * 1024.0 * 1024.0;
                out.push(DriveInfo { root, free_gb: avail as f64 / gb, total_gb: total as f64 / gb });
            }
        }
        out
    }
}

#[cfg(not(windows))]
mod drives {
    use super::DriveInfo;
    pub fn list() -> Vec<DriveInfo> {
        Vec::new()
    }
}

/// Fixed drives with their free space, so onboarding can suggest the one with room for the models.
#[tauri::command]
fn list_drives() -> Vec<DriveInfo> {
    drives::list()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // The 4 GB card belongs to the model and its K/V cache. Render the UI in software (CPU + system RAM) so the
    // WebView's GPU process never takes VRAM or decode time from the engine. Must be set before the webview starts.
    #[cfg(windows)]
    {
        const KEY: &str = "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS";
        let flags = "--disable-gpu --disable-gpu-compositing --disable-features=CalculateNativeWinOcclusion";
        let merged = match std::env::var(KEY) {
            Ok(v) if !v.contains("--disable-gpu") => format!("{v} {flags}"),
            Ok(v) => v,
            Err(_) => flags.to_string(),
        };
        std::env::set_var(KEY, merged);
    }

    let server_manager = ServerManager::new();
    let manager_for_setup = Arc::clone(&server_manager);
    let manager_for_exit = Arc::clone(&server_manager);
    // connectors (MCP servers): the manager closes their connections and child processes when the app exits
    let connectors = ConnectorsState::new();
    let connectors_for_exit = connectors.manager();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(server_manager)
        .manage(connectors)
        .invoke_handler(tauri::generate_handler![
            get_server_connection,
            get_engine_settings,
            set_engine_settings,
            restart_engine,
            list_drives,
            connectors::connectors_list,
            connectors::connector_add,
            connectors::connector_update,
            connectors::connector_remove,
            connectors::connector_connect,
            connectors::connector_disconnect,
            connectors::connector_tools,
            connectors::connector_set_permission,
            connectors::connector_call,
            connectors::connector_audit,
            connectors::connector_stderr
        ])
        .setup(move |_app| {
            manager_for_setup.start();
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(move |_app_handle, event| {
        if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
            manager_for_exit.stop();
            if let Some(m) = &connectors_for_exit {
                tauri::async_runtime::block_on(m.close_all());
            }
        }
    });
}
