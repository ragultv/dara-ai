pub mod engine_settings;
pub mod sidecar;

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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_manager = ServerManager::new();
    let manager_for_setup = Arc::clone(&server_manager);
    let manager_for_exit = Arc::clone(&server_manager);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(server_manager)
        .invoke_handler(tauri::generate_handler![
            get_server_connection,
            get_engine_settings,
            set_engine_settings,
            restart_engine
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
        }
    });
}
