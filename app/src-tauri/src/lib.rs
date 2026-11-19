pub mod sidecar;

use sidecar::{ServerConnection, ServerManager};
use std::sync::Arc;

#[tauri::command]
fn get_server_connection(
    manager: tauri::State<'_, Arc<ServerManager>>,
) -> Result<ServerConnection, String> {
    manager.get_connection()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_manager = ServerManager::new();
    let manager_for_setup = Arc::clone(&server_manager);
    let manager_for_exit = Arc::clone(&server_manager);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(server_manager)
        .invoke_handler(tauri::generate_handler![get_server_connection])
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
