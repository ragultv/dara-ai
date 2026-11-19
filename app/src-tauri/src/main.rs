// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    // Guard A: Anti-recursion defense.
    // If this desktop GUI binary is invoked with server CLI arguments,
    // immediately terminate to prevent recursive fork-bomb execution.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| {
        arg == "serve"
            || arg == "--host"
            || arg.starts_with("--host=")
            || arg == "--api-key"
            || arg.starts_with("--api-key=")
            || arg == "--port"
            || arg.starts_with("--port=")
    }) {
        eprintln!(
            "[Dara Desktop Error] The GUI executable cannot be run as the backend server. \
             The inference engine must be the C++/CUDA executable at build/cuda/dara.exe. Terminating."
        );
        std::process::exit(1);
    }

    dara_lib::run();
}
