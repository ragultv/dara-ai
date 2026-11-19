import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Ensure ~/.cargo/bin is in PATH so @tauri-apps/cli can invoke cargo
const cargoBin = path.join(os.homedir(), ".cargo", "bin");
const pathSep = process.platform === "win32" ? ";" : ":";
const currentPath = process.env.PATH || "";

if (!currentPath.split(pathSep).includes(cargoBin)) {
  process.env.PATH = `${cargoBin}${pathSep}${currentPath}`;
}

const args = process.argv.slice(2);
const tauriBin = process.platform === "win32" ? "tauri.cmd" : "tauri";
const localBin = path.resolve(__dirname, "..", "node_modules", ".bin", tauriBin);
const binToRun = fs.existsSync(localBin) ? `"${localBin}"` : tauriBin;

// Spawn local node_modules/.bin/tauri
const child = spawn(binToRun, args, {
  stdio: "inherit",
  shell: true,
  env: process.env
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
