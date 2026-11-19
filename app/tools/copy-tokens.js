import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Project root is two levels up from app/tools
const appDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(appDir, "..");
const srcTokensFile = path.resolve(repoRoot, "docs", "ui", "tokens.css");
const destTokensFile = path.resolve(appDir, "src", "tokens.css");

if (!fs.existsSync(srcTokensFile)) {
  console.error(`Error: Source tokens file does not exist at ${srcTokensFile}`);
  process.exit(1);
}

fs.mkdirSync(path.dirname(destTokensFile), { recursive: true });
fs.copyFileSync(srcTokensFile, destTokensFile);
console.log(`Copied ${srcTokensFile} -> ${destTokensFile}`);
