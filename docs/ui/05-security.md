# Security

The app is a local program that talks to a local server which runs the user's models and reads their files. The main risks are other things on the machine or in the user's browser reaching that server, and untrusted text (model output, web content pasted in) running code in the UI.

## Threat model

| Threat | Example | Defense |
| --- | --- | --- |
| Another local process or a web page calls the server | A malicious site makes `fetch("http://127.0.0.1:11435/...")`, or uses DNS rebinding to reach it | Bind to `127.0.0.1` only; random per-run API key required on every request; CORS allowlist; Host-header validation (backend gap, see below) |
| Network exposure | User sets `--host 0.0.0.0` | Off by default; the toggle in Settings forces the API key and warns; never default to it |
| XSS through model output | A reply contains `<img onerror=...>` or a `javascript:` link | Sanitize all rendered Markdown/HTML; strict CSP; no `innerHTML` with unsanitized text |
| A compromised WebView reaching the OS | XSS calls Tauri commands | Tauri capabilities with least privilege; no shell, no arbitrary fs, no arbitrary URLs |
| Secrets leaking | API key in logs, URLs, crash reports | Key only in memory and the `Authorization` header; never in URLs, localStorage, logs, or "Copy diagnostics" |
| Path/username leaks | Diagnostics or screenshots include `C:\Users\<name>` | Redact user folders in anything exported or copied |
| Malicious model files or updates | A tampered download | Downloads use pinned HTTPS URLs with a fixed revision; the server verifies each file's SHA-256; app updates signed |
| Prompt injection from pasted content | A document tells the model to act | The app has no tools that act on the system; the model's output is text only. Keep it that way; if tools are ever added, require per-action confirmation |

## Server connection

1. The Tauri side generates a 32-byte random key (CSPRNG) per run, starts `dara.exe serve --host 127.0.0.1 --port <p> --api-key <key>` and hands `{baseUrl, apiKey}` to the UI via one Tauri command. The key is never written to disk. Backend gap: the key is currently passed on the command line (visible to other processes of the same user in the process list); the server should read `DARA_API_KEY` from the environment. Until then, use it and track the gap (docs/BACKLOG.md).
2. Every `fetch` sends `Authorization: Bearer <key>`. `/health` too.
3. Allowed browser origins: `tauri://localhost`, `http(s)://tauri.localhost`, and the Vite dev origin in development only (`--cors-origin http://localhost:1420`). No wildcard, ever.
4. Backend gap: validate the `Host` header (accept only `127.0.0.1:<port>` and `localhost:<port>`) to stop DNS rebinding even if the key leaked. Track it; do not build around its absence.
5. If the user attaches to an external server (Settings), show a clear banner when the URL is not loopback or not HTTPS-or-loopback, and never send the key to a non-loopback `http://` URL without an explicit warning.

## Tauri hardening

- Capabilities (`src-tauri/capabilities/*.json`): grant only what is used: window controls, the tray, `core:event`, the sidecar spawn permission scoped to `dara.exe` with the fixed argument pattern, `dialog:open` for the models-folder picker, `autostart` if the setting exists, `opener` limited to `https://` links the user clicks. No `fs:*` wide scopes (scope to the app data and models directories), no `shell:execute` for arbitrary commands, no `http` plugin (the UI uses the WebView's fetch to loopback only).
- `tauri.conf.json` CSP (adjust ports as needed):
  `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' http://127.0.0.1:* http://localhost:*; object-src 'none'; base-uri 'none'; frame-src 'none'; form-action 'none'`
  (`'unsafe-inline'` for styles only, needed by some UI libraries' inline styles; never for scripts.) No remote scripts, fonts or images, ever.
- Disable devtools in release builds. Disable navigation and `window.open` except to allowed `https://` links through the opener plugin.
- Single instance: a second launch focuses the first window.
- Release builds are code-signed; updates (when added) are signed and verified by the updater.

## Rendering untrusted text

- Render Markdown with a library that does not emit raw HTML by default (e.g. `marked` with `html` tokens escaped, or `markdown-it` with `html: false`), then pass the result through **DOMPurify** with an allowlist (no `script`, `style`, `iframe`, `form`, `object`, event attributes; links only `https:`/`mailto:`; add `rel="noopener noreferrer"` and open externally via the opener plugin).
- Images from the model's Markdown (`![](url)`) are not fetched automatically (they would leak data to remote servers): render them as a link the user can open, or block them. Only user-attached local images and `data:`/`blob:` previews are displayed.
- Code blocks are text (`textContent`), never HTML from the model.
- User-supplied file names and model names are text nodes, never HTML.

## Data and privacy

- Everything stays on the machine. No analytics, crash reporting or update pings without an explicit, off-by-default setting that says what is sent.
- Conversations, settings and images are stored in the SQLite database and attachments folder under `%LOCALAPPDATA%\Dara` (see 08-database.md), readable only by the user (default Windows ACLs). The WebView has no SQL or file-system access; the Rust commands validate every input. "Delete all" removes them for real (files and database), with a confirm.
- Do not log message content or the API key, even in development logs. Logs hold events and error codes only.
- Copied diagnostics (Monitor, errors) contain versions, GPU name, VRAM, configuration and error text; paths are reduced to file names.

## Dependencies and supply chain

- Pin versions (lockfile committed, exact versions for security-sensitive packages: DOMPurify, the Markdown parser). `npm audit` and `cargo audit` run in CI and block on high severity.
- Few dependencies: justify each in the PR. No packages that run install scripts unless reviewed.
- Build in CI from a clean checkout; verify the sidecar binary hash recorded in the release.

## Input validation in the UI

The server validates requests, but the UI also checks: image type and size (png/jpeg/webp, limit 20 MB), numeric ranges for sampling settings (temperature 0 to 2, max tokens at least 1), port range, folder path exists and is writable. Validation messages say what to fix.

## Security checklist for every PR touching network, rendering or Tauri config

- [ ] No new Tauri permission without a reason in the PR description
- [ ] No `dangerouslySetInnerHTML` / `innerHTML` with anything but sanitized output
- [ ] No secrets or message content in logs, URLs or storage
- [ ] New remote host? There should be none; if so, it needs sign-off and a CSP change
- [ ] `npm audit` / `cargo audit` clean
