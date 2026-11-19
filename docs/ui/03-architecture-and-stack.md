# Architecture and stack

## Decisions (already made; do not relitigate without a measurement)

| Choice | Why |
| --- | --- |
| Tauri 2 (Rust shell, system WebView2) | A window costs tens of MB, not the 150+ MB of a bundled Chromium. The 300 MB budget (server + UI) rules out Electron |
| React 19 + Vite + TypeScript (strict) | Largest ecosystem and component choice, best tooling and AI-agent support; the runtime is about 45 KB gzipped, small next to the WebView. Chosen by the project owner over Svelte; Milestone 0 measures its RAM cost |
| Tailwind CSS v4 at build time, tokens via `@theme` mapping to `tokens.css` | Utility classes without a runtime; tokens stay the single color source |
| System font stack | No font files to load |
| No UI kit with a large runtime (no MUI, no Ant); accessible headless primitives are allowed and recommended for Dialog/Popover/Menu/Tooltip/Select (Radix UI or React Aria), plus Zustand for state; keep the tree of dependencies small | Memory and bundle size |
| Local storage in SQLite via `tauri-plugin-sql`, or JSON files for the first milestone | Conversations must survive restarts; keep it simple |

Budget: server + UI under 300 MB RAM total. Measure in Milestone 0 (see 07) with the WebView idle on Chat and again after a 50-message conversation; record the numbers in `app/PERF.md`. If the UI exceeds about 115 MB, stop and report before adding features.

## Process model

```
Dara.exe (Tauri)                    dara.exe serve  (sidecar, same install folder)
 ├─ Rust: spawn/stop sidecar        ├─ OpenAI-compatible API  http://127.0.0.1:11435
 │        pick port, API key        ├─ /dara/* (models, options, stats, load progress)
 │        tray, autostart           └─ GPU, models, K/V cache
 └─ WebView (React UI) ── fetch ──▶ server
```

- The Rust side starts `dara.exe serve --host 127.0.0.1 --port <p> --api-key <random>` as a Tauri sidecar, passes `{baseUrl, apiKey}` to the UI through a Tauri command (never through the URL, never logged), watches the process, and kills it on quit. If the port is busy, pick the next free port.
- If a Dara server is already running (user started it), the UI may attach to it: ask for the base URL and key in Settings. The sidecar is the default.
- The UI talks to the server only with `fetch` and Server-Sent Events. No Tauri command carries chat data.
- The server loads the recommended configuration by default (`--kv-format auto --flash auto`). The UI changes it only through `/dara/models/load` when the user chooses to.

## Folder layout (inside `app/`)

```
app/
  AGENTS.md                 agent guidance (points to docs/ui)
  PERF.md                   measured memory/startup numbers
  package.json, vite.config.ts, tsconfig.json, vite.config.ts uses @vitejs/plugin-react
  src-tauri/                Rust: sidecar, tray, capabilities, tauri.conf.json
  src/
    app.css                 @import docs tokens + Tailwind theme mapping
    lib/
      api/                  client.ts (fetch wrapper), sse.ts, types.ts (from docs/ui/dara-api.d.ts), errors.ts
      stores/               server.ts (connection), models, load, stats, settings, conversations (Zustand stores)
      ui/                   design-system components (Button, Dialog, ...)
      features/             chat/, models/, options/, monitor/, settings/, onboarding/
      markdown/             sanitized renderer
      util/                 format.ts (bytes, tok/s, durations), a11y helpers
    routes/ (or screens/)   one file per page
  tests/                    unit (Vitest), component (Testing Library), e2e (Playwright against a fake server)
  mocks/                    fake server serving docs/ui/samples
```

Keep `tokens.css` in `docs/ui/` as the source; copy it into the app at build time (a `prebuild` script) so there is one generator and no drift.

## State

- Zustand stores (one per concern) for server data and settings; local component state with `useState`/`useReducer`. No global event bus. Components read stores through narrow selectors (`useStore(s => s.x)`), never the whole store, so a change re-renders only what uses it.
- Never keep high-frequency data (streaming tokens, live tok/s, stats ticks) in a store that many components select from. The streaming message has its own small store/subscription and only the streaming message component re-renders; finished messages are `React.memo`'d and keyed by id.
- Connection store: `{status: 'starting'|'ready'|'offline'|'error', baseUrl, apiKey}`. Every request goes through `api/client.ts`, which adds the `Authorization: Bearer` header, a timeout, an `AbortSignal`, and turns error bodies `{error:{message,type,code?,download?}}` into a typed `ApiError`.
- Polling: `/dara/stats` every 1 s only while the Monitor page or the footer pill is visible; `/dara/load` every 300 ms only during a load; `/dara/models/pull/{job}` every 500 ms (or the SSE stream) during a download; stop on `document.hidden`.
- Streaming chat: read the SSE with `fetch` + `ReadableStream`; batch updates with `requestAnimationFrame` (not one React render per token); keep the raw text in a plain string and render Markdown on a throttle (every 50 to 80 ms) to avoid O(n^2) re-parsing in long replies; abort with `AbortController` on Stop (the server stops generating when the connection closes).
- Conversations: `{id, title, createdAt, messages[{id, role, content, images?, stats?}], settings}`. Store images as files, not base64 in the database. Keep only the active conversation's messages in memory; load others on demand.
- Long lists (long conversations, model logs) must be virtualized or windowed once past about 200 items.

## Performance rules

- Idle UI uses no CPU: no timers while hidden, no animation loops, no polling when nothing is visible.
- Lazy-load routes; the Chat route loads first. Markdown and syntax highlighting load on first use; highlight only finished code blocks, with a small highlighter (e.g. `shiki` with only the needed languages, or `highlight.js` core + a few).
- Images: create thumbnails (max 256 px) for the UI; send the original to the server; revoke object URLs.
- Bundle budget: JS under 400 KB gzipped for the first route. Check with `vite build` report in CI.

## Packaging

Tauri bundles an NSIS installer. The sidecar `dara.exe` is shipped next to the app; models are not bundled (downloaded on first run to `%LOCALAPPDATA%\Dara\models`). CI builds on the existing Windows CUDA runner; the UI build needs no GPU.
