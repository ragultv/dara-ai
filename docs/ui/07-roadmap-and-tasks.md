# Roadmap and tasks

Work one milestone at a time. Each ends with its acceptance criteria met, checks passing, a commit, and a review. Do not start the next milestone first.

## M0: Scaffold and the memory spike

- Create `app/` with Tauri 2 + Svelte 5 + Vite + TypeScript (strict) + Tailwind v4; ESLint, Prettier, Vitest, Playwright configured; `prebuild` copies `docs/ui/tokens.css`.
- Rust: spawn `dara.exe serve` as a sidecar on `127.0.0.1` with a random API key and free port; pass `{baseUrl, apiKey}` to the UI; stop it on quit; restart on crash (max 3 times, then show an error).
- UI: splash "Starting the engine", then show `GET /health` and the loaded model name.
- Fake server in `app/mocks` that serves `docs/ui/samples` (used by tests and dev without a GPU).
- Measure and record in `app/PERF.md`: idle RAM of the app (all processes) with and without the server; installer size; startup time.
- Acceptance: app starts the real server and shows its health; the same screen works against the mock; `npm run check/lint/test` pass; RAM numbers recorded. **Stop and report if UI RAM is over about 115 MB.**

## M1: Design system and shell

- Implement the components listed in 02 with both themes, keyboard and screen-reader support; a hidden `/#/kit` page showing every component in both themes.
- App shell: left rail, status pill, theme switch (System/Light/Dark), routing.
- Acceptance: axe clean on the kit page; contrast report still passes; no hex values outside tokens.

## M2: Models and first run

- Onboarding (first run), Models page: list, download with progress and cancel, load with progress, unload, delete.
- Handle errors from 04 (offline, 404 with download hint, out of disk).
- Acceptance: from an empty models folder, a user downloads Nano and ends in a loaded state (test with the real server on one machine, with the mock in CI).

## M3: Chat

- Conversation store (local), composer, streaming render with throttled Markdown, Stop, regenerate, copy, edit and resend, code block copy.
- Context meter, per-message stats, compaction note, per-request Flash and sampling controls, system prompt.
- Image attach (drag/drop/paste), thumbnail, vision-disabled handling.
- Acceptance: a 50-message conversation stays smooth (no jank while streaming); memory noted in `PERF.md`; XSS test payloads render inert; interrupted stream handled.

## M4: Options screen

- Everything in 01 page 3: K/V format control with quality labels, Flash toggle, result card, context ladder, recommended marker, measured vs not measured, "Measure on this PC" with progress and cancel, apply and load.
- Acceptance: matches `samples/options-nano.json` and a measured sample (record one with a real run); no computed numbers; both themes.

## M5: Monitor and Settings

- Monitor from `/dara/stats` with visibility-aware polling, sparkline, copy diagnostics (redacted).
- Settings: all of 01 page 5; network toggle forces API key; theme; start at login; tray.
- Acceptance: idle CPU near zero on every page; settings persist; diagnostics contain no usernames.

## M6: Hardening and packaging

- Security checklist (05) walk-through, CSP verified in the release build, capabilities minimal, devtools off, single instance.
- NSIS installer with the sidecar, code signing setup, CI job (build, test, audit).
- Accessibility and scaling pass (125/150 %, high contrast), performance pass.
- Acceptance: clean install on a second Windows machine works from first run to chat; release checklist done.

## Later (not in scope now)

Conversation search and export, prompt library, multiple windows, update channel, localization, voice input, per-conversation model settings, prompt-cache controls (after the backend adds them), saved presets.

## Backend items the UI team needs (owned by the backend, tracked in docs/BACKLOG.md)

1. `DARA_API_KEY` environment variable instead of only `--api-key` on the command line.
2. Host-header validation (DNS rebinding defense).
3. Public hosting for the Flash drafter downloads (currently from a private repository).
4. A graceful `POST /dara/shutdown` so the app can stop the server cleanly.

Raise any other missing endpoint as a backend request rather than working around it in the UI.
