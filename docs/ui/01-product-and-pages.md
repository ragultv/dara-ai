# Product and pages

## What Dara Desktop is

A local, private chat app for the Dara models (Nano, Core 4B, Pro 9B) on the user's own Windows GPU. It has the same feature set as other local-model apps (run a local server, manage models, chat, use images) but its own identity: it explains *what fits your GPU and how fast it will be*, using measured numbers.

Users: people with a 4 to 8 GB laptop GPU who do not want a terminal. Many have 4 GB cards, so context length and K/V choices matter and must be explained in plain words.

Voice: plain, direct, no hype, no jargon without a one-line explanation. Say "tokens per second" and "context" once with a hint; never "KV cache" without "memory for the conversation".

## Navigation

Left rail (icons + labels, collapses to icons below 900 px): **Chat**, **Models**, **Monitor**, **Settings**. A status pill in the footer shows the loaded model, context and a live tok/s while generating. System tray icon: open, load/unload model, quit.

## Pages

Every page needs these states: loading (skeleton, not spinner-only), empty, error (with a retry and the server's message), and offline (server not running).

### 0. Startup and first run

- The app starts the sidecar server (see 03). While `GET /health` fails, show a splash with the Dara mark and "Starting the engine".
- If no model is installed (`GET /v1/models`, none with `status: installed`), show onboarding: three cards (Nano / Core 4B / Pro 9B) with `description`, `download_size_mb`, `fits_gpu` and a "Recommended" badge picked from `fits_gpu`; one primary button "Download" (`POST /dara/models/pull`) with progress (bytes, %, MB/s, ETA) and a cancel.
- After the download, load it (`POST /dara/models/load`) and show load progress from `GET /dara/load` (poll every 300 ms); then go to Chat.
- If the server fails to start, show the stderr tail and buttons: "Try again", "Copy details".

### 1. Chat

- Layout: conversation list (left, collapsible), thread (center), optional details drawer (right).
- Composer: multi-line, Enter sends, Shift+Enter newline, attach-image button (png/jpeg/webp, drag and drop, paste), stop button while generating, model chip showing the loaded model (click opens the model switcher).
- Streaming: `POST /v1/chat/completions` with `stream: true`; render tokens as they arrive; the final chunk carries `usage`, `timings`, `context`, `speculative`.
- Per message (small, muted, on hover/focus or in the drawer): tokens, tok/s (`timings.decode_tokens_per_second`), time to first token (`ttft_ms`), cached tokens (`usage.prompt_tokens_details.cached_tokens`), Flash on/off and acceptance (`speculative`).
- Context meter above the composer: `context.used / context.capacity`; warning color at 85 %. When `context.compacted` is true, show a quiet note: "Older messages were summarized to fit" with the counts (`summarized_messages`, `dropped_messages`).
- Per-request switches in a "Generation" popover: Flash on/off (`flash`), temperature, max tokens, system prompt. Defaults come from Settings.
- Markdown rendering with code blocks (copy button), sanitized (see 05). Regenerate, copy, edit-and-resend on user messages.
- Conversations are stored locally (see 03). Title = first user message (truncated); rename and delete.
- Image attach: show a thumbnail chip; if the loaded model has `vision.active` false, disable with a tooltip "This model was loaded without image support" and a button to reload with vision.
- Errors: 404 `model_not_found` includes a `download` hint; offer "Download model". Busy: show "The engine is busy" and retry. Never lose the user's draft on error.

### 2. Models

- Grid of the three tiers from `GET /v1/models`: name, description, quant, sizes, status (not installed / downloading / installed / loaded), `fits_gpu` (`full` / `partial` / `no`) with a plain-words explanation, vision and Flash badges.
- Actions: Download (choose files: model, image support `mmproj`, Flash drafter `flash`), Load, Unload, Delete (confirm; shows disk freed), "Options" (opens page 3).
- Download rows: progress, speed, ETA, cancel (cancel keeps the partial file for resuming; say so).
- Disk summary from `/dara/stats` `disk`: models dir, free space. Block a download if free space is lower than the size, with the numbers.
- The 9B has no Flash drafter yet; show "Flash: not available for this model".

### 3. Model options (the signature screen)

Opened from a model card. Source: `GET /dara/models/{id}/options`. Purpose: let the user choose K/V format and Flash and see, for *this* GPU, the max context and the speed.

- Header: GPU name and VRAM (from `machine`), and a warning if measurements were taken on battery.
- Segmented control: **K/V format** (from `formats`): each shows `name`, `quality.text` (High / Medium / Experimental / Lower), `description`, `caveat`. Quality labels use colors from the design system and also text, never color alone.
- Toggle: **Flash** (faster generation; it uses some memory so the context shrinks; disabled with a reason when not installed or unavailable).
- Result card for the chosen combination (the `options[]` entry for that kv + flash): "Fastest setup: N tokens of context, X tok/s" (`fast`), plus a context ladder (`points[]`: 4K, 8K, 16K ... max) showing for each whether it fits fully on the GPU or some layers go to the CPU, and the speed if measured. Mark the recommended option (`recommended`) with its `reason`.
- Speeds: `speed.source === "measured"` shows tok/s (and `decode_tps_full_estimate` as "about Y when the context is full"); `not_measured` shows "Not measured yet" and a "Measure on this PC" button (`POST /dara/models/{id}/benchmark`, then poll `GET /dara/benchmarks/{job}` and show its `log` and `status`). Measuring takes several minutes and uses the GPU: ask for confirmation, advise plugging in the laptop, allow cancel.
- Apply: "Load with these settings" calls `POST /dara/models/load` with `{model, kv, flash, ctx?}`; show progress from `GET /dara/load`; on failure show the error and say the previous model was restored.
- Never display a derived number the API did not give. Do not compute "2x context" in the UI; show the numbers.

### 4. Monitor

Live from `GET /dara/stats` (poll every 1 s when visible, pause when hidden). Cards: GPU (name, utilization, VRAM used/total with a bar, temperature, power, clocks), engine (requests, errors, average prefill/decode tok/s, live tok/s while busy), model (loaded config, context, layers on the GPU, load time), downloads, disk. A small tok/s sparkline for the session (keep 120 points in memory). A "Copy diagnostics" button builds a text block from stats + version (no prompts, no file paths containing a username).

### 5. Settings

- General: theme (System / Light / Dark), start at login, minimize to tray.
- Server: port (default 11435), models folder (folder picker), "Allow other devices on my network" (off; explain the risk; when on, require the API key and show it), API key reveal/copy/regenerate.
- Defaults: K/V format (Auto), Flash (Auto), VRAM reserve (advanced, default 64 MB, explain), vision on/off, default temperature, max tokens, system prompt.
- Data: conversations location, export all (JSON), delete all (confirm).
- About: app version, `GET /dara/version`, links to docs, licenses.

### 6. Tray and system

Tray menu: Open Dara, current model (click to unload), Load model > (list), Quit (stops the sidecar). Closing the window minimizes to tray when enabled. On quit, wait for the current generation to finish or cancel it before stopping the server.

## Empty and error copy (examples)

- No model: "Download a model to start. Nano is the quickest to try."
- Offline: "The engine isn't running." [Start it] [Show details]
- GPU lost: "The GPU isn't available. If your laptop slept, restart Dara." (the server reports an error message; show it).
- Out of space: "Not enough disk space: needs 2.9 GB, 1.1 GB free."
