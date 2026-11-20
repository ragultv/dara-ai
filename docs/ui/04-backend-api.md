# Backend API (what the UI calls)

Base URL `http://127.0.0.1:11435` (the sidecar may pick another port). Every request sends `Authorization: Bearer <key>` when the server runs with `--api-key` (the app always does). JSON in and out; errors always look like `{"error":{"message","type"[,"code","download"]}}`. Full server reference: [../SERVER.md](../SERVER.md). Types: [dara-api.d.ts](dara-api.d.ts). Real responses: [samples/](samples/) (use them as mocks and test fixtures; they were recorded from a real server with a GTX 1650).

Rules for the client:

- One place (`api/client.ts`) does fetch, auth, timeout and error mapping. Components never call `fetch`.
- Treat unknown JSON fields as allowed (the API grows); never require a field the types mark optional.
- A fresh `dara serve` has a model loaded already (the recommended configuration); the UI must still handle "no model loaded" (`/health` has no `model`) and "model loading".
- The server serves one request at a time; extra requests queue (`queue_ms`, `/health.queued`). Show "waiting" rather than an error.

## Endpoints used by the UI

| Page | Call | Notes |
| --- | --- | --- |
| Startup | `GET /health` | `{status, model?, busy, queued, loading?}`; poll every 300 ms until `status: "ok"` after spawning the sidecar |
| Startup, Models | `GET /v1/models` (same as `GET /dara/models`) | `data[]`: one per tier. `dara`: `status` (`not_installed`/`partial` (some files missing)/`installed`/`downloading`), `loaded`, `fits_gpu` (`full`/`partial`/`no`), `files[]` (`kind`: model/mmproj/flash, `installed`, `size_mb`), `flash`, `vision`, `config` (the loaded configuration and why), `download` (running job) |
| Models | `POST /dara/models/pull` | `{model, files?: ["model","mmproj","flash"], stream?}` returns a job `{id, ...}`; `stream: true` returns SSE progress |
| Models | `GET /dara/models/pull/{job}` / `DELETE` | progress `{status, file, files_done, files_total, completed_bytes, total_bytes, percent, mb_per_second, eta_seconds, error?}` / cancel (partial file kept for resume) |
| Models | `DELETE /dara/models/{id}` | deletes the model's files (confirm in the UI first) |
| Models, Options | `POST /dara/models/load` | `{model, kv?: "auto"\|"q8"\|"k8v4"\|"q4"\|"tq4", flash?: "auto"\|true\|false, vision?: boolean, ctx?: number, async?: true}`. Use `async: true` and poll `/dara/load`; a second load while loading returns 409. A bad `kv` returns 400 and the previous model stays loaded |
| Models | `POST /dara/models/unload` | frees the GPU |
| Load progress | `GET /dara/load` | `{state: idle\|queued\|loading\|ready\|failed, stage, percent, done?, total?, model, seconds, config{kv,flash,context,layers_on_gpu,head,vision,source,reason}, error?}` |
| Options | `GET /dara/models/{id}/options` | `?vision=false`, `?formats=q8,q4`. See below |
| Options | `POST /dara/models/{id}/benchmark` | `{formats?, vision?, force?}` returns `{id,...}`; one at a time (409 if running) |
| Options | `GET /dara/benchmarks/{job}` / `DELETE` | `{status: running\|done\|failed\|cancelled, log[], error?}` / cancel |
| Chat | `POST /v1/chat/completions` | OpenAI format; see below |
| Chat | `POST /dara/tokenize` | `{messages}` or `{text}` returns a token count (for the draft counter) |
| Monitor | `GET /dara/stats` | GPU, engine, model, ram, disk, downloads, last request, uptime |
| About | `GET /dara/version` | `{version, api, built, calibration_file}`; show `api` mismatch as "update the app" |
| Data (history, settings, per-model config) | `/dara/conversations`, `/dara/messages`, `/dara/attachments`, `/dara/settings`, `/dara/model-configs`, `/dara/search`, `/dara/store`, `/dara/data` | Full table, payloads and usage rules in [08-database.md](08-database.md); samples `conversation.json`, `conversations.json`, `settings.json`, `search.json`, `model-configs.json`, `store-info.json` |
| Dev only | `GET /metrics` | Prometheus text; not used by the UI |

### Chat request

```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": true,
  "max_tokens": 1024,
  "temperature": 0.7,
  "flash": true
}
```

`max_tokens`: no server-side cap other than the context; omitted means the server default (1024). Check `choices[0].finish_reason`: `length` = hit `max_tokens` or the end of the context. Optional: `model` (a Dara id loads/switches to it), `top_k`, `top_p`, `min_p`, `repeat_penalty`, `seed`, `stop`, `cache_write: false`. Image: a user message `content` array `[{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}, {"type":"text","text":"..."}]`. The API is stateless: every request carries the whole message history (including earlier images); the server reuses the shared prefix from its cache (see `cached_tokens`), so resending is cheap.

### Chat response and stream

Non-streaming and the last streaming chunk carry: `usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details{cached_tokens,image_tokens}`), `timings` (`queue_ms`, `compact_ms`, `image_ms`, `prefill_tokens`, `prefill_ms`, `prefill_tokens_per_second`, `ttft_ms`, `decode_tokens`, `decode_ms`, `decode_tokens_per_second`, `total_ms`, `image_memory`), `context` (`capacity`, `used`, `compacted`, `summarized_messages`, `dropped_messages`, `truncated_last_message`, `summary_cached`) and `speculative` (`mode`, `verify_steps`, `drafted`, `accepted`, `acceptance` percent, `model_passes`, `tokens_per_pass`, `note`, `flash_stood_down_at_pass`). Streaming: `data: {...chat.completion.chunk...}` lines (`choices[0].delta.content`), a final chunk with the fields above and `finish_reason`, then `data: [DONE]`. See [samples/chat-response.json](samples/chat-response.json) and [samples/chat-stream.sse](samples/chat-stream.sse).

`flash_stood_down_at_pass` non-null means Flash paused itself for this conversation because it was not paying off; show the `note` text in the message details, not as an error.

### Options response (page 3)

Top level: `model`, `name`, `family`, `native_context`, `machine{gpu, vram_total_mb, vram_free_assumed_mb, driver, dara}`, `vision{included, reserved_mb, note}`, `formats[]`, `options[]`, `recommended`, `calibration`, `flash`, `notes[]`.

- `formats[]`: `id` (`q8`, `k8v4`, `q4`, `tq4`, `k16v8`), `name`, `description`, `bits_per_value`, `caveat`, `recommendable`, `quality{label: high|medium|experimental|lower, text, checked_on, min_top1_of_128, worst_clear_errors_of_128, worst_kl}`.
- `options[]`: one per (format, flash): `{kv, flash, fast, points[]}`. `fast` and every `points[]` entry: `{context, layers, layers_gpu, head: "gpu"|"cpu", fully_on_gpu, all_layers_on_gpu, flash_active, kv_cache_mb, ok, speed}`; `speed` is `{source:"measured", decode_tps_start, decode_tps_full_estimate?, prefill_tps?, measured_at, power}` or `{source:"not_measured"}`.
- `recommended`: `{kv, flash, context, reason, rule}`: the server's choice; mark it in the UI with its `reason`.
- `calibration`: `{status: none|partial|measured|stale (measured on another GPU, driver or version), entries, fast_options, fast_options_measured, note, entries_measured_on_battery, run{method,url}}`. When `entries_measured_on_battery > 0` warn that the speeds were measured unplugged.

Sample: [samples/options-nano.json](samples/options-nano.json) (not measured yet) .

### Stats (Monitor)

`gpu` (`name`, `vram_total_mb`, `vram_used_mb`, `vram_free_mb`, `vram_used_percent`, `utilization_percent`, `temperature_c`, `power_w`, `power_limit_w`, `clock_sm_mhz`, `pstate`, `nvml`), `engine` (`requests`, `errors`, `busy`, `live_tokens_per_second`, `avg_decode_tokens_per_second`, `avg_prefill_tokens_per_second`, `compactions`, totals), `model` (the loaded model, `kv_format`, `layers_on_gpu`, `output_head`, `load_seconds`, `placement`), `context{capacity, held_tokens}`, `ram`, `process`, `disk` (`models_dir`, `volume_free_gb`, ...), `downloads[]`, `last_request`, `uptime_seconds`. GPU fields can be missing when NVML is unavailable (`nvml: false`): hide those cards instead of showing zeros. `model.path`, `mmproj` and `disk.models_dir` contain local paths: never include them in copied diagnostics or logs without removing the user's folder name.

## Errors

| Status | `error.type` | UI behavior |
| --- | --- | --- |
| 400 | `invalid_request_error` | show the message (e.g. unknown K/V format); a bug if the UI sent it |
| 401 | `authentication_error` | the key is wrong or missing: reconnect; if the app owns the sidecar, restart it |
| 404 | `model_not_found` | has a `download` hint: offer to download |
| 409 | `conflict` | a load or benchmark is already running: show its progress |
| 5xx | `server_error` | show the message with "Try again" and "Copy details" |

Connection refused or a closed stream mid-reply: mark the server offline, keep the partial reply in the thread labelled "interrupted", and offer retry.

## Not in the backend yet (do not build UI that depends on these)

- Host-header validation and an API key from the environment (planned; see [05-security.md](05-security.md)).
- Prompt cache controls beyond `cache_write: false` (see [../BACKLOG.md](../BACKLOG.md)).
- Flash drafters for the 9B, and public download hosting for the drafters (the `flash` file download may fail until released; show the server's error).
