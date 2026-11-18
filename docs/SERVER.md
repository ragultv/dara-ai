# dara serve

`dara serve` runs Dara's models behind an HTTP API: OpenAI-compatible chat and completions, a model library with downloads, and Dara's own
endpoints for development (memory, VRAM, disk, GPU telemetry, throughput). It is built into `dara.exe`; there is nothing else to install.

```
dara pull dara-core-4b          # model + image encoder + Flash drafter, verified by SHA-256
dara serve --model dara-core-4b
```

## Models

| Id | Model | Download (model / mmproj / drafter) | Flash |
| --- | --- | --- | --- |
| `dara-nano` | Gemma 4 E2B instruct, Q4_K_M | 2.9 GB / 0.9 GB / 35 MB | yes |
| `dara-core-4b` | Qwen3.5 4B, Q4_K_M | 2.6 GB / 0.6 GB / 82 MB | yes |
| `dara-pro-9b` | Qwen3.5 9B, Q4_K_M | 5.3 GB / 0.9 GB / - | no drafter yet |

Model and mmproj files come from pinned Hugging Face revisions (unsloth GGUF repositories); drafters from the Dara release
`flash-drafters-v2` (set `DARA_FLASH_BASE_URL` to download them from elsewhere). Short names work everywhere: `nano`, `4b`, `9b`.

Models live in `%LOCALAPPDATA%\Dara\models\<id>\` (`--models-dir DIR` or `DARA_MODELS` to change). The older layout
(`models\qwen3.5-4b-gguf\...`, from `bench\get_models.py`) is recognised too.

```
dara pull <id> [--no-vision] [--no-flash]   # resumable; parallel 16 MiB range requests
dara list
dara rm <id>
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--model <id or gguf>` | none | load this model at start. Without it the server starts empty and loads the model the first request names |
| `--models-dir DIR` | `%LOCALAPPDATA%\Dara\models` | the model library |
| `--host`, `--port` | `127.0.0.1`, `11435` | listen address |
| `--vision` / `--mmproj <gguf>\|auto` | off | image input. **Costs VRAM**: the encoder's room is reserved when the context is sized (4 GB card: the 4B drops from 32/32 to 15/32 layers on the GPU; Nano from 131K to 32K context) |
| `--flash auto\|on\|off\|<file>` | `auto` | Flash drafter. `auto` keeps it only if the model still fits fully in VRAM with it (4 GB card: on for Nano, off for the 4B, where it would cost layers and be slower) |
| `--no-spec` | | no speculative decoding at all |
| `--ctx auto\|N`, `--min-context N` | `auto`, `32768` | context size (as for `dara`) |
| `--image-max-tokens N` | `400` | Qwen image token cap |
| `--max-tokens N` | `1024` | default reply length when a request does not set one |
| `--api-key KEY` | | require `Authorization: Bearer KEY` |
| `--cors-origin ORIGIN` | localhost + Tauri origins | extra allowed browser origin |
Requests run one at a time (a queue); `/dara/stats` and `/metrics` answer while a request runs.

## Endpoints

| Route | |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI chat. `stream: true` gives SSE chunks ending with `data: [DONE]`. Images: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` parts, or Ollama-style `"images": [base64]` on a message. Sampling: `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `stop`, `max_tokens`. `"model"`: a Dara id loads / switches to it (404 `model_not_found` with a `download` hint if it is not installed; other names use the loaded model). `"flash": true\|false` turns speculative decoding on or off for the request |
| `POST /v1/completions` | raw prompt completion |
| `GET /v1/models` (also `/dara/models`) | every Dara model: install status, loaded, file sizes, `flash` {available, installed, active, note}, `vision`, whether it fits this GPU, a running download |
| `GET /v1/models/{id}` | one model |
| `POST /dara/models/pull` | `{"model": "dara-nano", "files": ["model","mmproj","flash"], "stream": false}` -> a download job (`stream: true`: SSE progress) |
| `GET` / `DELETE /dara/models/pull/{job}` | progress (bytes, %, MB/s, ETA) / cancel (the partial file is kept for resuming) |
| `POST /dara/models/load` | `{"model": "dara-core-4b", "flash": "auto"\|true\|false, "vision": false, "ctx": 32768}` |
| `POST /dara/models/unload` | free the GPU |
| `DELETE /dara/models/{id}` | delete a model's files |
| `GET /health` | liveness |
| `POST /dara/tokenize` | `{"messages": [...]}` or `{"text": "..."}` -> token count |
| `POST /dara/compact` | `{"messages": [...], "max_tokens": N}` -> the messages as the context maintainer would send them |
| `GET /dara/stats` | everything below, as JSON |
| `GET /metrics` | the same as Prometheus text |

### Per-request numbers

Every chat / completion response (and the final stream chunk) carries:

```json
"usage":   {"prompt_tokens": 53, "completion_tokens": 2, "total_tokens": 55,
            "prompt_tokens_details": {"cached_tokens": 24, "image_tokens": 0}},
"timings": {"queue_ms": 0, "compact_ms": 0, "image_ms": 0,
            "prefill_tokens": 29, "prefill_ms": 183.3, "prefill_tokens_per_second": 158.3, "ttft_ms": 186.5,
            "decode_tokens": 2, "decode_ms": 32.6, "decode_tokens_per_second": 30.7, "total_ms": 219.1},
"context": {"capacity": 32768, "used": 54, "compacted": false, "summarized_messages": 0, ...},
"speculative": {"mode": "ngram", "verify_steps": 0, "drafted": 0, "accepted": 0, "model_passes": 1, "tokens_per_pass": 2.0}
```

`prefill_tokens` counts only the tokens actually run (`cached_tokens` came from the prompt cache). `decode_tokens_per_second` is measured from
the first generated token to the last.

### /dara/stats

`process` (private working set, working set, peak, commit), `ram` (total / available / used), `gpu` (name, VRAM total / free / used, weights in
VRAM; with the NVIDIA driver's NVML: utilization, temperature, power, clocks, P-state), `disk` (model / mmproj / drafter file sizes, free space on
the model's volume), `model` (family, quant, layers on GPU, output head location, context, Flash, vision, placement), `context` (tokens held),
`engine` (busy, queued, live tokens and tokens/s of the running request, totals, average prefill and decode speed), `last_request`.

## Flash and speculative decoding

Prompt lookup (copying from the context) and the trained Flash drafter propose a few tokens; one model pass checks them all.

- **Greedy** (`temperature: 0`): a draft is kept while it equals the model's own choice, so the output is identical to plain decoding.
- **Sampled** (`temperature > 0`): both drafters are deterministic, so the exact speculative-sampling rule is used: draft token x is kept
  with probability p(x) under the request's sampling distribution (temperature, top_k, top_p, min_p, repeat penalty); on rejection the
  token is drawn from p with x removed. Outputs follow the same distribution as plain sampling; only the random stream differs, so a fixed
  `seed` gives different text with Flash on and off.

The response's `speculative` object reports the mode, drafted / accepted tokens, tokens per model pass and, when Flash was asked for but
not used, a `note` saying why.

## Prompt cache

The state of the previous request is kept on the GPU. Nano rewinds to the longest common prefix with the new prompt. Core / Pro keep a
VRAM checkpoint of the recurrent state at the end of the last conversation (before the reply), so a follow-up turn only runs the new messages.
Requests with images are not cached.

## Context maintainer

When `prompt + max_tokens + 256` does not fit the context, the oldest messages (after the system message) are summarized by the model itself
(greedy, at most 384 tokens) and the summary is added to the system message; the most recent messages, about half the context, stay verbatim.
Summaries are cached by the hash of the messages they cover, so the following turns reuse the same summary and the same prompt prefix (and
therefore the prompt cache) until the conversation outgrows the context again. A single message larger than the context is cut from the front.
The response's `context` object reports what happened. On a GTX 1650 the first compaction of a ~45K-token conversation took 153 s; the next turn
reused it (15,488 of 15,496 prompt tokens cached, 0.27 s to the first token).

## Test

```
python -I bench\serve_test.py [--image picture.png] [--compact]
```
