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
| `--kv-format auto\|q8\|q4\|k8v4\|tq4` | `auto` | K/V cache format. `auto` = the configuration the options API recommends for this GPU (see "Defaults" below) |
| `--flash auto\|on\|off\|<file>` | `auto` | Flash drafter. `auto` = the recommendation (on only when it still keeps 32K tokens fully on the GPU: on for Nano, off for the 4B on a 4 GB card); `on` / `off` force it |
| `--ctx auto\|N` | `auto` | context size. `auto` = the largest that keeps every layer and the output head on the GPU, from the VRAM free at load time (the 9B, which cannot fit, gets 32K and the planner's placement) |
| `--vram-reserve MB` | `64` | VRAM the planner leaves free (0 = each model's own default: 128 MiB Qwen, 256 MiB Gemma). A smaller reserve gives a few thousand more tokens of context |
| `--no-vision` / `--mmproj <gguf>` | vision on | image input is on whenever the model's image encoder is installed (it costs a few MB and no context, see "Image input" below); `--no-vision` turns it off |
| `--no-spec` | | no speculative decoding at all |
| `--min-context N` | `0` | floor for `--ctx auto` (0: none; `dara --min-context` keeps its own default of 32768) |
| `--image-max-tokens N` | `400` | Qwen image token cap |
| `--max-tokens N` | `1024` | default reply length when a request does not set one |
| `--api-key KEY` | | require `Authorization: Bearer KEY`. The `DARA_API_KEY` environment variable does the same and is preferred (arguments are visible to other processes) |
| `--data-dir DIR` | `%LOCALAPPDATA%\Dara` (or `DARA_DATA_DIR`) | where the app database `dara.db` and its `attachments` folder live (chat history, settings, per-model configuration) |
| `--no-store` | | run without the app database (its endpoints answer 503) |
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
| `GET /dara/load` | progress of the current / last model load (stage, percent, error, chosen config) |
| `GET /dara/models/{id}/options` | what each K/V format, Flash setting and context size does on this GPU, with quality labels and measured speeds (below). `?vision=false`, `?formats=q8,q4` |
| `POST /dara/models/{id}/benchmark` | measure the speeds on this machine: `{"formats": ["q8","q4"], "vision": true, "force": false}` -> a job (`GET /dara/benchmarks/{job}` progress, `DELETE` cancels) |
| `GET /dara/version` | version and calibration file |
| `DELETE /dara/models/{id}` | delete a model's files |
| `GET /health` | liveness |
| `POST /dara/tokenize` | `{"messages": [...]}` or `{"text": "..."}` -> token count |
| `POST /dara/compact` | `{"messages": [...], "max_tokens": N}` -> the messages as the context maintainer would send them |
| `GET /dara/stats` | everything below, as JSON |
| `/dara/conversations`, `/dara/messages`, `/dara/attachments`, `/dara/settings`, `/dara/model-configs`, `/dara/search`, `/dara/store`, `/dara/data` | the app database: chat history, settings, per-model configuration, image attachments, full-text search, export and delete-all. Reference: [ui/08-database.md](ui/08-database.md); test: `tests/store_api_test.py` |
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

## Defaults, and load progress

When the K/V format and Flash are left on `auto` (the default), `dara serve` loads each model with the configuration the options API reports as
`recommended` for this GPU, computed from the VRAM that is free when the model loads; the context stays automatic. Anything you set explicitly wins, and the
rest is still recommended (`config.source` is `recommended`, `partly recommended` or `explicit`; `config.reason` says why). `/v1/models` shows the loaded
model's `config` (K/V format, Flash, context, layers on the GPU, output head, vision). On the GTX 1650: Nano loads INT8 + Flash with 131,072 tokens; the
Core 4B loads Q4 without Flash with about 35K tokens, every layer and the output head on the GPU. A load that fails after the old model was unloaded puts
the old model back; bad input (an unknown K/V format) is refused with a 400 before anything is unloaded.

Loading takes 5-15 s. `GET /dara/load` reports it: `state` (`idle`, `queued`, `loading`, `ready`, `failed`), `stage` (planning, loading layers 12/32, loading
the Flash drafter, loading the image encoder), `percent`, `seconds`, `error`, and the chosen `config`. `GET /health` carries the same as `loading` while one
runs. `POST /dara/models/load` waits by default; with `"async": true` it returns 202 at once and the load runs in the background (a second one is refused
with 409 until it finishes), so a UI can show the progress bar.

## Options: what fits this GPU, and how fast

`GET /dara/models/{id}/options` (CLI: `dara options <id>`) answers "what can I choose, and what will it do on my machine" for one model:

- **Placement is exact.** It runs the same placement planner a real load uses, in a dry run that stops before allocating anything (`ResidencyOptions::plan_only`),
  with the VRAM this process' own loaded model would give back counted as free. For every K/V format (INT8, INT8 keys + 4-bit values, Q4, TurboQuant), with
  and without the Flash drafter, it reports the **fast** point (the largest context with every layer and the output head on the GPU; when even a small
  context cannot be fully on the GPU, the planner's smallest placement, flagged by `fully_on_gpu: false`) and a ladder of contexts (4K ... 128K, native):
  context, layers on the GPU, where the output head runs, K/V cache size. On the 4B the dry run reproduced all 17 placements measured by real loads in the
  benchmark matrix (INT8 and Q4, Flash on and off, 16K to 128K) exactly.
- **Quality labels** come from the quality gate (`bench/quality_gate.py`: teacher-forced agreement and KL divergence against the FP32 reference on five
  workloads, via `tools/gen_quality_table.py` -> `src/serve/quality_table.h`): `high`, `medium` or `lower`, with the numbers. TurboQuant is listed but never
  recommended (no batched prompt-attention kernel yet).
- **Speeds are measured, once, on your machine** (`POST .../benchmark`, `dara bench <id>`; about 1.5 min for Nano, 8 min for the 4B): each distinct
  placement is loaded and timed (generation speed with an empty context, best of three; prompt speed on a ~600-token prompt) and saved to
  `%LOCALAPPDATA%\Dara\calibration.json` with the GPU, VRAM, driver and Dara version it is valid for (another machine, driver or version counts as not
  measured), the time, and the power state: a measurement taken on battery is flagged, because laptop GPUs run slower there. Until measured, an option says
  `"source": "not_measured"`; nothing is guessed. A short probe with ~6000 tokens held also gives `decode_tps_full_estimate`, an estimate of the speed with a
  full context (linear in the held context; within about 10% of the full fill-up curves measured on the GTX 1650). It is always plain decoding: with a long
  conversation Flash stands down, so that is the full-context speed with Flash on too.
- **A recommendation with its rule:** the highest-quality format that keeps at least 32K tokens fully on the GPU, otherwise the high-quality format that
  keeps the most layers on the GPU; Flash on only when it still keeps 32K tokens fully on the GPU. On the GTX 1650: Nano INT8 + Flash (119,808 tokens),
  Core 4B Q4 without Flash (29,696 tokens), Pro 9B Q4 (19 of 32 layers on the GPU; the response says it does not fit fully).
- The benchmark holds the engine while it runs (requests queue), unloads the current model for it and puts it back afterwards.

Measured on the GTX 1650 (on AC power), Core 4B, generation speed with an empty context:

| K/V | Flash | context | layers / head | tok/s | full-context estimate |
| --- | --- | --- | --- | --- | --- |
| INT8 | off | 17,408 (fast) | 32/32, GPU | 49.0 | 27.0 |
| INT8 | off | 32,768 | 32/32, CPU | 30.3 | |
| INT8 | off | 131,072 | 9/32, CPU | 9.8 | |
| INT8 | on | 5,120 (fast) | 32/32, GPU | 60.2 | |
| Q4 | off | 29,696 (fast) | 32/32, GPU | 48.9 | 20.5 |
| Q4 | off | 131,072 | 24/32, CPU | 18.8 | |
| Q4 | on | 8,192 (fast) | 32/32, GPU | 56.5 | |
| Q4 | on | 131,072 | 17/32, CPU | 13.2 | |

## Image input

The image encoder (the mmproj file) does not live in VRAM. Its weights stay in the memory-mapped file and are streamed to the GPU in 8 MiB row chunks
while an image is encoded (two slots, the next chunk copied while the current one is used: +2-3% encode time on a GTX 1650); the working memory of an
encode is carved from the part of the K/V caches the conversation has not reached yet (positions past the held length), and only what does not fit there
comes from free VRAM. Between images the encoder keeps 1-5 MB of VRAM.

| 4 GB GTX 1650 | encoder in VRAM (before) | streamed |
| --- | --- | --- |
| Nano + images: context | 32K | **119,808** (with the Flash drafter loaded) |
| Core 4B + images: layers on the GPU | 15 / 32 | **32 / 32** (same 32K context) |
| encoder VRAM kept between images: Nano / 4B / 9B | 291 / 632 / 867 MiB | 1.1 / 4.3 / 4.9 MiB |
| private RAM, Nano with images | 258 MB | 230 MB (the position tables are read from the mapped file) |
| encode time, 1920x1080 photo, Nano / 4B | 1.54 / 1.51 s | 1.57 / 1.58 s |

One image takes 110 MiB (Core 4B at the default 400-token budget), 120 MiB (Pro 9B) or 166 MiB (Nano) of working memory, plus 24 MiB for the weight
stream and 16 MiB of pinned host memory while it runs. Every response reports what its images used (`timings.image_memory`: scratch, from the unused
context, from free VRAM), `/dara/stats` the largest image (`model.image_scratch_max_mb`) and the resident part. If a conversation is so full and the
GPU so short of memory that the scratch cannot be found, the request fails with a message saying so (shorten the conversation, lower
`--image-max-tokens`, or load a smaller context).

**Images across turns (Nano).** The prompt cache covers image turns: when the conversation grows, the K/V of earlier images is reused (an image counts as
the same only if its pixels and position match, otherwise the cache is cut just before it) and only images past the reused prefix are encoded. Encoded
images are also kept in a small cache (8, by content hash), so re-sending a picture skips the encode. Measured on a GTX 1650 with Nano: a follow-up turn
after an image went from 1.8-5.4 s to 0.14-0.25 s time to first token, and a new image costs ~1.9 s however long the conversation is (before, each turn
re-encoded every earlier image and re-prefilled the whole history: 14-20 s at 7-9k tokens in the app). Core / Pro still re-prefill image conversations.

Embeddings are bit-identical to the resident encoder (tests/vision_stream_test.cu; the previous build's logits on image prompts match byte for byte).

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
