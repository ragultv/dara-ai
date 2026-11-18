# dara serve

`dara serve` runs one model behind an HTTP API: OpenAI-compatible chat and completions, plus Dara's own endpoints for development
(memory, VRAM, disk, GPU telemetry, throughput). It is built into `dara.exe`; there is nothing else to install.

```
dara serve --model models\qwen3.5-4b-gguf\Qwen3.5-4B-Q4_K_M.gguf
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--model <gguf>` | (required) | Core 4B / Pro 9B (qwen35) or Nano (gemma4); the family is read from the GGUF |
| `--host`, `--port` | `127.0.0.1`, `11435` | listen address |
| `--mmproj <gguf>` / `auto` | off | image input. `auto` takes the single `mmproj*.gguf` next to the model. **Costs VRAM**: the encoder's room is reserved when the context is sized (4 GB card: the 4B drops from 32/32 to 15/32 layers on the GPU; Nano from 131K to 32K context) |
| `--flash auto\|off\|<file>` | `auto` | Flash drafter for greedy requests. `auto` finds the bundled drafter (`flash\` next to dara.exe) and keeps it only if the model still fits fully in VRAM with it |
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
| `POST /v1/chat/completions` | OpenAI chat. `stream: true` gives SSE chunks ending with `data: [DONE]`. Images: `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` parts, or Ollama-style `"images": [base64]` on a message. Sampling: `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `seed`, `stop`, `max_tokens` |
| `POST /v1/completions` | raw prompt completion |
| `GET /v1/models` | the loaded model |
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
