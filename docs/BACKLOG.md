# Backlog

Planned work that is designed but not scheduled. Each item says what it is, why, and how it would be built.

## Prompt cache with explicit read / write (Claude-style)

**Today:** `dara serve` keeps one implicit cache slot: the state of the previous request stays on the GPU and the next request reuses the
common prefix (Nano: rewind to the longest common prefix; Core / Pro: a VRAM checkpoint of the recurrent state at the end of the last
conversation). Anything else overwrites it.

**Goal:** many cache entries that survive across requests, conversations and server restarts, addressed with Claude-style breakpoints and
reported with Claude-style usage fields.

### API
- `cache_control: {"type": "ephemeral", "ttl": "5m" | "1h"}` on any message or content part marks a breakpoint: "save the state up to here".
- `"cache": "auto"` (default) keeps today's behaviour plus automatic breakpoints at the end of the system prompt and at the end of the
  conversation; `"cache": "off"` disables both reading and writing for the request.
- Usage: `cache_creation_input_tokens`, `cache_read_input_tokens`, `input_tokens` (uncached), alongside OpenAI's
  `prompt_tokens_details.cached_tokens`. Timings: `cache_read_ms`, `cache_write_ms`.
- Minimum cacheable prefix 1024 tokens (configurable), so small prompts do not churn the cache.
- `GET /dara/cache`: entries, sizes, hits / misses, bytes per tier. `DELETE /dara/cache`: clear.

### What an entry is
- Qwen (Core / Pro): the DeltaNet recurrent state (fixed, ~50 MB on the 4B) plus the attention layers' INT8 K/V and scales for positions
  [0, n) (~18 KB per token on the 4B: 10K tokens ~ 230 MB), plus the CPU-side state of host-resident layers.
- Nano: the INT8 K/V (and scales) of the layers that own K/V, positions [0, n).
- Key: hash chain of the token ids up to the breakpoint + model file identity (size + SHA-256 prefix) + K/V format + cache format version.
  Image bytes are part of the hash, so prefixes with images are cacheable (the M-RoPE positions are a pure function of the content).

### Tiers
1. VRAM: the live state (as today).
2. Host RAM: pinned pool, **off by default** (300 MB RAM budget); `--cache-ram N`.
3. Disk: `%LOCALAPPDATA%\Dara\cache`, default budget 4 GB (`--cache-disk N`), survives restarts. Reading 230 MB from an SSD is ~0.1-0.2 s;
   re-prefilling 10K tokens on a GTX 1650 is ~10 s.

Writes happen at breakpoints during prefill: device -> pinned staging, then a background thread writes the file (generation does not wait).
Reads pick the longest matching breakpoint, upload it, and prefill only the rest. TTL refreshed on every read; LRU eviction within each
tier's budget.

### Model-side work
- `QwenModel::export_state(n, sink)` / `import_state(source)`: DeltaNet conv + S states, attention K/V [0, n) with INT8 scales, hosted layers'
  CPU states (`cpu_s_`), and the M-RoPE image spans.
- `GemmaModel::export_kv(n, sink)` / `import_kv(source)` (+ soft-token spans).
- File format `DCACHE01`: header (key, model identity, token count, format), per-layer sections, checksum. A file that does not match or
  fails its checksum is ignored and deleted, never trusted.

### Validation
- Restoring a breakpoint must give the same greedy tokens as running the same prompt uncached with the prefill split at the same point
  (4B and Nano), and `bench/quality_gate.py` must still pass.
- Measure read / write times per tier and the TTFT win on long system prompts and long documents.

## Other items

- **Desktop app (Tauri):** Dara's own UI on top of `dara serve` (model library with downloads, chat, the `/dara/stats` dashboard). UI
  budget ~75-115 MB so server + UI stay under 300 MB.
- **Flash drafter for Pro 9B:** none is trained yet; 9B greedy requests use prompt lookup only.
- **Drafter hosting:** the drafters are GitHub release assets of a private repository, so anonymous downloads fail; publish them (public
  release or a Hugging Face repo) and set the catalog URL (or `DARA_FLASH_BASE_URL`).
- **Vision encoder on tensor cores** (attention GEMMs) for faster image TTFT on sm_80+.
- **Nano small-op fusion** (decode launch overhead).
- **Tokenizer memory** (~30-40 MB of the private working set).
- **Images when the context is nearly full:** the encode borrows the unused tail of the K/V caches, which is cut into small pieces per layer; when
  the conversation fills the context the big buffers come from free VRAM instead. Splitting the largest working buffers (MLP activations, q/k/v) so they
  fit the pieces would make a full context cheaper still. A CPU fallback for encoding would remove the last failure case.
- **Server hardening for the desktop app:** read the API key from `DARA_API_KEY` (today only `--api-key`, visible in the process list); validate the
  `Host` header (accept only `127.0.0.1:<port>` / `localhost:<port>`) against DNS rebinding; `POST /dara/shutdown` for a graceful stop from the app.
