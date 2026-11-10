# Dara Runtime: Technical Design and Development Plan

Nov 8, 2026 · @ragul

## Purpose and scope

Dara Runtime is a Windows-native engine that runs one multimodal generation at a time and aims to beat llama.cpp on each of four model tiers.

**In scope**

- One active session, one engine thread, one GPU.
- Text and image input on all four tiers, including the vision encoder and projector.
- Paged KV cache with TurboQuant 4-bit and 3-bit, plus FP8 and BF16 baselines.
- Chunked prefill, CUDA graphs, fused kernels, session KV retention.
- Speculative decoding using each model's built-in drafter or multi-token-prediction head.
- Single-request HTTP API with SSE streaming, cancellation and a 503 when busy.

**Out of scope for now:** continuous batching, concurrent requests, admission control, multi-tenant serving, audio and video input, an embeddings endpoint, Vulkan and multi-GPU.

**Done means:** each tier loads, produces correct text and image-grounded output, and beats llama.cpp on its win conditions (see Performance targets) on the reference GPU, with quality inside the gates in Testing.

## Model tiers and the multimodal requirement

All four tiers are natively multimodal, so every tier ships with its vision encoder and projector; three of the four share one Qwen hybrid architecture.

| Tier | Model | Size | Vision input | Context | Architecture |
| --- | --- | --- | --- | --- | --- |
| Nano 2B | [Gemma 4 E2B](https://arxiv.org/html/2607.02770) | 2.3B effective, about 5B total | 150M-parameter ViT, 70 to 1,120 tokens per image | 128K | 4:1 local to global attention, per-layer embeddings, KV sharing across 20 of 35 layers |
| Core 4B | Qwen3.5-4B | 4B language model plus a vision encoder | Native vision-language | 262,144 native | 32 layers: 8 repeats of 3 DeltaNet blocks then 1 gated-attention block |
| Pro 9B | Qwen3.5-9B | 9B | Early-fusion vision-language | 262,144 native | 32 layers: 8 repeats of 3 DeltaNet blocks then 1 gated-attention block |
| Ultra 27B | Qwen3.8-27B | 27B | Native vision-language (images and video) | 262,144 native | Built on the Qwen3.5 architecture: 64 layers, 16 repeats of 3 DeltaNet blocks then 1 gated-attention block |

**What changed from the earlier plan.** The Core tier is now Qwen3.5-4B instead of Gemma 4 E4B, and the 27B tier is Qwen3.8-27B. That makes one Qwen backend serve three tiers and one Gemma backend serve one, which changes the build order (see Roadmap).

**Verified from sources:** the Gemma 4 figures come from the [Gemma 4 technical report](https://arxiv.org/html/2607.02770). The Qwen3.8-27B layout comes from its [official model card](https://huggingface.co/Qwen/Qwen3.8-27B). The Qwen3.5-4B and 9B layouts come from their model cards, as shown in search excerpts of [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Confirm every layer count, head count and context length against each model's `config.json` in Phase 0.

**Deferred:** Gemma E2B also accepts audio, and the Qwen models accept video. Neither is in the first release; both reuse the same embedding-splice path later.

## Hardware and memory budgets

A 24 GB NVIDIA card is the smallest reference GPU that holds all four tiers fully GPU-resident at long context; a 16 GB card covers the first three. The reference GPU is still an open decision (see Open decisions).

| Tier | Q4 weights | System RAM | KV per token, FP16 | KV at 128K, FP16 | KV at 128K, ideal 4-bit payload | Recurrent state (est.) |
| --- | --- | --- | --- | --- | --- | --- |
| Nano 2B | 1.5 GB on GPU, plus about 1.4 GB of per-layer embeddings in host RAM | 8 GB | Small: the report lists +0.05 GB of int8 KV at 32K | About 0.2 GB (int8) | Not needed | None |
| Core 4B | 2.5 to 3 GB, plus the vision tower | 8 to 16 GB | 32 KiB | 4 GiB | 1 GiB | About 50 MB |
| Pro 9B | 5 to 6 GB, plus the vision tower | 16 GB | 32 KiB | 4 GiB | 1 GiB | About 50 MB |
| Ultra 27B | 15 to 17 GB, plus the vision tower | 32 GB | 64 KiB | 8 GiB | 2 GiB | About 150 MB |

Q4 weight and RAM figures are the ones you supplied. Everything else is arithmetic from the model cards: only the 16 (27B) or 8 (4B, 9B) full-attention layers keep a KV cache, each with 4 KV heads of dimension 256, so the 27B stores 16 × 2 × 4 × 256 = 32,768 values per token. The 4-bit column is the ideal packed payload, not the expected footprint: TurboQuant metadata, residuals and page alignment add overhead, so measure actual bytes per token for every implementation. At the full 262,144-token context the 27B needs 16 GiB of FP16 KV or about 4 GiB at 4-bit.

**Three corrections to carry forward**

- **Gemma E2B on disk is larger than 1.5 GB.** The per-layer embedding tables (about 2.7B parameters, from the report's embedder figures) stay in pinned host RAM and are gathered per token; only the effective weights need VRAM.
- **Qwen vision towers are extra.** A community 4-bit GGUF of Qwen3.5-4B that bundles its vision tower is [3.41 GiB](https://app.alphaneural.io/models/leok7v/Qwen3.5-4B), above your 2.5 to 3 GB. Budget the tower separately and measure it in Phase 0.
- **The recurrent-state figures are estimates.** They assume one 128 × 128 state per value head per DeltaNet layer, held in FP32.

**Ultra on a 24 GB card** is tight: about 17 GB of weights, about 2 GiB of 4-bit KV at 128K and the vision tower leave only a few GB for activations, the CUDA context and the Windows desktop. Plan on 128K, not 262K, as the supported ceiling for this tier until measured.

**Hardware profiles.** Keep three, because a small development GPU cannot validate the larger tiers' performance claims.

| Profile | Purpose |
| --- | --- |
| 4 GB | Kernel correctness, CPU fallback, memory pressure and small-model smoke tests |
| 16 GB | Nano, Core and Pro performance testing |
| 24 GB | Ultra residency, long-context and maximum-capacity testing |

**Context has five different limits, and reports must not mix them:** the model's native limit, the runtime-supported limit, what fits in VRAM, what fits with host offload, and what passes the quality gates. A model accepting 262,144 tokens doesn't mean Dara runs that on a given GPU.

## Architecture

One HTTP thread hands each request to one engine thread, which drives a single session through shared memory and compute layers.

&#91;embedded content: architecture · 5 layers, 1 session\]

The Session stays its own object, so concurrency can be added later without rewriting memory or kernels.

**Architecture manifests.** Phase 0 produces a machine-readable manifest for each exact checkpoint: layer types and order, head counts and dimensions, recurrent-state shapes, rotary and multimodal position settings, vision encoder and projector, image-token layout, and draft-head availability and tensor layout. Backends configure themselves from the manifest and reject an unsupported checkpoint instead of assuming another model in the family has the same dimensions.

## Memory engine

One paged block pool serves every tier; what differs by family is which layers allocate from it and what extra state a model carries.

| Family | Layers that keep KV | Other per-session state | Consequence |
| --- | --- | --- | --- |
| Qwen3.5 and Qwen3.8 | 8 full-attention layers (4B, 9B) or 16 (27B), each with 4 KV heads of dimension 256 | Fixed-size recurrent state for 24 or 48 Gated DeltaNet layers | KV grows with context; recurrent state does not |
| Gemma 4 E2B | Global layers keep full-length KV; local sliding-window layers keep a bounded window | Per-layer embedding rows gathered from host RAM | Local layers recycle blocks as the window slides; KV sharing across layers cuts storage further |

```latex
B_{kv} = L_{attn} \cdot 2 \cdot H_{kv} \cdot d_{head} \cdot b_{kv}
```

Here L\_attn is the number of KV-bearing layers, H\_kv the KV heads, d\_head the head dimension and b\_kv the bytes per stored value. Context admission is one check: the requested prompt plus max tokens must fit in the free pool.

**Paged KV.** Fixed-size token blocks (start at 32 tokens, tunable) addressed through a per-session page table that kernels take as input. Context can then grow, shrink and roll back without copying, and attention kernels are never rewritten for paging later.

**KV modes.** BF16 reference, FP8, TurboQuant 4-bit as the lead compressed candidate, TurboQuant 3-bit, and 2-bit behind an experimental flag. K and V bit-widths are set independently, and the newest tokens plus attention-sink tokens can stay at higher precision. An automatic policy falls back to FP8 or BF16 when the head dimension or kernel doesn't support the chosen mode, and logs the reason.

**TurboQuant as a fused kernel.** The attention kernel reads packed K and V and dequantizes in registers; expanding to FP16 first would give back the bandwidth saving. Prototype rotating the query versus rotating the cached keys, and measure the residual-correction cost at 3-bit and below. Compression only pays when KV reads are a large share of memory traffic, so it switches on past a configurable context length. FP8 stays the default until a fused TurboQuant kernel beats it on a real long-context workload. Your spec cites a 2026 vLLM evaluation where aggressive TurboQuant modes cut throughput relative to FP8; I haven't verified that result.

**First kernel experiment.** Compare four variants on a long-context decode workload: FP8 attention; TurboQuant 4-bit with decompression to registers; TurboQuant 4-bit fused; and TurboQuant 3-bit fused. Measure p50 and p99 decode latency and memory bandwidth. A smaller cache that makes attention slower is not an improvement.

**Recurrent-state pool.** Fixed-size slots per session, FP32 by default, with checkpoint and restore operations. Speculative decoding needs them to roll back rejected drafts, and session retention needs them to resume from the end of a previous turn.

**Session retention.** After each turn, keep the session's KV pages, state checkpoint and cached image embeddings. The next request reuses the longest common token prefix and prefills only the new tokens. The Qwen3.8 card says it keeps earlier thinking blocks in history by default and that this improves KV cache utilization, so Dara should render history the same way to keep prefixes stable. On Gemma, reuse is valid only for the part of a prefix still inside the retained local window.

**Session identity.** Requests may carry a `session_id`. Reuse requires all of: the same model and checkpoint version, the same tokenizer and chat-template version, an exact token-prefix match, the same rotary and position configuration, and identical image content and image-token sequence. Because matching is by exact tokens, a mismatch costs a re-prefill, never wrong output. Default to one retained session; keeping several dormant ones needs an eviction policy (least recently used, bounded by the session-state budget) but no concurrent inference. On cancel or disconnect, keep only the tokens already delivered to the client and trim the rest, so engine state never runs ahead of the client's history.

| Placement strategy | Priority |
| --- | --- |
| Gemma per-layer embeddings in pinned host RAM | Core |
| Vision tower resident on GPU; evict only on Ultra if VRAM requires | Core, per-tier setting |
| GPU and CPU layer offload | Later; needed to run Ultra on cards under 24 GB |
| KV pages in host RAM | Experimental; PCIe bandwidth is far below VRAM bandwidth |

**Four accounting categories.** Budget each separately, which keeps VRAM predictable on Windows where the desktop and driver also take GPU memory.

1. Model weights and static tensors.
2. Temporary compute workspace.
3. Session state: KV, recurrent state and cached embeddings.
4. Host-resident resources: per-layer embeddings and offloaded weights.

## Compute engine

At batch size 1, decode is limited by memory bandwidth, so Dara wins by moving fewer bytes and launching fewer kernels, not by raw FLOPs.

```latex
t_{decode} \le \frac{BW_{mem}}{B_{weights} + B_{kv,read} + B_{state}}
```

Illustration: Ultra reads about 16 GB of weights per token, so a card with about 1 TB/s of bandwidth caps decode near 60 tokens per second before KV and state reads. The reference GPU isn't chosen yet, so Phase 0 measures how close llama.cpp already gets to this ceiling for each tier.

**Decode path**

- Fused dequantize-and-multiply kernels for the quantized weight formats, with tensor-core GEMM for prefill.
- Fused RMSNorm, projections, rotary embedding and KV write, to cut launches per layer.
- CUDA graphs per context-length bucket and per draft length. The page table is read through device-side indirection, so a captured graph stays valid as context grows.
- GPU-side sampling, with the recommended thinking and non-thinking presets from the Qwen cards as defaults.

**Attention kernels.** Decode uses a split-KV kernel, because one sequence has little parallelism to spread over the GPU. Prefill uses a tiled causal kernel. Gemma local layers use a sliding-window variant. Each also needs a TurboQuant read path.

**DeltaNet kernels.** A chunked-parallel form for prefill and a recurrent update for decode, checked against each other and against a reference implementation. State stays in FP32 until parity tests show lower precision is safe. Before any optimization, a dedicated test processes one prompt as a single prefill and in chunks of 128, 256, 512 and 1,024 tokens, then compares final hidden states, recurrent states and logits.

**Chunked prefill.** The chunk size adapts to a per-launch time budget that stays well under the Windows GPU watchdog, whose default timeout is about two seconds. A long prompt therefore can't trigger a driver reset, and cancellation can be checked between chunks.

**Speculative decoding.** Both families ship a drafting mechanism, but they differ: Qwen's multi-token prediction lives inside the model, while Gemma's is a separate 76M-parameter drafter with its own weights and its own integration. The Gemma 4 report describes a 76M-parameter drafter for E2B that cross-attends to the main model's KV and supports any draft length. The Qwen cards state that multi-token prediction was trained for each model. Confirm in Phase 0 that the Qwen checkpoints include the draft weights. Verification runs one batched forward over the draft tokens, then commits the accepted ones and rolls back the rest: KV and recurrent state on Qwen, KV only on Gemma. Verifying several draft tokens in one forward pass happens inside one request, so it stays within the single-request scope.

**CPU backend.** A reference path for correctness tests and a fallback for layer offload, starting with AVX2.

## Vision pipeline

Each image is encoded once by its tier's vision tower, projected to the language model's width and spliced into the token stream, so all four tiers share one image path with per-family adapters.

1. **Decode and validate.** Use the Windows Imaging Component or a bundled decoder; apply EXIF orientation and enforce size limits before any GPU work.
2. **Resize and normalize per family.** Gemma 4 resizes while preserving aspect ratio to a patch grid with a token budget of 70, 140, 280, 560 or 1,120 (patch size 16, 3 × 3 pooling), per the technical report. Qwen uses dynamic resolution; read the processor configuration from each exact official checkpoint and reproduce its preprocessing instead of hardcoding values. A review of the Qwen3.8-27B configuration reports a 27-layer encoder with hidden size 1,152, patch size 16 and spatial merge size 2, which Phase 0 must confirm.
3. **Run the encoder on the GPU.** Gemma E2B's is a 150M-parameter ViT with non-causal attention and 2D rotary embeddings. Measure each Qwen tower in Phase 0; they are extra weights beyond the Q4 figures above.
4. **Project.** Gemma pools patch outputs into soft tokens in the language model's embedding space; Qwen merges patches. Load the projector from native weights and also from the separate projector GGUF that llama.cpp uses, so benchmarks run on identical weights.
5. **Expand placeholders and assign positions.** The chat template's image marker becomes N image slots once step 2 fixes N. Qwen uses interleaved multimodal rotary positions (the 27B card lists sections 11, 11 and 10), so position IDs for image tokens are not simply sequential.
6. **Splice into prefill.** Replace the placeholder embeddings with the projected image embeddings. Treat an image's tokens as one span when chunking unless Phase 0 shows the language model attends to them causally; confirm this for each family.
7. **Cache.** Keep encoder outputs keyed by image hash with the retained session, so follow-up turns skip steps 1 to 4.

**Token budget is the quality and latency knob.** Default to 280 tokens per image on Gemma, the lower of the two settings the report evaluates, and let the request raise it to 1,120. Define the equivalent Qwen setting once its processor limits are read.

**Golden tensors.** Compare five intermediate outputs against the reference pipeline, not only final answers: the preprocessed image tensor, the vision encoder output, the projector output, the multimodal position IDs, and the final embedding sequence entering the language model. A mismatch then points at resizing, the encoder, the projector, position handling or the language model. Gemma also needs its per-layer embeddings checked, including image positions where an ordinary token ID may not exist.

**Residency.** Keep the encoder on the GPU for Nano, Core and Pro. On Ultra with a 24 GB card, offer a mode that runs the encoder, then evicts it to pinned host RAM, trading image latency for KV headroom.

**Deferred:** video frames and Gemma audio reuse steps 5 and 6 and add their own encoders later.

## Single-request API

One generation runs at a time, and any other request is rejected immediately with 503 before it costs any CPU.

**Request lifecycle**

1. Validate the basic request structure and size caps.
2. Atomically take the busy flag. If it is already taken, return `503 runtime_busy` with a `Retry-After` header.
3. Decode images, tokenize, and render the chat template, including each model's thinking controls.
4. Check the fit: prompt tokens plus `max_tokens` against the context the current KV mode and free VRAM allow. On failure, release the flag and return `400 context_too_long` with the limit.
5. Run chunked prefill, then decode, streaming deltas over SSE.
6. On completion, cancel or client disconnect, release the flag and keep the session for retention.

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI-compatible chat with text and `image_url` content parts, streaming or not |
| `POST /v1/completions` | Raw text completion, text only |
| `GET /v1/models` | Models on disk and the loaded one |
| `POST /v1/models/load`, `POST /v1/models/unload` | One model resident at a time; switching tiers unloads the old one |
| `GET /v1/status` | Busy flag, active request ID, KV mode, memory use |
| `GET /v1/metrics` | Prometheus-format runtime metrics |
| `DELETE /v1/requests/{id}` | Cancel the active request |

Removed from scope: embeddings, application-level batch submission and WebSocket.

**Behaviors**

- Cancellation and client disconnect set one atomic flag, checked between decode steps and between prefill chunks.
- The token channel between the engine and the HTTP thread is bounded; a client too slow to read is cancelled rather than allowed to stall the engine.
- Separate timeouts cover prefill and total generation.
- Every request has an ID; clients may supply their own so retries are recognizable.
- The final chunk carries usage: prompt tokens, image tokens, completion tokens, time to first token and decode tokens per second.
- The server binds to 127.0.0.1 by default and rejects `file://` image URLs unless enabled.
- Errors are JSON with stable codes, and a failed request never leaves blocks allocated.

**Session identity.** Requests may carry a `session_id`; the matching and cancellation rules are in the Memory engine section.

## Performance targets and benchmark method

Dara wins a tier when it beats llama.cpp on that tier's win conditions with quality inside the gates; plain short-context decode can't be the only test, because both engines sit near the same bandwidth ceiling.

| Tier | Win conditions against llama.cpp |
| --- | --- |
| Nano 2B | Plain decode (small models are launch-bound, so graphs and fusion should win), decode with the MTP drafter, time to first token with one image |
| Core 4B | Plain decode, prefill, long-context decode with TurboQuant, time to first token with one image |
| Pro 9B | Prefill, long-context decode, speculative decode, image time to first token; plain decode at parity or better |
| Ultra 27B | Maximum context on a 24 GB card, long-context decode, prefill, speculative decode; plain decode at parity or better |

Phase 0 decides how realistic the plain-decode goals are. If llama.cpp already reaches within a few percent of the bandwidth ceiling on a tier, that tier's win has to come from the other conditions.

| Metric | Measured at | Note |
| --- | --- | --- |
| Decode tokens/s | 1K, 8K, 32K and 128K context | Report the curve, not one number |
| Prefill tokens/s | 512 to 128K prompt tokens | Includes chunking overhead |
| Time to first token | Cold, warm (retained session), and with one image at 280 and 1,120 tokens | Separate numbers |
| Inter-token latency | p50 and p99 | Spikes reveal Windows desktop contention |
| Maximum context | Per KV mode, within the VRAM budget | Per tier and reference GPU |
| KV bytes per token | Each KV mode | Include TurboQuant metadata |
| Speculative decode | Tokens/s and draft acceptance rate | Measured apart from plain decode |
| Quality | Perplexity delta and long-context retrieval against BF16 KV; image question answering | Gates in the Testing section |

**Protocol**

- Same GGUF quantization, GPU, driver, context length and images on both engines; pin the llama.cpp commit and build flags.
- Run llama.cpp with flash attention and with its own quantized KV types (q8\_0 and q4\_0), so TurboQuant is compared with the strongest built-in alternative.
- Greedy sampling for speed runs, warm-up runs discarded, enough repeats to report median and p95.
- Record GPU clocks, power limit and whether the display runs on the same GPU.
- Check results into the repository, with a nightly run compared against them.

**Two correctness references.** Reference A is the original implementation (Transformers with the original checkpoint and processor) and establishes model correctness. Reference B is the same quantized weights, tokenizer and multimodal assets in llama.cpp, where supported, which isolates differences introduced by Dara's quantized kernels. Never expect Q4 logits to equal an unquantized reference. Pin the llama.cpp commit, disable its continuous batching for the baseline, and check Gemma per-layer-embedding support at that commit. Where the pinned build lacks a feature, record the limitation and use another validated reference rather than an unequal comparison.

## Roadmap

Nine phases take Dara from a measured baseline to a hardened single-request runtime, and each closes on a measurable gate.

&#91;embedded content: roadmap · 9 phases, 9 gates\]

Qwen comes first because one backend then covers three tiers, and speculation comes before the larger models because the 4B proves it cheaply. The checkpoint and restore interfaces are built in Phase 1, before kernel optimization starts, even though speculation arrives in Phase 5. Dates are not set: they need the reference GPU and team size.

## Technology stack

Dara is Windows-native C++/CUDA serving a single request at a time. Check each library's license before adopting it, and pin every version.

| Layer | Choice | Notes |
| --- | --- | --- |
| Core language | C++20 | Python stays out of the hot path |
| GPU kernels | CUDA C++ with a pinned CUDA Toolkit | Custom kernels for decode matmul, attention, DeltaNet and TurboQuant |
| GPU libraries | cuBLAS for prefill GEMM, CUTLASS for tensor-core building blocks, NVML for real free-VRAM queries | Use cuBLAS as a speed check too |
| Compiler and IDE | MSVC (Visual Studio 2022 Build Tools), NVCC | Pin both together; CUDA supports only certain MSVC versions |
| Build and deps | CMake + Ninja, vcpkg manifest mode | One command builds everything from a clean machine |
| Model loading | A small GGUF reader (or ggml's) and a safetensors reader | Safetensors for reference weights, GGUF for the quantized runtime |
| Tokenizers | Byte-level BPE for Qwen, SentencePiece for Gemma | Validate against the reference tokenizers token-for-token |
| Chat templates | A C++ Jinja engine such as minja | Needed for thinking flags and image markers |
| Image handling | Windows Imaging Component or stb/libjpeg-turbo, plus a bicubic resize that matches the reference | Gemma's reference uses bicubic resizing |
| HTTP and SSE | cpp-httplib to start; Boost.Beast if async is needed later | A single request doesn't need a heavy server |
| JSON | nlohmann/json, or yyjson if profiling shows a need | JSON is off the hot path |
| CPU kernels | AVX2 intrinsics, own thread pool | Reference path and offload fallback |
| Logging and metrics | spdlog, hand-written Prometheus text output | |
| Unit tests | GoogleTest + CTest | CPU references for every kernel |
| Parity and tooling | Python: PyTorch, Transformers, safetensors, NumPy, Pillow, pytest | Generates manifests, golden tensors and Reference A outputs |
| Baseline | llama.cpp built from one pinned commit (`llama-bench`, `llama-server`) | Reference B; disable its continuous batching |
| Profiling | Nsight Systems, Nsight Compute, compute-sanitizer, NVML, Windows Performance Analyzer | Run benchmarks natively on Windows, not under WSL |
| CI | Self-hosted Windows runner with an NVIDIA GPU | Nightly benchmark against stored baselines |
| Code quality | clang-format, clang-tidy, pre-commit hooks | |
| Packaging | CPack zip first, installer later | Ship CUDA runtime DLLs with it |

**The stack is not the advantage.** llama.cpp is also C++ and CUDA, so the language, compiler and libraries give no edge. Any gain has to come from a specific technique, listed here in order of how likely each is to win:

| Technique | Where it should win | Confidence |
| --- | --- | --- |
| Speculative decoding with the models' built-in drafters (Qwen MTP, Gemma's 76M drafter) | Single-stream decode on every tier; the only technique that gets past the memory-bandwidth ceiling | Highest. Check whether the pinned llama.cpp already supports these drafters, since that narrows the gap |
| Kernels for the Qwen hybrid architecture (chunked-parallel DeltaNet for prefill, recurrent update for decode) | Prefill on Core, Pro and Ultra | Medium. llama.cpp's generic path may be less tuned for this layout; measure it |
| Fused TurboQuant attention | Long-context decode and maximum context on Ultra at 24 GB | Unproven. Wins only if the fused kernel beats FP8 and llama.cpp's q8_0 and q4_0 KV types |
| CUDA graphs and fused decode kernels | Plain decode on Nano and Core, where small models lose a large share of time to kernel launches | Medium. Gains are modest, and llama.cpp already uses graphs in some cases |
| Split-KV decode attention | Decode at 32K and 128K context | Medium |
| Session KV retention plus cached image embeddings | Time to first token on follow-up turns and repeated images | Medium. llama.cpp's server also caches prompts, so compare against that setting |

**GPU architecture matrix.** Build for each GPU generation you test on. A 4 GB GTX 1650-class laptop is Turing, which has no BF16 tensor cores, and FP8 needs Ada or newer. That machine can validate correctness and the CPU fallback, but not the FP8 or BF16 paths. Confirm the compute capability of the 16 GB and 24 GB cards before fixing the kernel targets.

**Install first, for Phase 0:** Visual Studio Build Tools, CUDA Toolkit, CMake, Git, a Python environment with the reference packages, and a pinned llama.cpp build.

## Testing and quality gates

Correctness comes first at every phase: a speed result counts only after the layer below it passes.

| Layer | What it checks | Reference |
| --- | --- | --- |
| Kernels | Matmul formats, attention (causal, sliding window, split-KV), DeltaNet chunked against recurrent, TurboQuant round-trip error | CPU reference in GoogleTest |
| Model logits | Maximum logit difference and top-1 agreement on fixed prompts, per weight format | Reference A: Hugging Face Transformers; quantized weights also against Reference B, llama.cpp |
| Vision | Encoder and projector output similarity on fixed images, plus an image question-answering smoke set | The model's reference processor |
| KV quality gates | Perplexity delta and needle-style retrieval at 32K and 128K against BF16 KV, with image question answering unchanged | Thresholds fixed before Phase 4 starts |
| Speculative decoding | Greedy output identical with and without drafting; sampling distribution preserved | Plain decode |
| Soak | Repeated load and unload, cancel, disconnect and busy rejection; block count returns to baseline; no GPU resets | Metrics endpoint |

CI runs on a Windows machine with an NVIDIA GPU, with the MSVC and CUDA toolkit versions pinned. A nightly benchmark compares against the stored baselines and flags regressions.

## Risks and mitigations

The largest risk is promising a plain-decode win that memory bandwidth doesn't allow, so the plan measures the gap to the ceiling before committing to it.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| llama.cpp is already near the bandwidth ceiling on a tier | Plain decode can't be the win | Measure the gap in Phase 0; win on long context, prefill, speculation and capacity |
| Fused TurboQuant attention is slower than FP8 | The headline feature underdelivers | Prototype the kernel early at head dimension 256; keep FP8 as the fallback |
| DeltaNet kernels are numerically sensitive | Wrong output on three of four tiers | FP32 state, chunked-against-recurrent tests, a reference implementation |
| Vision preprocessing mismatches the reference | Quality drops with no crash | Golden embeddings from the reference processor for every tier |
| Ultra is too tight on 24 GB | Short context or no image support | 128K ceiling, optional encoder eviction, layer offload |
| Windows GPU watchdog or silent VRAM paging | Hangs or sudden slowdowns | Bounded launch times, own VRAM budget, query real free memory |
| New model releases change layouts | Rework | Select backends by architecture family from `config.json`; Qwen3.8 already reuses the Qwen3.5 type |
| Scope: four tiers, vision, TurboQuant, speculation | Schedule slips | Every phase ends in a measured gate, and the next one starts only after it passes |

**Added after review**

- Gemma per-layer embeddings: an incomplete implementation can produce plausible text with degraded quality, so they get their own correctness and benchmark gate in Phase 7, and host-resident placement is benchmarked against alternatives.
- TurboQuant can shrink KV without speeding up attention; FP8 stays the default until a fused kernel wins.
- Unsupported checkpoints: the runtime rejects any configuration that doesn't match a validated manifest.

## Open decisions and next actions

Six decisions block the start of Phase 0; the defaults below are my recommendations.

| Decision | Why it matters | Recommended default |
| --- | --- | --- |
| Reference GPUs | Sets every VRAM budget and the Ultra context ceiling | One 16 GB and one 24 GB NVIDIA card |
| First model | Decides which backend and kernels come first | Qwen3.5-4B: one backend then serves three tiers. The alternative, Gemma E2B, gives a faster first end-to-end slice with ordinary attention but leaves DeltaNet risk for later. |
| Speculation source | Avoids shipping a separate draft model | Use the models' own MTP heads, once Phase 0 confirms the weights are in the checkpoints |
| Ultra vision residency | Trades image latency against KV headroom | GPU-resident, with an optional eviction mode |
| Default image token budget | Quality against time to first token | 280 on Gemma, raisable to 1,120; set the Qwen equivalent after Phase 0 |
| llama.cpp baseline | Fairness of every comparison | Pin one commit, one quantization per tier, with and without flash attention and quantized KV |

**Next actions**

1. Choose the reference GPUs and pin the llama.cpp commit and build flags.
2. Build the Phase 0 harness: baseline runs for all four tiers including image prompts, plus reference logits and golden embeddings.
3. Read each model's `config.json` and write its state specification (KV layers, window, recurrent state size).
4. Spike the fused TurboQuant attention kernel at head dimension 256 against FP8 before committing to the Phase 4 design.
5. Define the `Session` and backend interfaces, including checkpoint and restore, before writing kernels.

**Added after review:** also decide how many dormant sessions to retain and the eviction policy. Phase 0 additionally produces an architecture manifest per exact checkpoint and records both correctness references.

## Sources

Checked on 2026-11-08. Pages opened in full:

- [Gemma 4 Technical Report](https://arxiv.org/html/2607.02770)
- [Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B)

Read only as search excerpts, so confirm against the full pages: the [Qwen3.5-4B card](https://huggingface.co/Qwen/Qwen3.5-4B), the [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B), a [community Qwen3.5-4B GGUF card](https://app.alphaneural.io/models/leok7v/Qwen3.5-4B), an [Artificial Analysis article](https://artificialanalysis.ai/articles/qwen3-5-small-models) and a [Qwen3.5 overview](https://besthub.dev/articles/qwen3-5-small-models-unveiled-from-0-8b-to-9b-with-full-capabilities-e409056d2b3c).

Not verified here: the vLLM TurboQuant result cited in your spec, and llama.cpp's support for the quantized KV types named in the benchmark protocol.
