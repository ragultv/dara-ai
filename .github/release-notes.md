## Faster long conversations, a prompt cache, and a simpler first run

Dara is a Windows app with its own CUDA engine that runs Gemma 4 and Qwen3.5 models on your own NVIDIA laptop GPU. Your chats stay on your computer. This release is mostly about speed in long conversations, and it removes the first-run welcome screen.

### Highlights

- **Long conversations no longer slow down as much.** The attention step now reads the conversation history once per group of heads instead of once per head. Dara Nano on a 4 GB laptop GPU decodes at **59 tokens/s at 16K tokens of context (was 42)** and **39 tokens/s at 65K (was 19)**.
- **Long prompts are read faster.** Reading a 16K-token prompt is 1.2x faster on Nano and 1.65x faster on Core 4B, with no loss in our quality check (the parts that round numbers differently were tested on a quality set before being turned on).
- **Coming back to a conversation is quick.** Dara keeps recent conversations' state in RAM. Returning to a chat takes about **0.15 s to the first word instead of 1.7 to 2.1 s** on Nano, and 0.4 s instead of 3.3 to 3.8 s on Core 4B (conversations of about 1,300 tokens). A conversation you return to after restarting the app is read again.
- **Dara Nano uses the new Flash drafter v3.** It predicts 69% of the next words correctly in held-out tests (v2: about 55%). With Flash on, Dara Nano averages **1.61x llama.cpp** over our 16 everyday tasks (v2: 1.50x), and up to 2.5x on edits.
- **No welcome screen.** The app opens straight on Chat. If no model is downloaded yet, Chat points you to **Models**, where you download one. The models folder is a setting in Settings > Engine.
- **No extra terminal window.** The engine now starts hidden in the background; the app is the only window you see.
- **Open source.** Dara is released under the MIT License.

### Speed against llama.cpp

Same model file, same prompt, 16 everyday tasks, median of three runs, 4 GB laptop GPU, llama.cpp build b12189:

| Model | Dara with Flash | Dara plain (Flash off) |
| --- | --- | --- |
| Dara Nano | 1.61x | 1.14x |
| Dara Core 4B | 1.65x | 1.25x |

Edits and conversions gain the most because most of the answer is already in your prompt. Open-ended writing gains around 1.3 to 1.6x, not several times. How every number is produced: `docs/BENCHMARKING.md`.

### Install

1. Download **Dara_0.1.0_x64-setup.exe** below. It installs for your user account only, with no administrator prompt.
2. You need an NVIDIA GPU from the **GTX 16 or RTX 20 series on** and a driver that supports **CUDA 13 (R580 or newer)**.
3. Open **Models**, pick one that fits your GPU and download it.

**Updating:** install over your existing copy. Your chats, settings and downloaded models are kept. Flash drafters are downloaded with each model.

To verify a download, compare `Get-FileHash <file>` with the `.sha256` file next to it.

### Known issues

- **The installer is not code-signed yet**, so Windows SmartScreen may show "Windows protected your PC". Choose **More info**, then **Run anyway**. We are working on signing.
- Reading a very long prompt for the first time still takes a while on a laptop GPU (about 109 s for 33K tokens and 6 minutes for 65K on Nano). Returning to a conversation you already had is fast; a brand-new huge prompt is not.
- Flash switches itself off for a conversation when its guesses stop paying, which in open-ended writing happens from about 4,000 tokens of context.
- With Flash on, a few words can differ from plain decoding where two words are nearly tied (the model still writes its own answer). In our 16-task run, 11 of 16 Nano answers were word-for-word identical.
- Speeds are measured on a 4 GB laptop GPU only. Other GPUs have not been measured; the check in `docs/RTX_LAPTOP_CHECK.md` takes about 15 minutes.
- There is no automatic update yet.

### Under the hood

- Grouped-query decode attention kernel (reads each K/V row once per group), bitwise identical to the old path.
- Grouped prefill attention, plus int8 attention scores on GPUs without tensor cores. Each new path has an off switch: `DARA_GQA_ATTN=0`, `DARA_ATTN_GROUP_PREFILL=0`, `DARA_ATTN_QK_INT8=0`, `DARA_PROMPT_CACHE_MB=0`.
- Prompt cache that parks other conversations' K/V (and, for Core 4B, its recurrent state) in host RAM; budget set with `DARA_PROMPT_CACHE_MB`.
- New benchmarks in `bench/` (Husky-style matrix, tool calls, first-word latency, context sweep, before/after check) and kernel tests in `tests/`.
