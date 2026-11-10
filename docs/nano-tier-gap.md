# Nano tier (Gemma 4 E2B)

Status: **implemented and validated** against an independent reference (see "Validation"). Files: `src/model/gemma_model.{h,cu}`,
`src/model/gemma_ops_cuda.{h,cu}`, `src/tokenizer/gemma_tokenizer.{h,cpp}`, `tools/nano_generate.cu`, `tools/ref_forward_gemma.py`,
`tools/make_gemma_tokenizer_golden.py`, `tests/gemma_tokenizer_test.cpp`. Build: `tools/build_nano.bat` (also run by `tools/build_cuda.bat`).

Source: `google/gemma-4-E2B-it` (revision `4adf15748de9e1a0097b6274cee9c15836c41c37`), GGUF `unsloth/gemma-4-E2B-it-GGUF`
(revision `0314792d7f1f7e229411f620751375812bb9faf2`, `gemma-4-E2B-it-Q4_K_M.gguf`, 3,106,738,272 bytes). Reference: the installed
`transformers` 5.5.4 Gemma 4 modules. Dara reads weights, dimensions, layer layout and tokenizer from the GGUF only.

## Architecture as implemented
35 layers, hidden 1536, 8 query heads, 1 KV head, vocab 262144, tied output head, final logit soft-cap 30.
* **Layer kinds** from `gemma4.attention.sliding_window_pattern`: sliding-window layers (window 512, head_dim 256, rope theta 1e4,
  default NeoX rope) and global layers (head_dim 512, "proportional" rope theta 1e6: only the first 64 frequency pairs rotate, which the
  GGUF states through `rope_freqs.weight`).
* **Attention** scaling is 1.0; q/k carry per-head RMSNorm with a learned weight (no +1 folding), V is RMS-normalised without a weight.
* **KV sharing**: layers 15-34 own no K/V and read layer 13's (sliding) or layer 14's (global). The GGUF still stores unused k/v
  tensors for them; they are not loaded.
* **Four norms per layer**, GELU(tanh)-gated MLP, width 6144 (12288 on the KV-sharing layers).
* **Per-layer input embeddings**: a 262144 x (35*256) Q5_K table (1.6 GB) kept in host RAM when it fits next to the RAM safety
  margin, otherwise read row by row from the file, plus a BF16 projection of the token embedding; each layer ends with a gate /
  projection / norm block, then a learned per-layer scalar.
* **Tokenizer**: SentencePiece-style BPE (`tokenizer.ggml.model = gemma4`), no regex pre-tokenizer: spaces become U+2581, BPE runs over
  the whole segment, unknown characters fall back to `<0xHH>` bytes, `<bos>` is prepended.

## Residency
Same machinery as the Qwen tiers: VRAM-resident layers in permanent slabs, remaining layers in pinned host RAM streamed through a
double-buffered staging arena, planner fed with real free VRAM, clean failure when it cannot fit. On the 4 GB development GPU the
whole model is resident: 1048 MiB of layers + 216 MiB head + K/V + workspace (counted device peak about 1.4-1.5 GiB).

## Validation (Reference B = the same GGUF, dequantized, run through the real HF Gemma 4 modules layer by layer in fp32)
* 8 positions: worst layer error 6.9e-7 relative, logits error 1.8e-6, top-1 8/8, greedy continuation identical.
* 700 positions (sliding window active past position 512): worst layer error 5.2e-6 over all 35 layers x 700 positions; logits at the
  last four positions agree on top-1 with relative errors <= 3.0e-6.
* Hosted-layer streaming (10 and 0 layers resident) is bitwise identical to the fully resident run.
* Tokenizer vs the Hugging Face tokenizer: 2332 text cases and 500 random id sequences identical (encode, decode, streaming decode).
  One documented difference: the GGUF types `<eos>` (id 1) NORMAL, so its text is ordinary text here (HF treats it as special); the
  runtime's end token is `<turn|>` (106), which matches.

## Speed (GTX 1650, resident, greedy decode)
Profiling (`nano_generate --profile`, GPU-timestamp events between operations) showed decode was launch-bound: host enqueue time (24.5 ms)
about equal to the GPU timeline (25.3 ms) for ~1000 small kernels per token. Single-token decode is now captured as a CUDA graph
(`src/model/decode_graph.h`, shared with the Qwen backend): 41 -> 49-54 tok/s with bitwise-identical logits. The position lives in device
memory so one graph serves every position; the graph is re-captured when its buffers move. `tests/nano_graph_test.cu` checks graphs-on vs
graphs-off bitwise over a re-capture, a KV rewind and 580 replays crossing the sliding window. `DARA_NO_GRAPH=1` disables graphs.
The MLP matmuls run at about 80% of memory bandwidth; the rest is many small kernels (per-layer-input block, norms, KV bookkeeping),
which fusion could still shrink.

## Not done yet
* Further speed work (attention is the simple correctness kernel, capped at 12,288 positions; small kernels are not fused);
  no speculative decoding for Nano.
* Chat template: done for text conversations (`src/tokenizer/gemma_chat.*`, `nano_generate --chat/--system/--think`), identical to the
  Hugging Face renderer on 600 random conversations (rendered text and token ids). Tools, tool calls, images, audio and reasoning
  fields are rejected, not approximated. Control tokens typed into message content stay text (template tokens are emitted by id).
* The 150M vision tower, the 76M drafter and the audio path are out of scope.
* Untied output heads are rejected (E2B ties them).
