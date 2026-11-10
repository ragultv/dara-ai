#pragma once
// Split-KV ("flash-decoding") attention for decode and short verification steps, for both KV formats and both model families.
//
// The one-block-per-head kernels leave most of the GPU idle when only a few tokens are in flight: with 16 heads on a 14-SM GPU a 1500-token
// context runs 16 blocks, each looping serially over every position. Here each (head, token) is split across blocks of kSplitChunk
// positions; each block computes a local softmax maximum, local normaliser and an unnormalised local output, and a small combine kernel merges
// the partials (the standard online-softmax recurrence). Results differ from the single-block kernels only by floating-point summation order.
//
// Formats: F32 (K/V [position][kv_heads][head_dim] floats) and TQ4 (tq_kv.h: packed 4-bit codes + norms; attention in the rotated domain).
// The grid is sized from `max_positions`, never from the current position, so a captured CUDA graph stays valid at every position: blocks past
// the current context exit immediately. There is no cap on the context length (scores only ever cover one chunk).
#include <cstddef>
#include <cstdint>

namespace dara::model {

constexpr int kSplitChunk = 128;  // positions per block

enum class AttnFmt { F32, TQ4, Q8, K16V8 };

// INT8 K/V cache: codes [position][kv_heads][hd] int8 and scales [position][kv_heads][hd/32] float (scale = max|x| / 127 per 32 values, ~9 bits per
// value). Appends T vectors x [T][kv_heads][hd] at positions d_pos[t]. hd % 32 == 0.
// FP16 K append (K16V8): x [T][kv_heads][hd] -> half cache [position][kv_heads][hd].
int f16kv_append_cuda(const float* d_x, const int64_t* d_pos, int T, int kv_heads, int hd, uint8_t* d_half, void* stream = nullptr);
int q8kv_append_cuda(const float* d_x, const int64_t* d_pos, int T, int kv_heads, int hd, uint8_t* d_codes, float* d_scales, void* stream = nullptr);

// Uploads the TurboQuant codebook this file's kernels read. Idempotent; call outside CUDA-graph capture (the model constructors do).
int attn_splitkv_init();

// Scratch floats needed per token in flight (partials, plus the rotated query for TQ4) for a context of up to max_positions.
size_t attn_splitkv_scratch_floats_per_token(AttnFmt fmt, int heads, int hd, int max_positions);

// out[t][h][:] = softmax(scale * q.K over the causal window) V  (* sigmoid(gate) when gate != nullptr), token t at position d_pos[t].
// K/Kn/V/Vn: F32 -> K, V are float* caches and Kn/Vn are ignored; TQ4 -> K, V are uint8_t* codes and Kn, Vn float* norms;
// Q8 -> K, V are int8 codes and Kn, Vn the per-32 float scales (hd % 128 == 0).
// window > 0: sliding window; q: [T][heads][hd]; gate: [T][heads*hd] or null; scratch must hold >= one token's worth (more tokens are batched
// per launch group). hd: 128, 256 or 512 for F32; 64..512 power of two for TQ4 (the kernels assume hd % 128 == 0 for F32).
int attn_splitkv_cuda(AttnFmt fmt, const float* d_q, const float* d_gate, const void* d_K, const float* d_Kn, const void* d_V, const float* d_Vn,
                      const int64_t* d_pos, int T, int heads, int kv_heads, int hd, int window, float scale, int max_positions, float* d_scratch,
                      size_t scratch_floats, float* d_out, void* stream = nullptr);

}  // namespace dara::model
