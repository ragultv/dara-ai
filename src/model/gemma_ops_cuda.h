#pragma once
// CUDA operators specific to the Gemma 4 ("Nano") architecture. Correctness first: simple, deterministic kernels validated
// against the Hugging Face reference; speed work comes after.
#include <cstddef>
#include <cstdint>

namespace dara::model {

// p[i] = first + i for i < n (device int64 positions for RoPE).
int fill_positions_cuda(int64_t* d_p, int64_t first, int n, void* stream = nullptr);

// out[i] = gelu_tanh(gate[i]) * up[i]  (Gemma's gated MLP activation, hidden_activation = gelu_pytorch_tanh). out may alias gate.
int geglu_cuda(const float* d_gate, const float* d_up, float* d_out, size_t n, void* stream = nullptr);

// x[i] *= s (in place).
int scale_cuda(float* d_x, float s, size_t n, void* stream = nullptr);

// x[i] = cap * tanh(x[i] / cap) (in place): the final logit soft-cap.
int softcap_cuda(float* d_x, float cap, size_t n, void* stream = nullptr);

// out[l][t][j] = (proj[t][l][j] + tok[t][l][j]) * scale   for proj/tok laid out [T][layers][width]; out is laid out [layers][T][width] so
// each layer's per-layer input is one contiguous [T][width] block. tok may be null (then only proj * scale).
int combine_per_layer_inputs_cuda(const float* d_proj, const float* d_tok, float* d_out, int T, int layers, int width, float scale, void* stream = nullptr);

// Gemma attention for T new tokens against a K/V cache ([position][kv_heads][head_dim], the cache's own layout). Token t sits at the
// absolute position d_pos[t] (device array), so a captured CUDA graph can be replayed at a different position without re-capturing.
// smem_positions: how many score slots to reserve per block; must be >= the largest number of positions any token attends to
// (window-bounded for sliding layers). q: [T][heads][head_dim], already normed and rotated.
//   * scaling is 1.0 (no 1/sqrt(d)); q_norm / k_norm carry the scale
//   * window > 0: causal sliding window, position p attends to [p - window + 1, p]; window == 0: full causal
//   * GQA: heads / kv_heads query heads share each KV head
// out: [T][heads * head_dim]. head_dim must be a multiple of 32 and <= 512.
int gemma_attention_cuda(const float* d_q, const float* d_K, const float* d_V, const int64_t* d_pos, int T, int heads, int kv_heads, int head_dim, int window,
                         int smem_positions, float* d_out, void* stream = nullptr);

// Writes k/v ([T][kv_width]) into the caches at positions d_pos[t]: Kc[d_pos[t]*kv_width + i] = k[t*kv_width + i] (same for V).
int kv_append_cuda(float* d_Kc, float* d_Vc, const float* d_k, const float* d_v, const int64_t* d_pos, int T, int kv_width, void* stream = nullptr);

// Largest number of attended positions the attention kernel supports (it keeps one score per position in shared memory).
constexpr int kGemmaMaxContext = 12288;

}  // namespace dara::model
