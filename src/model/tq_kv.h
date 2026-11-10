#pragma once
// TurboQuant-style 4-bit KV cache (data-oblivious vector quantization), shared by every tier.
//
// Per head vector x (head_dim d, a power of two): store its norm n = |x| and a 4-bit code per coordinate of z = H D (x / n), where D is a
// fixed pseudo-random +-1 diagonal and H the (unnormalised) Walsh-Hadamard matrix. For a unit vector z_i is ~N(0,1), so each coordinate is
// quantised with the 16-level Lloyd-Max codebook of the standard normal (computed at start-up; MSE ~0.0095 of the unit variance).
// Rotation is orthogonal (R = H D / sqrt(d)), so attention runs in the ROTATED domain and the cache is never expanded:
//     q . k  =  (n_k / d) * sum_i (H D q)_i * c[code_k,i]        (rotate the query once per head)
//     sum_j p_j v_j  =  (1/d) D H ( sum_j p_j n_j c[code_j,.] )   (inverse-rotate the output once)
// Cache layout per layer: codes [position][kv_head][d/2] bytes (low nibble = even coordinate), norms [position][kv_head] floats.
// Storage: d/2 + 4 bytes per head vector = 4.125 bits/value at d=256 (FP32 KV is 32), FP16 KV is 16.
// This is the MSE-optimal variant; the 1-bit QJL residual correction of the paper's inner-product variant is NOT included.
#include <cstddef>
#include <cstdint>

namespace dara::model {

// head_dim values supported by the kernels (power of two, 64..512).
bool tq_supported_head_dim(int hd);

// Bytes of cache per position per kv head for head_dim d (codes + norm).
inline size_t tq_bytes_per_head_vector(int d) { return static_cast<size_t>(d) / 2 + sizeof(float); }

// Uploads the codebook to the device. Idempotent; must be called OUTSIDE CUDA-graph capture before the first kernel use.
int tq_init();

// The 16 centroids (ascending) of the standard-normal Lloyd-Max quantiser, and its mean squared error (for tests / reporting).
const float* tq_centroids();
double tq_codebook_mse();
const float* tq_boundaries();  // the 15 decision boundaries the device uses (for CPU reference implementations)

// Quantises and stores n tokens: x is [n][kv_heads][d] (already RoPE'd for keys); token t goes to position d_pos[t] (device array).
int tq_append_cuda(const float* d_x, const int64_t* d_pos, int n, int kv_heads, int d, uint8_t* d_codes, float* d_norms, void* stream = nullptr);

// Causal attention over the quantised cache for n query tokens (token t at position d_pos[t], attends to positions
// [window > 0 ? p-window+1 : 0, p]). q: [n][heads][d] (normed and rotated by RoPE already). scale multiplies the scores
// (Qwen 1/sqrt(d), Gemma 1.0). gate (nullable): [n][heads*d]; the output is multiplied by sigmoid(gate) (Qwen's output gate).
// smem_positions: score slots reserved per block (>= the longest context any token sees, <= kTqMaxPositions). out: [n][heads*d].
constexpr int kTqMaxPositions = 11000;
int tq_attention_cuda(const float* d_q, const float* d_gate, const uint8_t* d_Kc, const float* d_Kn, const uint8_t* d_Vc, const float* d_Vn,
                      const int64_t* d_pos, int n, int heads, int kv_heads, int d, int window, float scale, int smem_positions, float* d_out,
                      void* stream = nullptr);

}  // namespace dara::model
