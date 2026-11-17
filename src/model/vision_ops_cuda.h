// CUDA operators of the image encoders (Gemma 4 vision tower). Simple, deterministic kernels; the matrix products reuse the text path's
// BF16 GEMM (matvec_multi_cuda).
#pragma once

#include <cstddef>

namespace dara::model {

// x[i] = min(max(x[i], lo), hi)   (Gemma 4 "clippable" linears clamp their input and output)
int clamp_cuda(float* d_x, size_t n, float lo, float hi, void* stream = nullptr);
// out[i] = min(max(x[i], lo), hi)
int clamp_copy_cuda(const float* d_x, float* d_out, size_t n, float lo, float hi, void* stream = nullptr);

// 2D rotary embedding in place on x [tokens][heads][hd] (hd % 4 == 0): the first hd/2 channels rotate by the patch x position, the
// last hd/2 by its y position; within each half the rotate-half pairing (i, i + hd/4) with inv_freq[i] = 1 / theta^(4i/hd).
int rope2d_cuda(float* d_x, int tokens, int heads, int hd, const int* d_px, const int* d_py, float theta, void* stream = nullptr);

// Bidirectional (non-causal) multi-head attention over all `tokens`: out = softmax(scale * q k^T) v per head; q, k, v, out [tokens][heads*hd],
// hd 64 or 72. q/k/v rows may be strided (ld, in floats; 0 = heads * hd).
int vision_attention_cuda(const float* d_q, const float* d_k, const float* d_v, int tokens, int heads, float scale, float* d_out, void* stream = nullptr,
                          int hd = 64, size_t ld_qkv = 0);

// Qwen vision 2D RoPE in place on x (rows of `ld` floats, `heads` heads of hd): rotate-half over the whole head; the first hd/4 frequencies
// use the row position, the next hd/4 the column position; inv_freq[k] = 1 / theta^(2k / (hd/2)).
int rope2d_rh_cuda(float* d_x, size_t ld, int tokens, int heads, int hd, const int* d_row, const int* d_col, float theta, void* stream = nullptr);

// LayerNorm with weight and bias over rows of n: out = (x - mean) / sqrt(var + eps) * w + b (biased variance, as torch). out may alias x.
int layernorm_cuda(const float* d_x, const float* d_w, const float* d_b, int rows, int n, float eps, float* d_out, void* stream = nullptr);
// y[r][i] += b[i] for rows of n
int bias_add_cuda(float* d_y, const float* d_b, int rows, int n, void* stream = nullptr);
// in place: gelu with the tanh approximation (gelu_pytorch_tanh) / the exact erf form (nn.GELU)
int gelu_tanh_cuda(float* d_x, size_t n, void* stream = nullptr);
int gelu_erf_cuda(float* d_x, size_t n, void* stream = nullptr);

}  // namespace dara::model
