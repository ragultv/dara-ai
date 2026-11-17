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

// Bidirectional (non-causal) multi-head attention over all `tokens`: out = softmax(scale * q k^T) v per head; q, k, v, out [tokens][heads*64].
int vision_attention_cuda(const float* d_q, const float* d_k, const float* d_v, int tokens, int heads, float scale, float* d_out, void* stream = nullptr);

}  // namespace dara::model
