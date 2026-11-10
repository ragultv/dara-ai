#pragma once
// CUDA RMSNorm and RoPE (correctness-first; see ops.h for the exact definitions and weight conventions).
#include <cstddef>
#include <cstdint>

namespace dara::model {

// y[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * scale, for `rows` rows of length n. scale is the EFFECTIVE multiplier
// (the GGUF tensor as-is for the folded (1+w) norms). x and y may alias. Returns cudaError_t as int.
int rmsnorm_cuda(const float* d_x, const float* d_scale, int rows, int n, float eps, float* d_y, void* stream = nullptr);

// In-place partial NeoX RoPE on x[tokens][heads][head_dim]: rotates the first rot_dim dims with
// angle = float32(pos[t]) * inv_freq[i], cos/sin in float32 (the reference's arithmetic). d_pos: [tokens] int64.
int rope_cuda(float* d_x, int tokens, int heads, int head_dim, int rot_dim, const int64_t* d_pos, const float* d_inv_freq,
              void* stream = nullptr);

// out[i] = a[i] + b[i]  (residual connection). out may alias a or b.
int add_cuda(const float* d_a, const float* d_b, float* d_out, size_t n, void* stream = nullptr);

// out[i] = silu(gate[i]) * up[i]  (the MLP's gated activation). out may alias gate.
int swiglu_cuda(const float* d_gate, const float* d_up, float* d_out, size_t n, void* stream = nullptr);

// Index of the largest of n floats (ties -> lowest index: for finite values the same as a host `>` scan), written to *d_out on the device.
// d_scratch: argmax_scratch_bytes() device bytes.
size_t argmax_scratch_bytes();
int argmax_cuda(const float* d_x, int n, void* d_scratch, int* d_out, void* stream = nullptr);

}  // namespace dara::model
