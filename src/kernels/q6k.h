#pragma once
// GGUF Q6_K, bit-exact with ggml's block_q6_K (256 weights, 210 bytes, 6.5625 bits/weight):
//   ql[128] : low 4 bits of each weight
//   qh[64]  : high 2 bits of each weight
//   scales[16] : int8 scale per sub-block of 16 weights
//   d       : fp16 super-block scale
//   weight  = d * scales[sub] * (q - 32),  q = low4 | (high2 << 4)  in 0..63
// Each 128-weight half (n = 0, 128) uses ql[64n/128..], qh[32n/128..], scales[8n/128..]. For position l in
// 0..31 of a half, four weights y[l], y[l+32], y[l+64], y[l+96] come from ql[l], ql[l+32] (low and high
// nibbles of each) and the four 2-bit fields of qh[l]; their scales are sc[l/16 + {0,2,4,6}].
// A block is 210 bytes, so blocks are only 2-byte aligned. Rows must be a whole number of blocks.
#include <cstddef>
#include <cstdint>

namespace dara::kernels {

constexpr int kQ6KBlockWeights = 256;
constexpr int kQ6KBlockBytes = 210;

struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6K) == kQ6KBlockBytes, "BlockQ6K must match ggml's block_q6_K (and be unpadded)");
static_assert(alignof(BlockQ6K) == 2, "Q6_K blocks are 2-byte aligned");

// ggml float arithmetic: (d * scale) * (q - 32), rounded once.
void q6k_dequant(const BlockQ6K* b, size_t nblocks, float* out);
// Exact doubles.
void q6k_dequant_f64(const BlockQ6K* b, size_t nblocks, double* out);

// Simple symmetric quantizer producing a valid Q6_K encoding (not ggml's error-minimizing search).
void q6k_quantize(const float* x, size_t cols, BlockQ6K* out);

void gemv_q6k_cpu_f32(const BlockQ6K* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols);
void gemv_q6k_cpu_f64(const BlockQ6K* W, size_t ld_blocks, const float* x, double* y, double* sum_abs, size_t rows,
                      size_t cols);

#ifdef __CUDACC__
// Device pointers; W must be 2-byte aligned and x 16-byte aligned. Returns cudaError_t as int; 0 on success.
int gemv_q6k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream = nullptr);
#endif

}  // namespace dara::kernels
