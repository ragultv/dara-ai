#pragma once
// GGUF Q8_0, bit-exact with ggml's block_q8_0 (32 weights, 34 bytes, 8.5 bits/weight):
//   d  : fp16 scale
//   qs : 32 signed 8-bit values;  weight = d * qs[i]
// Rows must be a whole number of blocks (cols % 32 == 0). Blocks are only 2-byte aligned.
#include <cstddef>
#include <cstdint>

namespace dara::kernels {

constexpr int kQ80BlockWeights = 32;
constexpr int kQ80BlockBytes = 34;

struct BlockQ80 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ80) == kQ80BlockBytes, "BlockQ80 must match ggml's block_q8_0 (and be unpadded)");

void q80_dequant(const BlockQ80* b, size_t nblocks, float* out);      // d * q: exact in FP32
void q80_dequant_f64(const BlockQ80* b, size_t nblocks, double* out);
void q80_quantize(const float* x, size_t cols, BlockQ80* out);

#ifdef __CUDACC__
// W must be 2-byte aligned and x 16-byte aligned. Returns cudaError_t as int; 0 on success.
int gemv_q80_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream = nullptr);
#endif

}  // namespace dara::kernels
