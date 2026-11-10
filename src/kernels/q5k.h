#pragma once
// GGUF Q5_K, bit-exact with ggml's block_q5_K (256 weights, 176 bytes, 5.5 bits/weight):
//   d, dmin : fp16 super-block scale and min
//   scales  : 12 bytes with eight 6-bit scales and mins (same packing as Q4_K, ggml get_scale_min_k4)
//   qh      : 32 bytes; the 5th bit of every weight
//   qs      : 128 bytes of low nibbles, laid out like Q4_K
//   Chunk c (0..3) covers weights 64c..64c+63. For l in 0..31:
//     weight 64c + l      : q = (qs[32c + l] & 0xF) | (((qh[l] >> (2c))     & 1) << 4), sub-block 2c
//     weight 64c + 32 + l : q = (qs[32c + l] >> 4)  | (((qh[l] >> (2c + 1)) & 1) << 4), sub-block 2c+1
//   weight = d*scale[sub]*q - dmin*min[sub]
// Rows must be a whole number of blocks (cols % 256 == 0). Blocks are 16-byte aligned if the base is.
#include <cstddef>
#include <cstdint>

namespace dara::kernels {

constexpr int kQ5KBlockWeights = 256;
constexpr int kQ5KBlockBytes = 176;

struct BlockQ5K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ5K) == kQ5KBlockBytes, "BlockQ5K must match ggml's block_q5_K");

void q5k_dequant(const BlockQ5K* b, size_t nblocks, float* out);      // ggml float arithmetic
void q5k_dequant_f64(const BlockQ5K* b, size_t nblocks, double* out);  // exact
void q5k_quantize(const float* x, size_t cols, BlockQ5K* out);         // simple, valid, not ggml's search

#ifdef __CUDACC__
// W and x must be 16-byte aligned. Returns cudaError_t as int; 0 on success.
int gemv_q5k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream = nullptr);
#endif

}  // namespace dara::kernels
