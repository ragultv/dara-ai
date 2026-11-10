#pragma once
// GGUF Q4_K, bit-exact with ggml's block_q4_K (256 weights, 144 bytes, 4.5 bits/weight):
//   d, dmin : fp16 super-block scale and min
//   scales  : 12 bytes holding eight 6-bit scales and eight 6-bit mins (ggml get_scale_min_k4 packing)
//   qs      : 128 bytes, 256 4-bit values. Bytes are used in four chunks of 32; within chunk c the low
//             nibbles are sub-block 2c (weights 64c..64c+31) and the high nibbles are sub-block 2c+1
//             (weights 64c+32..64c+63).
//   weight  = d*scale[sub]*q - dmin*min[sub]
// Rows must be a whole number of blocks (cols % 256 == 0), exactly as ggml requires.
#include <cstddef>
#include <cstdint>

namespace dara::kernels {

constexpr int kQ4KBlockWeights = 256;
constexpr int kQ4KBlockBytes = 144;

struct BlockQ4K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(BlockQ4K) == kQ4KBlockBytes, "BlockQ4K must match ggml's block_q4_K");

// Unpack the 6-bit scale and min of sub-block j (0..7). Same as ggml's get_scale_min_k4.
inline void q4k_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Dequantize `nblocks` blocks with ggml's float arithmetic (what a CPU engine would produce).
void q4k_dequant(const BlockQ4K* b, size_t nblocks, float* out);
// Same weights as exact doubles: d*scale*q - dmin*min with no intermediate rounding.
void q4k_dequant_f64(const BlockQ4K* b, size_t nblocks, double* out);

// Simple per-sub-block min/max quantizer that produces a valid Q4_K encoding. It is NOT ggml's
// error-minimizing search, so its quantization error is somewhat higher than llama.cpp's.
void q4k_quantize(const float* x, size_t cols, BlockQ4K* out);

// y[r] = sum_c W[r][c] * x[c]; row r starts at W + r*ld_blocks. FP32 accumulation, sequential.
void gemv_q4k_cpu_f32(const BlockQ4K* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols);
// Double-precision truth plus sum|w*x| per row (error scale).
void gemv_q4k_cpu_f64(const BlockQ4K* W, size_t ld_blocks, const float* x, double* y, double* sum_abs, size_t rows,
                      size_t cols);

#ifdef __CUDACC__
// Device pointers; W and ld_blocks*144 must be 16-byte aligned (always true for GGUF data).
// Returns a cudaError_t as int; 0 on success. Asynchronous on `stream`.
int gemv_q4k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream = nullptr);
#endif

}  // namespace dara::kernels
