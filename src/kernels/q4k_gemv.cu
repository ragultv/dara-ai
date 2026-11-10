// Fused Q4_K dequantize + GEMV for decode. One warp per output row. For each 144-byte block the 32
// lanes each take one 32-bit word of the 128 quant bytes (4 bytes = 8 weights: 4 low nibbles from
// sub-block 2c and the 4 matching high nibbles from sub-block 2c+1, where c = lane / 8). Weights are
// decoded in registers (d*scale*q - dmin*min, ggml's exact arithmetic) and multiplied with FP32
// activations; accumulation is FP32. The matrix is never expanded in memory.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;

constexpr float k2p23 = 8388608.0f;  // 2^23

// Exact small-int -> float without the slow I2F instruction: (2^23 + v) - 2^23, v < 2^23.
__device__ __forceinline__ float small2f(uint32_t v) { return __uint_as_float(0x4B000000u | v) - k2p23; }

// Nibble k (0..3) of a byte-masked word -> float. One PRMT builds the bit pattern 0x4B0000nn.
template <int K>
__device__ __forceinline__ float nib(uint32_t masked) {
    return __uint_as_float(__byte_perm(masked, 0x4B000000u, 0x7650 | K)) - k2p23;
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q4k_gemv_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x,
                float* __restrict__ y, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;  // whole warp exits together

    const uint8_t* rp = W + row * row_bytes;
    const int c = lane >> 3;                   // chunk of 32 quant bytes this lane reads
    const int xoff = 64 * c + (lane & 7) * 4;  // low-nibble activations; +32 for the high nibbles
    // Sub-blocks j0 = 2c, j1 = 2c+1. Scale bytes live at index (j & 3) of the three scale words;
    // for j >= 4 (c >= 2) the 6-bit values are split across words (ggml get_scale_min_k4).
    const int sh0 = 8 * ((2 * c) & 3), sh1 = 8 * ((2 * c + 1) & 3);
    const bool hi_pair = c >= 2;
    float acc = 0.0f;

#pragma unroll 4
    for (int b = 0; b < nblk; ++b) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kQ4KBlockBytes;
        const uint4 hdr = *reinterpret_cast<const uint4*>(blk);                       // d, dmin, scales[12]
        const uint32_t qw = *reinterpret_cast<const uint32_t*>(blk + 16 + 4 * lane);  // 4 quant bytes
        const float4 xlo = *reinterpret_cast<const float4*>(x + b * 256 + xoff);
        const float4 xhi = *reinterpret_cast<const float4*>(x + b * 256 + xoff + 32);

        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));

        // Branchless scale/min decode for both sub-blocks (hdr.y/z/w = scale bytes 0-3 / 4-7 / 8-11).
        const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
        const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
        const uint32_t s0 = hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
        const uint32_t m0 = hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
        const uint32_t s1 = hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
        const uint32_t m1v = hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);
        const float d1 = d * small2f(s0), m1 = dmin * small2f(m0);  // <=11-bit x 6-bit: exact in FP32
        const float d2 = d * small2f(s1), m2 = dmin * small2f(m1v);

        const uint32_t lo4 = qw & 0x0F0F0F0Fu;
        const uint32_t hi4 = (qw >> 4) & 0x0F0F0F0Fu;
        acc = fmaf(fmaf(d1, nib<0>(lo4), -m1), xlo.x, acc);
        acc = fmaf(fmaf(d2, nib<0>(hi4), -m2), xhi.x, acc);
        acc = fmaf(fmaf(d1, nib<1>(lo4), -m1), xlo.y, acc);
        acc = fmaf(fmaf(d2, nib<1>(hi4), -m2), xhi.y, acc);
        acc = fmaf(fmaf(d1, nib<2>(lo4), -m1), xlo.z, acc);
        acc = fmaf(fmaf(d2, nib<2>(hi4), -m2), xhi.z, acc);
        acc = fmaf(fmaf(d1, nib<3>(lo4), -m1), xlo.w, acc);
        acc = fmaf(fmaf(d2, nib<3>(hi4), -m2), xhi.w, acc);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

}  // namespace

int gemv_q4k_cuda_ref(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0) return 0;
    if (!dW || !dx || !dy || cols == 0 || cols % kQ4KBlockWeights != 0 || cols > static_cast<size_t>(INT_MAX) ||
        ld_blocks < cols / kQ4KBlockWeights)
        return static_cast<int>(cudaErrorInvalidValue);
    // The kernel issues 16-byte loads on block headers and on x; refuse anything else.
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    q4k_gemv_kernel<<<blocks, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW),
                                                           ld_blocks * kQ4KBlockBytes, dx, dy, rows,
                                                           static_cast<int>(cols / kQ4KBlockWeights));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
