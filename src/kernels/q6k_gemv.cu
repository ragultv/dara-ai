// Fused Q6_K dequantize + GEMV for decode. One warp per output row; the 32 lanes split into two
// half-warps that take alternating blocks (b = half-warp, half-warp + 2, ...). Within a block, lane r
// (0..15) handles half h = r/8 of the block and four consecutive positions l0 = 4*(r%8): it reads one
// 32-bit group of ql[l0..], ql[l0+32..] and qh[l0..], which together hold 16 weights
// (y[l], y[l+32], y[l+64], y[l+96] for four values of l). Weights are decoded in registers with ggml's
// exact arithmetic ((d*scale) * (q-32), one rounding) and multiplied with FP32 activations; accumulation
// is FP32. The matrix is never expanded in memory.
//
// Q6_K blocks are 210 bytes, so a block address is only 2-byte aligned. All weight loads are therefore
// 16-bit (two per 32-bit group): no over-read, no misaligned access, any even base address works.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q6k.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) {
    return __ldg(reinterpret_cast<const unsigned short*>(p));
}
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }

// Byte K of a packed word of 6-bit values -> float(q - 32), exactly: (2^23 + q) - (2^23 + 32).
template <int K>
__device__ __forceinline__ float qc(uint32_t packed) {
    return __uint_as_float(__byte_perm(packed, 0x4B000000u, 0x7650 | K)) - 8388640.0f;
}

__device__ __forceinline__ int sbyte(uint32_t w, int idx) { return static_cast<int>(static_cast<int8_t>((w >> (8 * idx)) & 0xffu)); }

__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q6k_gemv_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x,
                float* __restrict__ y, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;  // whole warp exits together

    const uint8_t* rp = W + row * row_bytes;
    const int g = lane >> 4;        // which block of the pair
    const int r = lane & 15;
    const int h = r >> 3;           // which 128-weight half of the block
    const int l0 = 4 * (r & 7);     // first of four consecutive positions
    const int is = (r & 7) >> 2;    // l0 / 16: scale pair index
    float acc = 0.0f;

#pragma unroll 2
    for (int b = g; b < nblk; b += 2) {
        const uint8_t* p = rp + static_cast<size_t>(b) * kQ6KBlockBytes;
        const uint32_t A = ld32u(p + 64 * h + l0);          // ql[l0..l0+3]
        const uint32_t B = ld32u(p + 64 * h + l0 + 32);     // ql[l0+32..]
        const uint32_t H = ld32u(p + 128 + 32 * h + l0);    // qh[l0..l0+3]
        const uint32_t s03 = ld32u(p + 192 + 8 * h);        // scales[8h + 0..3]
        const uint32_t s47 = ld32u(p + 192 + 8 * h + 4);    // scales[8h + 4..7]
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(ld16(p + 208))));

        const float dsc0 = d * static_cast<float>(sbyte(s03, is));       // exact: 11-bit * 8-bit
        const float dsc1 = d * static_cast<float>(sbyte(s03, is + 2));
        const float dsc2 = d * static_cast<float>(sbyte(s47, is));
        const float dsc3 = d * static_cast<float>(sbyte(s47, is + 2));

        // Four packed words of 6-bit values, one byte per position: weights y[l], y[l+32], y[l+64], y[l+96].
        const uint32_t Q0 = (A & 0x0F0F0F0Fu) | ((H & 0x03030303u) << 4);
        const uint32_t Q1 = (B & 0x0F0F0F0Fu) | (((H >> 2) & 0x03030303u) << 4);
        const uint32_t Q2 = ((A >> 4) & 0x0F0F0F0Fu) | (((H >> 4) & 0x03030303u) << 4);
        const uint32_t Q3 = ((B >> 4) & 0x0F0F0F0Fu) | (((H >> 6) & 0x03030303u) << 4);

        const float* xb = x + static_cast<size_t>(b) * 256 + 128 * h + l0;
        const float4 x0 = *reinterpret_cast<const float4*>(xb);
        const float4 x1 = *reinterpret_cast<const float4*>(xb + 32);
        const float4 x2 = *reinterpret_cast<const float4*>(xb + 64);
        const float4 x3 = *reinterpret_cast<const float4*>(xb + 96);

        acc = fmaf(dsc0 * qc<0>(Q0), x0.x, acc);
        acc = fmaf(dsc0 * qc<1>(Q0), x0.y, acc);
        acc = fmaf(dsc0 * qc<2>(Q0), x0.z, acc);
        acc = fmaf(dsc0 * qc<3>(Q0), x0.w, acc);
        acc = fmaf(dsc1 * qc<0>(Q1), x1.x, acc);
        acc = fmaf(dsc1 * qc<1>(Q1), x1.y, acc);
        acc = fmaf(dsc1 * qc<2>(Q1), x1.z, acc);
        acc = fmaf(dsc1 * qc<3>(Q1), x1.w, acc);
        acc = fmaf(dsc2 * qc<0>(Q2), x2.x, acc);
        acc = fmaf(dsc2 * qc<1>(Q2), x2.y, acc);
        acc = fmaf(dsc2 * qc<2>(Q2), x2.z, acc);
        acc = fmaf(dsc2 * qc<3>(Q2), x2.w, acc);
        acc = fmaf(dsc3 * qc<0>(Q3), x3.x, acc);
        acc = fmaf(dsc3 * qc<1>(Q3), x3.y, acc);
        acc = fmaf(dsc3 * qc<2>(Q3), x3.z, acc);
        acc = fmaf(dsc3 * qc<3>(Q3), x3.w, acc);
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

}  // namespace

int gemv_q6k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0) return 0;
    if (!dW || !dx || !dy || cols == 0 || cols % kQ6KBlockWeights != 0 || cols > static_cast<size_t>(INT_MAX) ||
        ld_blocks < cols / kQ6KBlockWeights)
        return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 2 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    q6k_gemv_kernel<<<blocks, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW),
                                                           ld_blocks * kQ6KBlockBytes, dx, dy, rows,
                                                           static_cast<int>(cols / kQ6KBlockWeights));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
