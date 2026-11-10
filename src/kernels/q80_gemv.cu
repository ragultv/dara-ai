// Fused Q8_0 dequantize + GEMV. One warp per row; lane i takes blocks i, i+32, ... (34 bytes each, only
// 2-byte aligned, so weights are read as 16-bit pairs). weight = d * q is exact in FP32 (11-bit x 8-bit);
// accumulation is FP32. The real Q8_0 tensors in Qwen3.5-4B are 32-row projections, so this kernel is
// launch-bound and deliberately simple.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q8_0.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) {
    return __ldg(reinterpret_cast<const unsigned short*>(p));
}

__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q80_gemv_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x,
                float* __restrict__ y, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;

    const uint8_t* rp = W + row * row_bytes;
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const uint8_t* p = rp + static_cast<size_t>(b) * kQ80BlockBytes;
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(ld16(p))));
        const float2* xb = reinterpret_cast<const float2*>(x + static_cast<size_t>(b) * 32);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t v = ld16(p + 2 + 2 * k);
            const float2 xv = xb[k];
            acc = fmaf(d * static_cast<float>(static_cast<int8_t>(v & 0xffu)), xv.x, acc);
            acc = fmaf(d * static_cast<float>(static_cast<int8_t>(v >> 8)), xv.y, acc);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

}  // namespace

int gemv_q80_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols,
                  void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0) return 0;
    if (!dW || !dx || !dy || cols == 0 || cols % kQ80BlockWeights != 0 || cols > static_cast<size_t>(INT_MAX) ||
        ld_blocks < cols / kQ80BlockWeights)
        return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 2 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    q80_gemv_kernel<<<blocks, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW),
                                                           ld_blocks * kQ80BlockBytes, dx, dy, rows,
                                                           static_cast<int>(cols / kQ80BlockWeights));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
