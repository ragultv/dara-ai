// Decode GEMV, first correct version. One warp per output row; lanes stride across the row so
// consecutive lanes read consecutive memory. Weights are 16-bit (FP16 or BF16), x/y FP32,
// accumulation FP32. Deliberately unoptimized beyond 16-byte vector loads: this is the
// correctness and bandwidth baseline.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/gemv.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;

template <WeightType T>
__device__ __forceinline__ float cvt(uint16_t h) {
    if constexpr (T == WeightType::BF16) return __uint_as_float(static_cast<uint32_t>(h) << 16);
    else return __half2float(__ushort_as_half(h));
}

// Two packed 16-bit weights -> two floats (low element first).
template <WeightType T>
__device__ __forceinline__ float2 cvt2(uint32_t p) {
    if constexpr (T == WeightType::BF16)
        return make_float2(__uint_as_float(p << 16), __uint_as_float(p & 0xffff0000u));
    else
        return __half22float2(*reinterpret_cast<const __half2*>(&p));
}

template <WeightType T, bool Vec>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
gemv_kernel(const uint16_t* __restrict__ W, size_t ld, const float* __restrict__ x, float* __restrict__ y,
            size_t rows, int cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;  // whole warp exits together; no block-level sync below

    const uint16_t* w = W + row * ld;
    float acc = 0.0f;
    int tail_start = 0;

    if constexpr (Vec) {
        // Requires: w 16-byte aligned (W and ld*2 multiples of 16), x 16-byte aligned.
        const int n8 = cols >> 3;
        const uint4* w4 = reinterpret_cast<const uint4*>(w);
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = lane; i < n8; i += 32) {
            const uint4 v = __ldg(w4 + i);
            const float4 xa = x4[2 * i];
            const float4 xb = x4[2 * i + 1];
            float2 a = cvt2<T>(v.x), b = cvt2<T>(v.y), c = cvt2<T>(v.z), d = cvt2<T>(v.w);
            acc = fmaf(a.x, xa.x, acc); acc = fmaf(a.y, xa.y, acc);
            acc = fmaf(b.x, xa.z, acc); acc = fmaf(b.y, xa.w, acc);
            acc = fmaf(c.x, xb.x, acc); acc = fmaf(c.y, xb.y, acc);
            acc = fmaf(d.x, xb.z, acc); acc = fmaf(d.y, xb.w, acc);
        }
        tail_start = n8 * 8;
    }
    for (int c = tail_start + lane; c < cols; c += 32) acc = fmaf(cvt<T>(w[c]), x[c], acc);

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

template <WeightType T>
int launch(const uint16_t* W, size_t ld, const float* x, float* y, size_t rows, int cols, cudaStream_t s) {
    const bool vec = reinterpret_cast<uintptr_t>(W) % 16 == 0 && (ld % 8) == 0 &&
                     reinterpret_cast<uintptr_t>(x) % 16 == 0;
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const dim3 block(kWarpsPerBlock * 32);
    if (vec) gemv_kernel<T, true><<<blocks, block, 0, s>>>(W, ld, x, y, rows, cols);
    else gemv_kernel<T, false><<<blocks, block, 0, s>>>(W, ld, x, y, rows, cols);
    return static_cast<int>(cudaGetLastError());
}

// Multi-vector variant: y[t][r] = sum_c W[r][c] * x[t][c] for NT vectors, reading each weight once. Same per-vector accumulation order as the
// vectorised single-vector kernel above, so every output is bitwise identical to gemv_cuda on that vector. Requires the Vec alignment conditions.
template <WeightType T, int NT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
gemv_multi_kernel(const uint16_t* __restrict__ W, size_t ld, const float* __restrict__ x, size_t ldx, float* __restrict__ y, size_t ldy, size_t rows, int cols) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint16_t* w = W + row * ld;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    const int n8 = cols >> 3;
    const uint4* w4 = reinterpret_cast<const uint4*>(w);
    for (int i = lane; i < n8; i += 32) {
        const uint4 v = __ldg(w4 + i);
        const float2 a = cvt2<T>(v.x), b = cvt2<T>(v.y), c = cvt2<T>(v.z), d = cvt2<T>(v.w);
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* x4 = reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * ldx);
            const float4 xa = x4[2 * i], xb = x4[2 * i + 1];
            float r = acc[t];
            r = fmaf(a.x, xa.x, r); r = fmaf(a.y, xa.y, r);
            r = fmaf(b.x, xa.z, r); r = fmaf(b.y, xa.w, r);
            r = fmaf(c.x, xb.x, r); r = fmaf(c.y, xb.y, r);
            r = fmaf(d.x, xb.z, r); r = fmaf(d.y, xb.w, r);
            acc[t] = r;
        }
    }
    for (int c = n8 * 8 + lane; c < cols; c += 32) {
        const float wv = cvt<T>(w[c]);
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[t] = fmaf(wv, x[static_cast<size_t>(t) * ldx + c], acc[t]);
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float a = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = a;
    }
}

template <WeightType T, int NT>
int launch_multi(const uint16_t* W, size_t ld, const float* x, size_t ldx, float* y, size_t ldy, size_t rows, int cols, cudaStream_t s) {
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gemv_multi_kernel<T, NT><<<blocks, kWarpsPerBlock * 32, 0, s>>>(W, ld, x, ldx, y, ldy, rows, cols);
    return static_cast<int>(cudaGetLastError());
}

template <WeightType T>
int launch_multi_n(int n, const uint16_t* W, size_t ld, const float* x, size_t ldx, float* y, size_t ldy, size_t rows, int cols, cudaStream_t s) {
    switch (n) {
        case 1: return launch_multi<T, 1>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 2: return launch_multi<T, 2>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 3: return launch_multi<T, 3>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 4: return launch_multi<T, 4>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 5: return launch_multi<T, 5>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 6: return launch_multi<T, 6>(W, ld, x, ldx, y, ldy, rows, cols, s);
        case 7: return launch_multi<T, 7>(W, ld, x, ldx, y, ldy, rows, cols, s);
        default: return launch_multi<T, 8>(W, ld, x, ldx, y, ldy, rows, cols, s);
    }
}

}  // namespace

int gemv_multi_cuda(WeightType t, const void* dW, size_t ld, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int nt, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0 || nt == 0) return 0;
    if (nt < 0 || cols > static_cast<size_t>(INT_MAX) || ld < cols || ldx < cols || !dW || !dx || !dy || cols == 0) return static_cast<int>(cudaErrorInvalidValue);
    const bool vec = reinterpret_cast<uintptr_t>(dW) % 16 == 0 && (ld % 8) == 0 && reinterpret_cast<uintptr_t>(dx) % 16 == 0 && ldx % 4 == 0;
    if (!vec) return static_cast<int>(cudaErrorMisalignedAddress);  // callers fall back to the single-vector kernel
    auto W = static_cast<const uint16_t*>(dW);
    for (int t0 = 0; t0 < nt; t0 += 8) {
        const int n = nt - t0 < 8 ? nt - t0 : 8;
        const float* xg = dx + static_cast<size_t>(t0) * ldx;
        float* yg = dy + static_cast<size_t>(t0) * ldy;
        const int e = t == WeightType::BF16 ? launch_multi_n<WeightType::BF16>(n, W, ld, xg, ldx, yg, ldy, rows, static_cast<int>(cols), s)
                                            : launch_multi_n<WeightType::F16>(n, W, ld, xg, ldx, yg, ldy, rows, static_cast<int>(cols), s);
        if (e != 0) return e;
    }
    return 0;
}

int gemv_cuda(WeightType t, const void* dW, size_t ld, const float* dx, float* dy, size_t rows, size_t cols,
              void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0) return 0;
    if (cols > static_cast<size_t>(INT_MAX) || ld < cols || !dW || !dx || !dy)
        return static_cast<int>(cudaErrorInvalidValue);
    if (cols == 0) return static_cast<int>(cudaMemsetAsync(dy, 0, rows * sizeof(float), s));
    auto W = static_cast<const uint16_t*>(dW);
    return t == WeightType::BF16 ? launch<WeightType::BF16>(W, ld, dx, dy, rows, static_cast<int>(cols), s)
                                 : launch<WeightType::F16>(W, ld, dx, dy, rows, static_cast<int>(cols), s);
}

}  // namespace dara::kernels
