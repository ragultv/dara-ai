// Multi-row fused dequantize + GEMV: y[t] = W * x[t] for NT activation vectors (t < NT <= 8) while reading each weight block ONCE.
// Decode reads the whole model per token; verifying/prefilling several tokens with the single-row kernels would read it per token.
// These kernels decode every weight in registers exactly as the single-row kernels do and apply it to all NT vectors, with the
// same per-vector accumulation order, so every output row is BITWISE identical to the single-row kernel's.
// x[t] = x + t*ldx (ldx multiple of 4 floats), y[t] = y + t*ldy.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/q6k.h"
#include "kernels/q8_0.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr float k2p23 = 8388608.0f;

__device__ __forceinline__ float small2f(uint32_t v) { return __uint_as_float(0x4B000000u | v) - k2p23; }
template <int K>
__device__ __forceinline__ float nib(uint32_t masked) { return __uint_as_float(__byte_perm(masked, 0x4B000000u, 0x7650 | K)) - k2p23; }

// ---- Q4_K ------------------------------------------------------------------------------------------------------------------
// R rows per warp: the activation loads (the bottleneck once several vectors share a weight) are shared by the R rows.
template <int NT, int R>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q4k_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x, size_t ldx, float* __restrict__ y,
                 size_t ldy, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row0 = (static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp) * R;
    if (row0 >= rows) return;

    const uint8_t* rp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rp[r] = W + (row0 + r < rows ? row0 + r : row0) * row_bytes;  // tail rows reuse a valid row; not stored
    const int c = lane >> 3;
    const int xoff = 64 * c + (lane & 7) * 4;
    const int sh0 = 8 * ((2 * c) & 3), sh1 = 8 * ((2 * c + 1) & 3);
    const bool hi_pair = c >= 2;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

#pragma unroll 1
    for (int b = 0; b < nblk; ++b) {
        float wl[R][4], wh[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = rp[r] + static_cast<size_t>(b) * kQ4KBlockBytes;
            const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            const uint32_t qw = *reinterpret_cast<const uint32_t*>(blk + 16 + 4 * lane);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
            const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
            const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
            const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
            const uint32_t s0 = hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
            const uint32_t m0 = hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
            const uint32_t s1 = hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
            const uint32_t m1v = hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);
            const float d1 = d * small2f(s0), m1 = dmin * small2f(m0);
            const float d2 = d * small2f(s1), m2 = dmin * small2f(m1v);
            const uint32_t lo4 = qw & 0x0F0F0F0Fu;
            const uint32_t hi4 = (qw >> 4) & 0x0F0F0F0Fu;
            wl[r][0] = fmaf(d1, nib<0>(lo4), -m1); wh[r][0] = fmaf(d2, nib<0>(hi4), -m2);
            wl[r][1] = fmaf(d1, nib<1>(lo4), -m1); wh[r][1] = fmaf(d2, nib<1>(hi4), -m2);
            wl[r][2] = fmaf(d1, nib<2>(lo4), -m1); wh[r][2] = fmaf(d2, nib<2>(hi4), -m2);
            wl[r][3] = fmaf(d1, nib<3>(lo4), -m1); wh[r][3] = fmaf(d2, nib<3>(hi4), -m2);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float* xt = x + static_cast<size_t>(t) * ldx + b * 256 + xoff;
            const float4 xlo = *reinterpret_cast<const float4*>(xt);
            const float4 xhi = *reinterpret_cast<const float4*>(xt + 32);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = acc[r][t];
                a = fmaf(wl[r][0], xlo.x, a); a = fmaf(wh[r][0], xhi.x, a);
                a = fmaf(wl[r][1], xlo.y, a); a = fmaf(wh[r][1], xhi.y, a);
                a = fmaf(wl[r][2], xlo.z, a); a = fmaf(wh[r][2], xhi.z, a);
                a = fmaf(wl[r][3], xlo.w, a); a = fmaf(wh[r][3], xhi.w, a);
                acc[r][t] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0 && row0 + r < rows) y[static_cast<size_t>(t) * ldy + row0 + r] = a;
        }
}

// ---- Q6_K ------------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t ld16(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned short*>(p)); }
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) { return ld16(p) | (ld16(p + 2) << 16); }
template <int K>
__device__ __forceinline__ float qc(uint32_t packed) { return __uint_as_float(__byte_perm(packed, 0x4B000000u, 0x7650 | K)) - 8388640.0f; }
__device__ __forceinline__ int sbyte(uint32_t w, int idx) { return static_cast<int>(static_cast<int8_t>((w >> (8 * idx)) & 0xffu)); }

template <int NT, int R>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q6k_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x, size_t ldx, float* __restrict__ y,
                 size_t ldy, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row0 = (static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp) * R;
    if (row0 >= rows) return;

    const uint8_t* rp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rp[r] = W + (row0 + r < rows ? row0 + r : row0) * row_bytes;
    const int g = lane >> 4;
    const int rr = lane & 15;
    const int h = rr >> 3;
    const int l0 = 4 * (rr & 7);
    const int is = (rr & 7) >> 2;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

#pragma unroll 1
    for (int b = g; b < nblk; b += 2) {
        float w[R][16];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* p = rp[r] + static_cast<size_t>(b) * kQ6KBlockBytes;
            const uint32_t A = ld32u(p + 64 * h + l0);
            const uint32_t B = ld32u(p + 64 * h + l0 + 32);
            const uint32_t H = ld32u(p + 128 + 32 * h + l0);
            const uint32_t s03 = ld32u(p + 192 + 8 * h);
            const uint32_t s47 = ld32u(p + 192 + 8 * h + 4);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(ld16(p + 208))));
            const float dsc0 = d * static_cast<float>(sbyte(s03, is));
            const float dsc1 = d * static_cast<float>(sbyte(s03, is + 2));
            const float dsc2 = d * static_cast<float>(sbyte(s47, is));
            const float dsc3 = d * static_cast<float>(sbyte(s47, is + 2));
            const uint32_t Q0 = (A & 0x0F0F0F0Fu) | ((H & 0x03030303u) << 4);
            const uint32_t Q1 = (B & 0x0F0F0F0Fu) | (((H >> 2) & 0x03030303u) << 4);
            const uint32_t Q2 = ((A >> 4) & 0x0F0F0F0Fu) | (((H >> 4) & 0x03030303u) << 4);
            const uint32_t Q3 = ((B >> 4) & 0x0F0F0F0Fu) | (((H >> 6) & 0x03030303u) << 4);
            w[r][0] = dsc0 * qc<0>(Q0); w[r][1] = dsc0 * qc<1>(Q0); w[r][2] = dsc0 * qc<2>(Q0); w[r][3] = dsc0 * qc<3>(Q0);
            w[r][4] = dsc1 * qc<0>(Q1); w[r][5] = dsc1 * qc<1>(Q1); w[r][6] = dsc1 * qc<2>(Q1); w[r][7] = dsc1 * qc<3>(Q1);
            w[r][8] = dsc2 * qc<0>(Q2); w[r][9] = dsc2 * qc<1>(Q2); w[r][10] = dsc2 * qc<2>(Q2); w[r][11] = dsc2 * qc<3>(Q2);
            w[r][12] = dsc3 * qc<0>(Q3); w[r][13] = dsc3 * qc<1>(Q3); w[r][14] = dsc3 * qc<2>(Q3); w[r][15] = dsc3 * qc<3>(Q3);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float* xb = x + static_cast<size_t>(t) * ldx + static_cast<size_t>(b) * 256 + 128 * h + l0;
            const float4 x0 = *reinterpret_cast<const float4*>(xb);
            const float4 x1 = *reinterpret_cast<const float4*>(xb + 32);
            const float4 x2 = *reinterpret_cast<const float4*>(xb + 64);
            const float4 x3 = *reinterpret_cast<const float4*>(xb + 96);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = acc[r][t];
                a = fmaf(w[r][0], x0.x, a); a = fmaf(w[r][1], x0.y, a); a = fmaf(w[r][2], x0.z, a); a = fmaf(w[r][3], x0.w, a);
                a = fmaf(w[r][4], x1.x, a); a = fmaf(w[r][5], x1.y, a); a = fmaf(w[r][6], x1.z, a); a = fmaf(w[r][7], x1.w, a);
                a = fmaf(w[r][8], x2.x, a); a = fmaf(w[r][9], x2.y, a); a = fmaf(w[r][10], x2.z, a); a = fmaf(w[r][11], x2.w, a);
                a = fmaf(w[r][12], x3.x, a); a = fmaf(w[r][13], x3.y, a); a = fmaf(w[r][14], x3.z, a); a = fmaf(w[r][15], x3.w, a);
                acc[r][t] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0 && row0 + r < rows) y[static_cast<size_t>(t) * ldy + row0 + r] = a;
        }
}

// ---- Q5_K (same structure as Q4_K plus the 5th bit from qh; arithmetic identical to q5k_gemv_kernel per vector) ----------------------------
template <int NT, int R>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q5k_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x, size_t ldx, float* __restrict__ y, size_t ldy, size_t rows,
                 int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row0 = (static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp) * R;
    if (row0 >= rows) return;
    const uint8_t* rp[R];
#pragma unroll
    for (int r = 0; r < R; ++r) rp[r] = W + (row0 + r < rows ? row0 + r : row0) * row_bytes;
    const int c = lane >> 3;
    const int l0 = (lane & 7) * 4;
    const int xoff = 64 * c + l0;
    const int sh0 = 8 * ((2 * c) & 3), sh1 = 8 * ((2 * c + 1) & 3);
    const bool hi_pair = c >= 2;
    const int qh_lo = 2 * c, qh_hi = 2 * c + 1;
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) acc[r][t] = 0.0f;

#pragma unroll 1
    for (int b = 0; b < nblk; ++b) {
        float wl[R][4], wh[R][4];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t* blk = rp[r] + static_cast<size_t>(b) * kQ5KBlockBytes;
            const uint4 hdr = *reinterpret_cast<const uint4*>(blk);
            const uint32_t qhw = *reinterpret_cast<const uint32_t*>(blk + 16 + l0);
            const uint32_t qw = *reinterpret_cast<const uint32_t*>(blk + 48 + 4 * lane);
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
            const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
            const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
            const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
            const uint32_t s0 = hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
            const uint32_t m0 = hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
            const uint32_t s1 = hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
            const uint32_t m1v = hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);
            const float d1 = d * small2f(s0), m1 = dmin * small2f(m0);
            const float d2 = d * small2f(s1), m2 = dmin * small2f(m1v);
            const uint32_t lo5 = (qw & 0x0F0F0F0Fu) | (((qhw >> qh_lo) & 0x01010101u) << 4);
            const uint32_t hi5 = ((qw >> 4) & 0x0F0F0F0Fu) | (((qhw >> qh_hi) & 0x01010101u) << 4);
            wl[r][0] = fmaf(d1, nib<0>(lo5), -m1); wh[r][0] = fmaf(d2, nib<0>(hi5), -m2);
            wl[r][1] = fmaf(d1, nib<1>(lo5), -m1); wh[r][1] = fmaf(d2, nib<1>(hi5), -m2);
            wl[r][2] = fmaf(d1, nib<2>(lo5), -m1); wh[r][2] = fmaf(d2, nib<2>(hi5), -m2);
            wl[r][3] = fmaf(d1, nib<3>(lo5), -m1); wh[r][3] = fmaf(d2, nib<3>(hi5), -m2);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float* xt = x + static_cast<size_t>(t) * ldx + b * 256 + xoff;
            const float4 xlo = *reinterpret_cast<const float4*>(xt);
            const float4 xhi = *reinterpret_cast<const float4*>(xt + 32);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = acc[r][t];
                a = fmaf(wl[r][0], xlo.x, a); a = fmaf(wh[r][0], xhi.x, a);
                a = fmaf(wl[r][1], xlo.y, a); a = fmaf(wh[r][1], xhi.y, a);
                a = fmaf(wl[r][2], xlo.z, a); a = fmaf(wh[r][2], xhi.z, a);
                a = fmaf(wl[r][3], xlo.w, a); a = fmaf(wh[r][3], xhi.w, a);
                acc[r][t] = a;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            float a = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0 && row0 + r < rows) y[static_cast<size_t>(t) * ldy + row0 + r] = a;
        }
}

// ---- Q8_0 (one warp per row, lane i takes blocks i, i+32, ...; arithmetic identical to q80_gemv_kernel per vector) --------------------------
template <int NT>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q80_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ x, size_t ldx, float* __restrict__ y, size_t ldy, size_t rows,
                 int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint8_t* rp = W + row * row_bytes;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const uint8_t* p = rp + static_cast<size_t>(b) * kQ80BlockBytes;
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(__ldg(reinterpret_cast<const unsigned short*>(p)))));
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const uint32_t v = __ldg(reinterpret_cast<const unsigned short*>(p + 2 + 2 * k));
            const float w0 = d * static_cast<float>(static_cast<int8_t>(v & 0xffu)), w1 = d * static_cast<float>(static_cast<int8_t>(v >> 8));
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float2 xv = reinterpret_cast<const float2*>(x + static_cast<size_t>(t) * ldx + static_cast<size_t>(b) * 32)[k];
                acc[t] = fmaf(w0, xv.x, acc[t]);
                acc[t] = fmaf(w1, xv.y, acc[t]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float a = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = a;
    }
}

int g_force_rows_per_warp = 0;  // experiment hook (0 = heuristic)

template <int NT, int R>
int launch_q4k(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, cudaStream_t s) {
    const unsigned g2 = static_cast<unsigned>((rows + kWarpsPerBlock * R - 1) / (kWarpsPerBlock * R));
    q4k_multi_kernel<NT, R><<<g2, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ4KBlockBytes, dx, ldx, dy, ldy, rows,
                                                               static_cast<int>(cols / kQ4KBlockWeights));
    return static_cast<int>(cudaGetLastError());
}

template <int NT>
int launch_nt(QuantType q, const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, cudaStream_t s) {
    const unsigned grid = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    if (q == QuantType::Q4_K) {
        const int want = g_force_rows_per_warp ? g_force_rows_per_warp : (NT >= 3 && NT <= 4 ? 4 : NT >= 2 ? 2 : 1);
        if (want >= 4 && NT <= 4) return launch_q4k<NT, (NT <= 4 ? 4 : 2)>(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s);
        if (want >= 2) return launch_q4k<NT, 2>(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s);
        return launch_q4k<NT, 1>(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s);
    } else if (q == QuantType::Q5_K) {
        constexpr int R = NT >= 2 ? 2 : 1;
        const unsigned g2 = static_cast<unsigned>((rows + kWarpsPerBlock * R - 1) / (kWarpsPerBlock * R));
        q5k_multi_kernel<NT, R><<<g2, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ5KBlockBytes, dx, ldx, dy, ldy, rows,
                                                                   static_cast<int>(cols / kQ5KBlockWeights));
    } else if (q == QuantType::Q8_0) {
        q80_multi_kernel<NT><<<grid, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ80BlockBytes, dx, ldx, dy, ldy, rows,
                                                                  static_cast<int>(cols / kQ80BlockWeights));
    } else {
        constexpr int R = NT >= 2 ? 2 : 1;
        const unsigned g2 = static_cast<unsigned>((rows + kWarpsPerBlock * R - 1) / (kWarpsPerBlock * R));
        q6k_multi_kernel<NT, R><<<g2, kWarpsPerBlock * 32, 0, s>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ6KBlockBytes, dx, ldx, dy, ldy, rows,
                                                                   static_cast<int>(cols / kQ6KBlockWeights));
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

void set_rows_per_warp_experiment(int r) { g_force_rows_per_warp = r; }

int gemv_quant_multi_cuda(QuantType q, const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows,
                          size_t cols, int nt, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    if (nt <= 0 || rows == 0) return nt == 0 ? 0 : static_cast<int>(cudaErrorInvalidValue);
    if (!dW || !dx || !dy || ldx % 4 != 0 || ldx < cols) return static_cast<int>(cudaErrorInvalidValue);
    while (nt > 0) {
        const int n = nt > 8 ? 8 : nt;
        int e;
        if (n == 1) {  // single vector: the original single-row kernel is the better-tuned one
            e = gemv_quant_cuda(q, dW, ld_blocks, dx, dy, rows, cols, stream);
        } else {  // Q4_K, Q5_K, Q6_K, Q8_0: every format reads each weight once per group of up to 8 vectors
            if (reinterpret_cast<uintptr_t>(dx) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
            switch (n) {
                case 1: e = launch_nt<1>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 2: e = launch_nt<2>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 3: e = launch_nt<3>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 4: e = launch_nt<4>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 5: e = launch_nt<5>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 6: e = launch_nt<6>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                case 7: e = launch_nt<7>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
                default: e = launch_nt<8>(q, dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, s); break;
            }
        }
        if (e != 0) return e;
        dx += static_cast<size_t>(n) * ldx;
        dy += static_cast<size_t>(n) * ldy;
        nt -= n;
    }
    return 0;
}

}  // namespace dara::kernels
