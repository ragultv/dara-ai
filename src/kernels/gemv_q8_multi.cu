// int8-activation (dp4a) GEMV for a FEW vectors at once (2..16): the "one weight read, several rows checked" kernel for speculative verification and
// short multi-token steps. Same lane mapping and arithmetic as the single-vector kernels in gemv_q8.cu; each lane loads and unpacks its 16 weight bytes
// ONCE per block and multiplies them with up to kMaxVec activation vectors, keeping one accumulator per vector. Decode is memory-bound, so checking 8
// tokens costs far less than 8 single-token steps (the weights are read once).
//
// For vector t of the group, the result per row equals the single-vector kernel's on that vector up to the order of the final warp reduction (identical
// per-lane arithmetic), so outputs agree with gemv_q8_cuda to float rounding.
#include <algorithm>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/q6k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kMaxVec = 8;   // vectors per launch (larger groups are split)
constexpr float k2p23 = 8388608.0f;
__device__ __forceinline__ float small2f(uint32_t v) { return __uint_as_float(0x4B000000u | v) - k2p23; }

struct ActView {   // activation layout of quantize_activations_cuda: q[T][K] | d[T][K/32] | s[T][K/16]
    const int8_t* q;
    const float* d;
    const int* s;
};

template <bool Q5, int NV>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q45_q8_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, ActView a, size_t K, float* __restrict__ y, size_t ldy, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    constexpr int kBlockBytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    constexpr int kQsOff = Q5 ? 48 : 16;
    const uint8_t* rp = W + row * row_bytes;
    const int bi = lane >> 3, q = lane & 7, g = q >> 1, half = q & 1;
    const int sh0 = 8 * ((2 * g) & 3), sh1 = 8 * ((2 * g + 1) & 3);
    const bool hi_pair = g >= 2;
    float acc[NV];
#pragma unroll
    for (int t = 0; t < NV; ++t) acc[t] = 0.0f;
    for (int b = bi; b < nblk; b += 4) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kBlockBytes;
        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
        const uint4 qs = __ldg(reinterpret_cast<const uint4*>(blk + kQsOff + 16 * q));
        uint4 qh = make_uint4(0, 0, 0, 0);
        if (Q5) qh = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * half));
        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
        const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
        const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
        const float ds0 = d * small2f(hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u));
        const float dm0 = dmin * small2f(hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u));
        const float ds1 = d * small2f(hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u));
        const float dm1 = dmin * small2f(hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u));
        uint32_t lo[4], hi[4];
        {
            const uint32_t w[4] = {qs.x, qs.y, qs.z, qs.w};
            const uint32_t h4[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[i] = w[i] & 0x0F0F0F0Fu;
                hi[i] = (w[i] >> 4) & 0x0F0F0F0Fu;
                if (Q5) {
                    lo[i] |= ((h4[i] >> (2 * g)) & 0x01010101u) << 4;
                    hi[i] |= ((h4[i] >> (2 * g + 1)) & 0x01010101u) << 4;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NV; ++t) {
            const int8_t* ap = a.q + t * K + static_cast<size_t>(b) * 256 + 64 * g + 16 * half;
            const uint4 alo = *reinterpret_cast<const uint4*>(ap), ahi = *reinterpret_cast<const uint4*>(ap + 32);
            const float* dt = a.d + t * (K / 32);
            const int* st = a.s + t * (K / 16);
            const uint32_t al[4] = {alo.x, alo.y, alo.z, alo.w}, ah[4] = {ahi.x, ahi.y, ahi.z, ahi.w};
            int plo = 0, phi = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                plo = __dp4a(static_cast<int>(lo[i]), static_cast<int>(al[i]), plo);
                phi = __dp4a(static_cast<int>(hi[i]), static_cast<int>(ah[i]), phi);
            }
            acc[t] = fmaf(dt[b * 8 + 2 * g], fmaf(ds0, static_cast<float>(plo), -dm0 * static_cast<float>(st[b * 16 + 4 * g + half])), acc[t]);
            acc[t] = fmaf(dt[b * 8 + 2 * g + 1], fmaf(ds1, static_cast<float>(phi), -dm1 * static_cast<float>(st[b * 16 + 4 * g + 2 + half])), acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NV; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = v;
    }
}

// Multi-ROW variant (the faster one): a warp handles R consecutive rows, so each activation load (and its scale / sum loads) is reused for R rows instead
// of one. Per 256-weight block and lane: R weight chunks are loaded and unpacked once, then for each of the NV vectors 2 x 16-byte activation loads feed
// R x 8 dp4a. Same per-row arithmetic as the single-row kernel.
template <bool Q5, int NV, int R>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q45_q8_multi_rows_kernel(const uint8_t* __restrict__ W, size_t row_bytes, ActView a, size_t K, float* __restrict__ y, size_t ldy, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row0 = (static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp) * R;
    if (row0 >= rows) return;
    constexpr int kBlockBytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    constexpr int kQsOff = Q5 ? 48 : 16;
    const int bi = lane >> 3, q = lane & 7, g = q >> 1, half = q & 1;
    const int sh0 = 8 * ((2 * g) & 3), sh1 = 8 * ((2 * g + 1) & 3);
    const bool hi_pair = g >= 2;
    float acc[R][NV];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NV; ++t) acc[r][t] = 0.0f;
    for (int b = bi; b < nblk; b += 4) {
        uint32_t lo[R][4], hi[R][4];
        float ds0[R], dm0[R], ds1[R], dm1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const bool rv = row0 + r < rows;
            const uint8_t* blk = W + (rv ? row0 + r : row0) * row_bytes + static_cast<size_t>(b) * kBlockBytes;
            const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
            const uint4 qs = __ldg(reinterpret_cast<const uint4*>(blk + kQsOff + 16 * q));
            uint4 qh = make_uint4(0, 0, 0, 0);
            if (Q5) qh = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * half));
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
            const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
            const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
            const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
            ds0[r] = d * small2f(hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u));
            dm0[r] = dmin * small2f(hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u));
            ds1[r] = d * small2f(hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u));
            dm1[r] = dmin * small2f(hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u));
            const uint32_t w[4] = {qs.x, qs.y, qs.z, qs.w};
            const uint32_t h4[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[r][i] = w[i] & 0x0F0F0F0Fu;
                hi[r][i] = (w[i] >> 4) & 0x0F0F0F0Fu;
                if (Q5) {
                    lo[r][i] |= ((h4[i] >> (2 * g)) & 0x01010101u) << 4;
                    hi[r][i] |= ((h4[i] >> (2 * g + 1)) & 0x01010101u) << 4;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < NV; ++t) {
            const int8_t* ap = a.q + t * K + static_cast<size_t>(b) * 256 + 64 * g + 16 * half;
            const uint4 alo = *reinterpret_cast<const uint4*>(ap), ahi = *reinterpret_cast<const uint4*>(ap + 32);
            const float dlo = a.d[t * (K / 32) + b * 8 + 2 * g], dhi = a.d[t * (K / 32) + b * 8 + 2 * g + 1];
            const float slo = static_cast<float>(a.s[t * (K / 16) + b * 16 + 4 * g + half]), shi = static_cast<float>(a.s[t * (K / 16) + b * 16 + 4 * g + 2 + half]);
            const uint32_t al[4] = {alo.x, alo.y, alo.z, alo.w}, ah[4] = {ahi.x, ahi.y, ahi.z, ahi.w};
#pragma unroll
            for (int r = 0; r < R; ++r) {
                int plo = 0, phi = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    plo = __dp4a(static_cast<int>(lo[r][i]), static_cast<int>(al[i]), plo);
                    phi = __dp4a(static_cast<int>(hi[r][i]), static_cast<int>(ah[i]), phi);
                }
                acc[r][t] = fmaf(dlo, fmaf(ds0[r], static_cast<float>(plo), -dm0[r] * slo), acc[r][t]);
                acc[r][t] = fmaf(dhi, fmaf(ds1[r], static_cast<float>(phi), -dm1[r] * shi), acc[r][t]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < NV; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
            if (lane == 0 && row0 + r < rows) y[static_cast<size_t>(t) * ldy + row0 + r] = v;
        }
}

__device__ __forceinline__ uint4 ld16_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
    const unsigned sh = static_cast<unsigned>(a & 2) * 8;
    const uint32_t w0 = __ldg(w), w1 = __ldg(w + 1), w2 = __ldg(w + 2), w3 = __ldg(w + 3), w4 = __ldg(w + 4);
    return make_uint4(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh), __funnelshift_r(w2, w3, sh), __funnelshift_r(w3, w4, sh));
}

template <int NV>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q6_q8_multi_kernel(const uint8_t* __restrict__ W, size_t row_bytes, ActView a, size_t K, float* __restrict__ y, size_t ldy, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint8_t* rp = W + row * row_bytes;
    const int bi = lane >> 2, sub = lane & 3, half = sub >> 1, part = sub & 1, l0 = 16 * part;
    float acc[NV];
#pragma unroll
    for (int t = 0; t < NV; ++t) acc[t] = 0.0f;
    for (int b = bi; b < nblk; b += 8) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kQ6KBlockBytes;
        const uint4 A = ld16_a2(blk + 64 * half + l0);
        const uint4 B = ld16_a2(blk + 64 * half + 32 + l0);
        const uint4 Hh = ld16_a2(blk + 128 + 32 * half + l0);
        const int8_t* scp = reinterpret_cast<const int8_t*>(blk + 192 + 8 * half + part);
        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(blk + 208))));
        const uint32_t a4[4] = {A.x, A.y, A.z, A.w}, b4[4] = {B.x, B.y, B.z, B.w}, h4[4] = {Hh.x, Hh.y, Hh.z, Hh.w};
        uint32_t qv[4][4];
        float sck[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            sck[k] = d * static_cast<float>(__ldg(scp + 2 * k));
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t src = (k & 1) ? b4[i] : a4[i];
                qv[k][i] = (((k & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu) | (((h4[i] >> (2 * k)) & 0x03030303u) << 4);
            }
        }
        const int ab = b * 8 + 4 * half;
#pragma unroll
        for (int t = 0; t < NV; ++t) {
            const int8_t* qa = a.q + t * K;
            const float* dt = a.d + t * (K / 32);
            const int* st = a.s + t * (K / 16);
            float tmp = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint4 av = *reinterpret_cast<const uint4*>(qa + static_cast<size_t>(ab + k) * 32 + l0);
                const uint32_t ai[4] = {av.x, av.y, av.z, av.w};
                int p = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) p = __dp4a(static_cast<int>(qv[k][i]), static_cast<int>(ai[i]), p);
                tmp = fmaf(dt[ab + k] * sck[k], static_cast<float>(p - 32 * st[(ab + k) * 2 + part]), tmp);
            }
            acc[t] += tmp;
        }
    }
#pragma unroll
    for (int t = 0; t < NV; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = v;
    }
}

template <int NV>
int launch_nv(QuantType qt, const uint8_t* W, size_t ld_blocks, ActView a, size_t K, float* y, size_t ldy, size_t rows, cudaStream_t s) {
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const int nblk = static_cast<int>(K / 256);
    constexpr int R = 4;   // rows per warp: activation loads reused R times (measured better than 2 at every NV >= 2)
    const unsigned blocks_r = static_cast<unsigned>((rows + kWarpsPerBlock * R - 1) / (kWarpsPerBlock * R));
    static const bool one_row = [] { const char* e = std::getenv("DARA_MULTI_ONE_ROW"); return e && e[0] == '1'; }();   // A/B: the 1-row kernel
    switch (qt) {
        case QuantType::Q4_K:
            if (one_row) q45_q8_multi_kernel<false, NV><<<blocks, kWarpsPerBlock * 32, 0, s>>>(W, ld_blocks * kQ4KBlockBytes, a, K, y, ldy, rows, nblk);
            else q45_q8_multi_rows_kernel<false, NV, R><<<blocks_r, kWarpsPerBlock * 32, 0, s>>>(W, ld_blocks * kQ4KBlockBytes, a, K, y, ldy, rows, nblk);
            break;
        case QuantType::Q5_K:
            if (one_row) q45_q8_multi_kernel<true, NV><<<blocks, kWarpsPerBlock * 32, 0, s>>>(W, ld_blocks * kQ5KBlockBytes, a, K, y, ldy, rows, nblk);
            else q45_q8_multi_rows_kernel<true, NV, R><<<blocks_r, kWarpsPerBlock * 32, 0, s>>>(W, ld_blocks * kQ5KBlockBytes, a, K, y, ldy, rows, nblk);
            break;
        case QuantType::Q6_K: q6_q8_multi_kernel<NV><<<blocks, kWarpsPerBlock * 32, 0, s>>>(W, ld_blocks * kQ6KBlockBytes, a, K, y, ldy, rows, nblk); break;
        default: return static_cast<int>(cudaErrorNotSupported);
    }
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

// y[t*ldy + r] = W[r] . x_t for the T int8 vectors in `scratch` (quantize_activations_cuda with T vectors). Q4_K / Q5_K / Q6_K, any T >= 1 (groups of 8).
int gemv_q8_multi_cuda(QuantType qt, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !scratch || !dy || T < 0 || K == 0 || K % 256 != 0 || ld_blocks < K / 256) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % (qt == QuantType::Q6_K ? 4 : 16) != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base);
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float));
    auto st = static_cast<cudaStream_t>(stream);
    const auto* W = static_cast<const uint8_t*>(dW);
    for (int t0 = 0; t0 < T; t0 += kMaxVec) {
        const int n = std::min(kMaxVec, T - t0);
        const ActView a{q + static_cast<size_t>(t0) * K, d + static_cast<size_t>(t0) * (K / 32), s + static_cast<size_t>(t0) * (K / 16)};
        float* y = dy + static_cast<size_t>(t0) * ldy;
        int e;
        switch (n) {
            case 1: e = launch_nv<1>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 2: e = launch_nv<2>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 3: e = launch_nv<3>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 4: e = launch_nv<4>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 5: e = launch_nv<5>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 6: e = launch_nv<6>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            case 7: e = launch_nv<7>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
            default: e = launch_nv<8>(qt, W, ld_blocks, a, K, y, ldy, rows, st); break;
        }
        if (e) return e;
    }
    return 0;
}

}  // namespace dara::kernels
