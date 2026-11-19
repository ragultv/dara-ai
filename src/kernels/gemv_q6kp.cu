// Q6_K decode GEMV on a PLANAR copy of the weights (int8 activations, dp4a), for the output head.
//
// Why: a GGUF Q6_K block is 210 bytes, so blocks are only 2-byte aligned and a GEMV cannot use 16-byte loads; the FP32 kernel reaches ~57% of the
// measured memory bandwidth on the 4B's head (521 MB per token, the largest single matrix) and the int8 kernel on the GGUF layout (funnel-shift loads) is
// slower still. Repacking the matrix at load time into planes -- all ql bytes, all qh bytes, all scales, all block scales, each [row][block] -- keeps the
// same bytes (210 per block) but makes every lane's load a single aligned 16-byte load.
//
// Arithmetic as the GGUF-layout int8 kernel (gemv_q8.cu): lane (block b, half, part) owns 64 weights = 4 groups k of 16, each with one int8 scale inside one
// 32-value activation block:  sum = d * sum_k sc_k * dx_k * (dp4a(q6, a) - 32 * sum16(a)).  NT activation vectors share each weight load (Flash verification).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

#include "kernels/q6k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int kWarps = 8;

// minBlocks 3: at most ~85 registers, 3 blocks (24 warps) per SM keep enough loads in flight (uncapped NT >= 4 used 116-160 registers: 1 block per SM)
template <int NT>
__global__ void __launch_bounds__(kWarps * 32, 3)
q6kp_q8_kernel(const uint8_t* __restrict__ QL, const uint8_t* __restrict__ QH, const int8_t* __restrict__ SC, const unsigned short* __restrict__ D, int nblk,
               size_t rows, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa, int K, float* __restrict__ y, size_t ldy) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const int bi = lane >> 2, sub = lane & 3, half = sub >> 1, part = sub & 1, l0 = 16 * part;
    const uint8_t* ql = QL + row * static_cast<size_t>(nblk) * 128;
    const uint8_t* qh = QH + row * static_cast<size_t>(nblk) * 64;
    const int8_t* sc = SC + row * static_cast<size_t>(nblk) * 16;
    const unsigned short* dd = D + row * static_cast<size_t>(nblk);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int b = bi; b < nblk; b += 8) {
        const uint4 A = __ldg(reinterpret_cast<const uint4*>(ql + static_cast<size_t>(b) * 128 + 64 * half + l0));        // k = 0 (lo), 2 (hi)
        const uint4 B = __ldg(reinterpret_cast<const uint4*>(ql + static_cast<size_t>(b) * 128 + 64 * half + 32 + l0));   // k = 1 (lo), 3 (hi)
        const uint4 Hh = __ldg(reinterpret_cast<const uint4*>(qh + static_cast<size_t>(b) * 64 + 32 * half + l0));        // 2 bits per k
        const uint2 S = __ldg(reinterpret_cast<const uint2*>(sc + static_cast<size_t>(b) * 16 + 8 * half));               // sc[part + 2k]
        const float d = __half2float(__ushort_as_half(__ldg(dd + b)));
        const uint32_t a4[4] = {A.x, A.y, A.z, A.w}, b4[4] = {B.x, B.y, B.z, B.w}, h4[4] = {Hh.x, Hh.y, Hh.z, Hh.w};
        uint32_t q6[4][4];   // [k][i]: four packed 6-bit weights (0..63)
#pragma unroll
        for (int k = 0; k < 4; ++k)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t src = (k & 1) ? b4[i] : a4[i];
                q6[k][i] = (((k & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu) | (((h4[i] >> (2 * k)) & 0x03030303u) << 4);
            }
        const uint32_t sw[2] = {S.x, S.y};
        float scf[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {   // byte (part + 2k) of the 8 scale bytes of this half
            const int byte = part + 2 * k;
            scf[k] = static_cast<float>(static_cast<int8_t>((sw[byte >> 2] >> (8 * (byte & 3))) & 0xffu));
        }
        const int ab = b * 8 + 4 * half;   // activation 32-block of group k is ab + k
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const int8_t* qt = qa + static_cast<size_t>(t) * K;
            const float* dt = da + static_cast<size_t>(t) * (K / 32);
            const int* st = sa + static_cast<size_t>(t) * (K / 16);
            // the 4 groups' activation scales and 16-value sums: 3 vector loads instead of 8 scalar ones (ab is a multiple of 4)
            const float4 dv = *reinterpret_cast<const float4*>(dt + ab);
            const int4 s0 = *reinterpret_cast<const int4*>(st + ab * 2), s1 = *reinterpret_cast<const int4*>(st + ab * 2 + 4);
            const float dk[4] = {dv.x, dv.y, dv.z, dv.w};
            const int sk[4] = {part ? s0.y : s0.x, part ? s0.w : s0.z, part ? s1.y : s1.x, part ? s1.w : s1.z};
            float tmp = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint4 av = *reinterpret_cast<const uint4*>(qt + static_cast<size_t>(ab + k) * 32 + l0);
                int p = __dp4a(static_cast<int>(q6[k][0]), static_cast<int>(av.x), 0);
                p = __dp4a(static_cast<int>(q6[k][1]), static_cast<int>(av.y), p);
                p = __dp4a(static_cast<int>(q6[k][2]), static_cast<int>(av.z), p);
                p = __dp4a(static_cast<int>(q6[k][3]), static_cast<int>(av.w), p);
                tmp = fmaf(dk[k] * scf[k], static_cast<float>(p - 32 * sk[k]), tmp);
            }
            acc[t] = fmaf(d, tmp, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = v;
    }
}

template <int NT>
void launch(const uint8_t* base, size_t rows, int nblk, const int8_t* qa, const float* da, const int* sa, int K, float* y, size_t ldy, cudaStream_t s) {
    const size_t nb = rows * static_cast<size_t>(nblk);
    const uint8_t* QL = base;
    const uint8_t* QH = QL + nb * 128;
    const int8_t* SC = reinterpret_cast<const int8_t*>(QH + nb * 64);
    const unsigned short* D = reinterpret_cast<const unsigned short*>(QH + nb * 64 + nb * 16);
    q6kp_q8_kernel<NT><<<static_cast<unsigned>((rows + kWarps - 1) / kWarps), kWarps * 32, 0, s>>>(QL, QH, SC, D, nblk, rows, qa, da, sa, K, y, ldy);
}

// Single-vector decode with R rows per warp: the warp walks the R rows' (block, quarter) work units as one stream, so a row whose unit count is not a
// multiple of 32 (4B head: 10 blocks = 40 units) leaves no lanes idle (4 rows x 40 = 160 = 5 full warp steps).
template <int R>
__global__ void __launch_bounds__(kWarps * 32)
q6kp_q8_rows_kernel(const uint8_t* __restrict__ QL, const uint8_t* __restrict__ QH, const int8_t* __restrict__ SC, const unsigned short* __restrict__ D, int nblk,
                    size_t rows, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa, float* __restrict__ y) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row0 = (static_cast<size_t>(blockIdx.x) * kWarps + warp) * R;
    if (row0 >= rows) return;
    const int U = nblk * 4, total = R * U;
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    for (int u = lane; u < total; u += 32) {
        const int r = u / U, w = u - r * U;
        const size_t row = row0 + r;
        if (row >= rows) break;
        const int b = w >> 2, sub = w & 3, half = sub >> 1, part = sub & 1, l0 = 16 * part;
        const size_t blk = row * static_cast<size_t>(nblk) + b;
        const uint4 A = __ldg(reinterpret_cast<const uint4*>(QL + blk * 128 + 64 * half + l0));
        const uint4 B = __ldg(reinterpret_cast<const uint4*>(QL + blk * 128 + 64 * half + 32 + l0));
        const uint4 Hh = __ldg(reinterpret_cast<const uint4*>(QH + blk * 64 + 32 * half + l0));
        const uint2 S = __ldg(reinterpret_cast<const uint2*>(SC + blk * 16 + 8 * half));
        const float d = __half2float(__ushort_as_half(__ldg(D + blk)));
        const uint32_t a4[4] = {A.x, A.y, A.z, A.w}, b4[4] = {B.x, B.y, B.z, B.w}, h4[4] = {Hh.x, Hh.y, Hh.z, Hh.w};
        const uint32_t sw[2] = {S.x, S.y};
        const int ab = b * 8 + 4 * half;
        const float4 dv = *reinterpret_cast<const float4*>(da + ab);
        const int4 s0 = *reinterpret_cast<const int4*>(sa + ab * 2), s1 = *reinterpret_cast<const int4*>(sa + ab * 2 + 4);
        const float dk[4] = {dv.x, dv.y, dv.z, dv.w};
        const int sk[4] = {part ? s0.y : s0.x, part ? s0.w : s0.z, part ? s1.y : s1.x, part ? s1.w : s1.z};
        float tmp = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint4 av = *reinterpret_cast<const uint4*>(qa + static_cast<size_t>(ab + k) * 32 + l0);
            const uint32_t a[4] = {av.x, av.y, av.z, av.w};
            int p = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t src = (k & 1) ? b4[i] : a4[i];
                const uint32_t q = (((k & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu) | (((h4[i] >> (2 * k)) & 0x03030303u) << 4);
                p = __dp4a(static_cast<int>(q), static_cast<int>(a[i]), p);
            }
            const int byte = part + 2 * k;
            const float scf = static_cast<float>(static_cast<int8_t>((sw[byte >> 2] >> (8 * (byte & 3))) & 0xffu));
            tmp = fmaf(dk[k] * scf, static_cast<float>(p - 32 * sk[k]), tmp);
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) if (rr == r) acc[rr] = fmaf(d, tmp, acc[rr]);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0 && row0 + r < rows) y[row0 + r] = v;
    }
}

template <int R>
void launch_rows(const uint8_t* base, size_t rows, int nblk, const int8_t* qa, const float* da, const int* sa, float* y, cudaStream_t s) {
    const size_t nb = rows * static_cast<size_t>(nblk);
    const uint8_t* QL = base;
    const uint8_t* QH = QL + nb * 128;
    const int8_t* SC = reinterpret_cast<const int8_t*>(QH + nb * 64);
    const unsigned short* D = reinterpret_cast<const unsigned short*>(QH + nb * 64 + nb * 16);
    const size_t warps = (rows + R - 1) / R;
    q6kp_q8_rows_kernel<R><<<static_cast<unsigned>((warps + kWarps - 1) / kWarps), kWarps * 32, 0, s>>>(QL, QH, SC, D, nblk, rows, qa, da, sa, y);
}

// Several vectors (speculative verification): FP32 activations, each lane dequantises its 64 weights once and multiplies them into every vector. On these
// GPUs the int8 path per vector costs more than the FP32 FMAs once the weight decode is shared (the same finding as the GGUF-layout kernels).
template <int NT>
__global__ void __launch_bounds__(kWarps * 32)
q6kp_f32_kernel(const uint8_t* __restrict__ QL, const uint8_t* __restrict__ QH, const int8_t* __restrict__ SC, const unsigned short* __restrict__ D, int nblk,
                size_t rows, const float* __restrict__ x, size_t ldx, float* __restrict__ y, size_t ldy) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarps + warp;
    if (row >= rows) return;
    const int bi = lane >> 2, sub = lane & 3, half = sub >> 1, part = sub & 1, l0 = 16 * part;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
    for (int b = bi; b < nblk; b += 8) {
        const size_t blk = row * static_cast<size_t>(nblk) + b;
        const uint4 A = __ldg(reinterpret_cast<const uint4*>(QL + blk * 128 + 64 * half + l0));
        const uint4 B = __ldg(reinterpret_cast<const uint4*>(QL + blk * 128 + 64 * half + 32 + l0));
        const uint4 Hh = __ldg(reinterpret_cast<const uint4*>(QH + blk * 64 + 32 * half + l0));
        const uint2 S = __ldg(reinterpret_cast<const uint2*>(SC + blk * 16 + 8 * half));
        const float d = __half2float(__ushort_as_half(__ldg(D + blk)));
        const uint32_t a4[4] = {A.x, A.y, A.z, A.w}, b4[4] = {B.x, B.y, B.z, B.w}, h4[4] = {Hh.x, Hh.y, Hh.z, Hh.w};
        const uint32_t sw[2] = {S.x, S.y};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int byte = part + 2 * k;
            const float ds = d * static_cast<float>(static_cast<int8_t>((sw[byte >> 2] >> (8 * (byte & 3))) & 0xffu));
            float w[16];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t src = (k & 1) ? b4[i] : a4[i];
                const uint32_t q = (((k & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu) | (((h4[i] >> (2 * k)) & 0x03030303u) << 4);
#pragma unroll
                for (int j = 0; j < 4; ++j) w[4 * i + j] = ds * static_cast<float>(static_cast<int>((q >> (8 * j)) & 0xffu) - 32);
            }
            const size_t col = static_cast<size_t>(b) * 256 + 128 * half + 32 * k + l0;
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<size_t>(t) * ldx + col);
                float s = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 v = xp[i];
                    s = fmaf(w[4 * i], v.x, s); s = fmaf(w[4 * i + 1], v.y, s); s = fmaf(w[4 * i + 2], v.z, s); s = fmaf(w[4 * i + 3], v.w, s);
                }
                acc[t] += s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) y[static_cast<size_t>(t) * ldy + row] = v;
    }
}

template <int NT>
void launch_f32(const uint8_t* base, size_t rows, int nblk, const float* x, size_t ldx, float* y, size_t ldy, cudaStream_t s) {
    const size_t nb = rows * static_cast<size_t>(nblk);
    const uint8_t* QL = base;
    const uint8_t* QH = QL + nb * 128;
    const int8_t* SC = reinterpret_cast<const int8_t*>(QH + nb * 64);
    const unsigned short* D = reinterpret_cast<const unsigned short*>(QH + nb * 64 + nb * 16);
    q6kp_f32_kernel<NT><<<static_cast<unsigned>((rows + kWarps - 1) / kWarps), kWarps * 32, 0, s>>>(QL, QH, SC, D, nblk, rows, x, ldx, y, ldy);
}

}  // namespace

void repack_q6k_planar(const uint8_t* src, size_t rows, size_t nblk, uint8_t* dst) {
    const size_t nb = rows * nblk;
    uint8_t* QL = dst;
    uint8_t* QH = QL + nb * 128;
    uint8_t* SC = QH + nb * 64;
    uint8_t* D = SC + nb * 16;
    for (size_t i = 0; i < nb; ++i) {   // GGUF Q6_K block: ql[128] qh[64] scales[16] d(f16)
        const uint8_t* blk = src + i * kQ6KBlockBytes;
        std::memcpy(QL + i * 128, blk, 128);
        std::memcpy(QH + i * 64, blk + 128, 64);
        std::memcpy(SC + i * 16, blk + 192, 16);
        std::memcpy(D + i * 2, blk + 208, 2);
    }
}

int gemv_q6kp_q8_cuda(const void* dW, size_t rows, size_t K, const void* scratch, int T, float* dy, size_t ldy, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !scratch || !dy || K % 256 != 0 || T < 0) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(scratch) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base);
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float));
    const int nblk = static_cast<int>(K / 256), Ki = static_cast<int>(K);
    auto w = static_cast<const uint8_t*>(dW);
    auto st = static_cast<cudaStream_t>(stream);
    if (T == 1) {   // decode: R rows per warp so that every lane has work in every step (R * nblk * 4 a multiple of 32)
        static const int force_r = [] { const char* e = std::getenv("DARA_Q6P_ROWS"); return e ? std::atoi(e) : 0; }();
        int R = 1;
        while (R < 8 && (R * nblk * 4) % 32 != 0) R *= 2;
        if (force_r > 0) R = force_r;
        switch (R) {
            case 1: launch_rows<1>(w, rows, nblk, q, d, s, dy, st); break;
            case 2: launch_rows<2>(w, rows, nblk, q, d, s, dy, st); break;
            case 4: launch_rows<4>(w, rows, nblk, q, d, s, dy, st); break;
            default: launch_rows<8>(w, rows, nblk, q, d, s, dy, st); break;
        }
        return static_cast<int>(cudaGetLastError());
    }
    for (int t0 = 0; t0 < T; t0 += 8) {   // up to 8 vectors per pass over the weights
        const int n = T - t0 < 8 ? T - t0 : 8;
        const int8_t* qt = q + static_cast<size_t>(t0) * K;
        const float* dt = d + static_cast<size_t>(t0) * (K / 32);
        const int* sst = s + static_cast<size_t>(t0) * (K / 16);
        float* yt = dy + static_cast<size_t>(t0) * ldy;
        switch (n) {
            case 1: launch<1>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 2: launch<2>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 3: launch<3>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 4: launch<4>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 5: launch<5>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 6: launch<6>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            case 7: launch<7>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
            default: launch<8>(w, rows, nblk, qt, dt, sst, Ki, yt, ldy, st); break;
        }
    }
    return static_cast<int>(cudaGetLastError());
}


int gemv_q6kp_f32_cuda(const void* dW, size_t rows, size_t K, const float* dx, size_t ldx, int T, float* dy, size_t ldy, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !dx || !dy || K % 256 != 0 || T < 0 || ldx % 4 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    const int nblk = static_cast<int>(K / 256);
    auto w = static_cast<const uint8_t*>(dW);
    auto st = static_cast<cudaStream_t>(stream);
    for (int t0 = 0; t0 < T; t0 += 8) {
        const int n = T - t0 < 8 ? T - t0 : 8;
        const float* xt = dx + static_cast<size_t>(t0) * ldx;
        float* yt = dy + static_cast<size_t>(t0) * ldy;
        switch (n) {
            case 1: launch_f32<1>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 2: launch_f32<2>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 3: launch_f32<3>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 4: launch_f32<4>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 5: launch_f32<5>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 6: launch_f32<6>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            case 7: launch_f32<7>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
            default: launch_f32<8>(w, rows, nblk, xt, ldx, yt, ldy, st); break;
        }
    }
    return static_cast<int>(cudaGetLastError());
}
}  // namespace dara::kernels
