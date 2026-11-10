// Tiled Q4_K x FP32 GEMM for prefill:  Y[t][n] = sum_k W[n][k] * X[t][k]   (W quantised Q4_K, X and Y row-major per token).
//
// Why: the multi-vector GEMV re-reads every weight once per group of 8 tokens and is limited by that traffic and per-token overhead (~1 TFLOPS on a
// GTX 1650). Here a block computes a 128 (output rows) x 64 (tokens) tile: each 32-wide k-slice of the weight tile (exactly one Q4_K sub-block per row)
// is decoded ONCE into shared memory and reused by all 64 tokens. Weights are decoded with the GEMV kernel's exact arithmetic (fmaf(d*sc, q, -dmin*m)),
// so only the order of accumulation differs from the GEMV path (validated to ~1e-6 relative).
//
// Threads: 256 = 16 x 16; thread (tx, ty) owns rows ty*8 .. ty*8+7 and tokens tx*4 .. tx*4+3 (a 8x4 register tile: 32 FMA per 3 vector shared loads).
// The next k-slice is fetched from global memory into registers while the current one is multiplied, and written to shared memory afterwards.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int BM = 128, BN = 64, BK = 32, TM = 8, TN = 4, kThreads = 256;
constexpr int WS = BM + 4, XS = BN + 4;   // padded rows keep 16-byte vector loads aligned

struct Regs {          // one k-slice (sub-block) of the data this thread loads
    uint4 hdr[2];      // block headers of the thread's two weight rows (d, dmin, scales)
    uint2 qw[2];       // 8 quant bytes of each of those rows (low 4 bits)
    uint2 qh[2];       // Q5_K only: the matching 8 bytes of 5th bits
    float4 xa, xb;     // 8 activations of the thread's token
};

template <bool Q5>
__device__ __forceinline__ void load_slice(Regs& r, const uint8_t* const wp[2], const bool wv[2], const float* xp, bool xv, int step, int lq) {
    const int b = step >> 3, j = step & 7;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint8_t* blk = wp[i] + static_cast<size_t>(b) * (Q5 ? kQ5KBlockBytes : kQ4KBlockBytes);
        r.hdr[i] = *reinterpret_cast<const uint4*>(blk);
        r.qw[i] = *reinterpret_cast<const uint2*>(blk + (Q5 ? 48 : 16) + 32 * (j >> 1) + 8 * lq);
        if (Q5) r.qh[i] = *reinterpret_cast<const uint2*>(blk + 16 + 8 * lq);
        (void)wv;
    }
    const float* xs = xp + static_cast<size_t>(b) * 256 + j * 32 + lq * 8;
    if (xv) { r.xa = *reinterpret_cast<const float4*>(xs); r.xb = *reinterpret_cast<const float4*>(xs + 4); }
    else { r.xa = make_float4(0.f, 0.f, 0.f, 0.f); r.xb = r.xa; }
}

template <bool Q5>
__global__ void __launch_bounds__(kThreads, 2)
q4k_gemm_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ X, size_t ldx, float* __restrict__ Y, size_t ldy, int N, int T, int nblk) {
    __shared__ __align__(16) float Ws[BK][WS];   // decoded weights, [k][row]
    __shared__ __align__(16) float Xs[BK][XS];   // activations,     [k][token]
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;

    const int lq = tid & 3;                     // which 8 consecutive k of the 32-wide slice this thread loads
    const int lr[2] = {tid >> 2, (tid >> 2) + 64};   // the two weight rows (of the 128) it loads
    const bool wv[2] = {row0 + lr[0] < N, row0 + lr[1] < N};
    const uint8_t* wp[2] = {W + static_cast<size_t>(wv[0] ? row0 + lr[0] : 0) * row_bytes, W + static_cast<size_t>(wv[1] ? row0 + lr[1] : 0) * row_bytes};
    const int xt = tid >> 2;                    // the token (of the 64) it loads
    const bool xv = tok0 + xt < T;
    const float* xp = X + static_cast<size_t>(xv ? tok0 + xt : 0) * ldx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int steps = nblk * 8;
    Regs cur;
    load_slice<Q5>(cur, wp, wv, xp, xv, 0, lq);
    for (int s = 0; s < steps; ++s) {
        // ---- decode the slice held in registers into shared memory
        const int j = s & 7;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint4 hdr = cur.hdr[i];
            const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
            const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
            const uint32_t sh = 8u * (j & 3);
            const uint32_t a0 = (hdr.y >> sh) & 0xffu, a1 = (hdr.z >> sh) & 0xffu, a2 = (hdr.w >> sh) & 0xffu;
            const bool hi = j >= 4;
            const uint32_t sc = hi ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
            const uint32_t mn = hi ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
            const float d1 = d * static_cast<float>(sc), m1 = dmin * static_cast<float>(mn);
            const int shift = (j & 1) * 4;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const uint32_t word = e < 4 ? cur.qw[i].x : cur.qw[i].y;
                uint32_t qv = (word >> (8 * (e & 3) + shift)) & 0xFu;
                if (Q5) qv |= (((e < 4 ? cur.qh[i].x : cur.qh[i].y) >> (8 * (e & 3) + j)) & 1u) << 4;  // the 5th bit: bit j of qh[l]
                Ws[lq * 8 + e][lr[i]] = wv[i] ? fmaf(d1, static_cast<float>(qv), -m1) : 0.0f;
            }
        }
        Xs[lq * 8 + 0][xt] = cur.xa.x; Xs[lq * 8 + 1][xt] = cur.xa.y; Xs[lq * 8 + 2][xt] = cur.xa.z; Xs[lq * 8 + 3][xt] = cur.xa.w;
        Xs[lq * 8 + 4][xt] = cur.xb.x; Xs[lq * 8 + 5][xt] = cur.xb.y; Xs[lq * 8 + 6][xt] = cur.xb.z; Xs[lq * 8 + 7][xt] = cur.xb.w;
        __syncthreads();
        // ---- fetch the next slice while multiplying this one
        if (s + 1 < steps) load_slice<Q5>(cur, wp, wv, xp, xv, s + 1, lq);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = *reinterpret_cast<const float4*>(&Ws[kk][ty * 8]);
            const float4 a1 = *reinterpret_cast<const float4*>(&Ws[kk][ty * 8 + 4]);
            const float4 bv = *reinterpret_cast<const float4*>(&Xs[kk][tx * 4]);
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float bb[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int jj = 0; jj < TN; ++jj) acc[i][jj] = fmaf(a[i], bb[jj], acc[i][jj]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int jj = 0; jj < TN; ++jj) {
        const int t = tok0 + tx * 4 + jj;
        if (t >= T) continue;
        float* yr = Y + static_cast<size_t>(t) * ldy;
        const int r0 = row0 + ty * 8;
        if (r0 + 7 < N && (ldy % 4 == 0)) {
            *reinterpret_cast<float4*>(yr + r0) = make_float4(acc[0][jj], acc[1][jj], acc[2][jj], acc[3][jj]);
            *reinterpret_cast<float4*>(yr + r0 + 4) = make_float4(acc[4][jj], acc[5][jj], acc[6][jj], acc[7][jj]);
        } else {
#pragma unroll
            for (int i = 0; i < TM; ++i) if (r0 + i < N) yr[r0 + i] = acc[i][jj];
        }
    }
}

}  // namespace

template <bool Q5>
static int launch_gemm(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    constexpr int block_weights = 256;
    const size_t block_bytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    if (rows == 0 || T <= 0) return rows == 0 || T == 0 ? 0 : static_cast<int>(cudaErrorInvalidValue);
    if (!dW || !dx || !dy || cols == 0 || cols % block_weights != 0 || ld_blocks < cols / block_weights || ldx % 4 != 0 || ldx < cols)
        return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0 || reinterpret_cast<uintptr_t>(dy) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    const dim3 grid(static_cast<unsigned>((rows + BM - 1) / BM), static_cast<unsigned>((T + BN - 1) / BN));
    q4k_gemm_kernel<Q5><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * block_bytes, dx, ldx, dy, ldy,
                                                                                  static_cast<int>(rows), T, static_cast<int>(cols / block_weights));
    return static_cast<int>(cudaGetLastError());
}

int gemm_q4k_cuda(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    return launch_gemm<false>(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, T, stream);
}
int gemm_q5k_cuda(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    return launch_gemm<true>(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, T, stream);
}

int gemm_quant_cuda(QuantType t, const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    switch (t) {
        case QuantType::Q4_K: return gemm_q4k_cuda(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, T, stream);
        case QuantType::Q5_K: return gemm_q5k_cuda(dW, ld_blocks, dx, ldx, dy, ldy, rows, cols, T, stream);
        default: return static_cast<int>(cudaErrorNotSupported);
    }
}

bool gemm_available(QuantType t) { return t == QuantType::Q4_K || t == QuantType::Q5_K; }

}  // namespace dara::kernels
