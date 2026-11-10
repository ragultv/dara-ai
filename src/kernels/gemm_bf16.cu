// Tiled BF16 x FP32 GEMM for prefill:  Y[t][n] = sum_k W[n][k] * X[t][k]   (W bf16 row-major with leading dimension ld, X and Y row-major per token).
//
// Why: Nano's per-layer-input matrices (per_layer_model_proj, inp_gate, proj) are BF16. Their only multi-token path was the multi-vector GEMV, which re-reads
// the weights per group of 8 tokens and ran at ~0.2 TFLOPS (Nano prefill profile: the per-layer input block took ~150 ms of 512 tokens, ~10x its FLOP cost).
// Same tile as gemm_q4k.cu: 128 rows x 64 tokens per block, 32-wide k-slices staged in shared memory, 8 x 4 outputs per thread, next slice prefetched into
// registers. The weights are exact (bf16 -> fp32 is a shift); only the order of accumulation differs from the GEMV path (~1e-6 relative).
#include <cuda_runtime.h>

#include <cstdint>

#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int BM = 128, BN = 64, BK = 32, TM = 8, TN = 4, kThreads = 256;
constexpr int WS = BM + 4, XS = BN + 4;

struct RegsB {
    uint4 w[2];        // 8 bf16 weights of each of the thread's two rows
    float4 xa, xb;     // 8 activations of the thread's token
};

__device__ __forceinline__ void load_slice(RegsB& r, const uint16_t* const wp[2], const float* xp, bool xv, int k0, int lq) {
#pragma unroll
    for (int i = 0; i < 2; ++i) r.w[i] = *reinterpret_cast<const uint4*>(wp[i] + k0 + lq * 8);
    if (xv) { r.xa = *reinterpret_cast<const float4*>(xp + k0 + lq * 8); r.xb = *reinterpret_cast<const float4*>(xp + k0 + lq * 8 + 4); }
    else { r.xa = make_float4(0.f, 0.f, 0.f, 0.f); r.xb = r.xa; }
}

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

__global__ void __launch_bounds__(kThreads, 2)
bf16_gemm_kernel(const uint16_t* __restrict__ W, size_t ld, const float* __restrict__ X, size_t ldx, float* __restrict__ Y, size_t ldy, int N, int T, int K) {
    __shared__ __align__(16) float Ws[BK][WS];
    __shared__ __align__(16) float Xs[BK][XS];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int lq = tid & 3;
    const int lr[2] = {tid >> 2, (tid >> 2) + 64};
    const bool wv[2] = {row0 + lr[0] < N, row0 + lr[1] < N};
    const uint16_t* wp[2] = {W + static_cast<size_t>(wv[0] ? row0 + lr[0] : 0) * ld, W + static_cast<size_t>(wv[1] ? row0 + lr[1] : 0) * ld};
    const int xt = tid >> 2;
    const bool xv = tok0 + xt < T;
    const float* xp = X + static_cast<size_t>(xv ? tok0 + xt : 0) * ldx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    RegsB cur;
    load_slice(cur, wp, xp, xv, 0, lq);
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t u[4] = {cur.w[i].x, cur.w[i].y, cur.w[i].z, cur.w[i].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                Ws[lq * 8 + 2 * e][lr[i]] = wv[i] ? bf_lo(u[e]) : 0.0f;
                Ws[lq * 8 + 2 * e + 1][lr[i]] = wv[i] ? bf_hi(u[e]) : 0.0f;
            }
        }
        Xs[lq * 8 + 0][xt] = cur.xa.x; Xs[lq * 8 + 1][xt] = cur.xa.y; Xs[lq * 8 + 2][xt] = cur.xa.z; Xs[lq * 8 + 3][xt] = cur.xa.w;
        Xs[lq * 8 + 4][xt] = cur.xb.x; Xs[lq * 8 + 5][xt] = cur.xb.y; Xs[lq * 8 + 6][xt] = cur.xb.z; Xs[lq * 8 + 7][xt] = cur.xb.w;
        __syncthreads();
        if (k0 + BK < K) load_slice(cur, wp, xp, xv, k0 + BK, lq);
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
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = tok0 + tx * 4 + j;
        if (t >= T) continue;
        float* yr = Y + static_cast<size_t>(t) * ldy;
        const int r0 = row0 + ty * 8;
        if (r0 + 7 < N && (ldy % 4 == 0)) {
            *reinterpret_cast<float4*>(yr + r0) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            *reinterpret_cast<float4*>(yr + r0 + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
        } else {
#pragma unroll
            for (int i = 0; i < TM; ++i) if (r0 + i < N) yr[r0 + i] = acc[i][j];
        }
    }
}

}  // namespace

// BF16 weights [rows][ld] (cols <= ld), FP32 activations [T][ldx], output [T][ldy]. Needs cols % 32 == 0, ld % 8 == 0, ldx % 4 == 0 and 16-byte aligned pointers;
// returns cudaErrorInvalidValue / cudaErrorMisalignedAddress otherwise (callers fall back to the GEMV path).
int gemm_bf16_cuda(const void* dW, size_t ld, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !dx || !dy || T < 0 || cols == 0 || cols % BK != 0 || ld < cols || ld % 8 != 0 || ldx % 4 != 0 || ldx < cols) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0 || reinterpret_cast<uintptr_t>(dy) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    const dim3 grid(static_cast<unsigned>((rows + BM - 1) / BM), static_cast<unsigned>((T + BN - 1) / BN));
    bf16_gemm_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint16_t*>(dW), ld, dx, ldx, dy, ldy, static_cast<int>(rows), T,
                                                                               static_cast<int>(cols));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
