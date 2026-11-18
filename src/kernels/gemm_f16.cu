// FP16 tensor-core GEMM for the vision encoders: Y[t*ldy + n] = sum_k W[n][k] * X[t*ldx + k], W fp16 [N][K] (the mmproj F16 weights, exact), X fp32 rounded
// to fp16 as it is staged, FP32 accumulation. mma.m16n8k8 (Turing and newer: GTX 16xx run it without tensor-core hardware, RTX at tensor-core rate).
//
// Tile: 128 weight rows x 64 tokens per block, k advanced 32 per step; 8 warps, each 32 rows x 32 tokens = 2 x 4 tiles of 16 x 8.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int BM = 128, BN = 64, BK = 32, kThreads = 256;
constexpr int PAD = 8;   // halves of padding per shared row (bank spread; keeps 16-byte alignment)

__global__ void __launch_bounds__(kThreads, 2)
gemm_f16_kernel(const __half* __restrict__ W, size_t ldw, const float* __restrict__ X, size_t ldx, float* __restrict__ Y, size_t ldy, int N, int T, int K) {
    __shared__ __align__(16) __half Ws[BM][BK + PAD];
    __shared__ __align__(16) __half Xs[BN][BK + PAD];
    const int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int wr = 32 * (wid & 3), wc = 32 * (wid >> 2);
    float c[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) c[mi][ni][e] = 0.0f;

    // loaders: weights 128 x 32 halves = 512 chunks of 8 halves (2 per thread); activations 64 x 32 floats = 256 chunks of 8 floats (1 per thread)
    for (int k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int ch = tid + it * kThreads, r = ch >> 2, kc = (ch & 3) * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (row0 + r < N) v = __ldg(reinterpret_cast<const uint4*>(W + static_cast<size_t>(row0 + r) * ldw + k0 + kc));
            *reinterpret_cast<uint4*>(&Ws[r][kc]) = v;
        }
        {
            const int t = tid >> 2, kc = (tid & 3) * 8;
            __half2 h[4];
            if (tok0 + t < T) {
                const float* xp = X + static_cast<size_t>(tok0 + t) * ldx + k0 + kc;
                const float4 a = *reinterpret_cast<const float4*>(xp), b = *reinterpret_cast<const float4*>(xp + 4);
                h[0] = __floats2half2_rn(a.x, a.y); h[1] = __floats2half2_rn(a.z, a.w); h[2] = __floats2half2_rn(b.x, b.y); h[3] = __floats2half2_rn(b.z, b.w);
            } else {
                h[0] = h[1] = h[2] = h[3] = __floats2half2_rn(0.0f, 0.0f);
            }
            *reinterpret_cast<uint4*>(&Xs[t][kc]) = *reinterpret_cast<const uint4*>(h);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            uint32_t a[2][2], b[4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                a[mi][0] = *reinterpret_cast<const uint32_t*>(&Ws[wr + 16 * mi + gid][kk + 2 * tig]);
                a[mi][1] = *reinterpret_cast<const uint32_t*>(&Ws[wr + 16 * mi + gid + 8][kk + 2 * tig]);
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) b[ni] = *reinterpret_cast<const uint32_t*>(&Xs[wc + 8 * ni + gid][kk + 2 * tig]);
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                                 : "+f"(c[mi][ni][0]), "+f"(c[mi][ni][1]), "+f"(c[mi][ni][2]), "+f"(c[mi][ni][3])
                                 : "r"(a[mi][0]), "r"(a[mi][1]), "r"(b[ni]));
        }
        __syncthreads();
    }
    // c[mi][ni]: {row gid, tokens 2tig, 2tig+1}, {row gid+8, same tokens}
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = row0 + wr + 16 * mi + gid + 8 * (e >> 1), t = tok0 + wc + 8 * ni + 2 * tig + (e & 1);
                if (r < N && t < T) Y[static_cast<size_t>(t) * ldy + r] = c[mi][ni][e];
            }
}

}  // namespace

int gemm_f16_cuda(const void* dW, size_t ldw, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !dx || !dy || T < 0 || cols == 0 || cols % BK != 0 || ldw % 8 != 0 || ldx % 4 != 0 || ldw < cols || ldx < cols) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    const dim3 grid(static_cast<unsigned>((rows + BM - 1) / BM), static_cast<unsigned>((T + BN - 1) / BN));
    gemm_f16_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const __half*>(dW), ldw, dx, ldx, dy, ldy, static_cast<int>(rows), T,
                                                                             static_cast<int>(cols));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
