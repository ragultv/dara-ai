// Tiled Q6_K x int8-activation GEMM for prefill (dp4a); the Q6_K member of gemm_q8.cu (same tile: 128 rows x 64 tokens, 8 x 4 outputs per thread).
//
// Q6_K blocks are 210 bytes, so only 2-byte aligned: weights are read with 4-byte-aligned loads and funnel shifts (as in gemv_q8.cu) and keep their GGUF
// layout. One k-step = one 128-weight half of a block: per row ql[64] + qh[32] + 8 int8 scales, unpacked to 128 int8 values q in 0..63. With
// weight = d * sc[k/16] * (q - 32), each 16-weight group contributes  d*sc * (dp4a-sum(q, a) - 32 * sum16(a))  times the activation block scale.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q6k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int BM = 128, BN = 64, TM = 8, TN = 4, kThreads = 256;
constexpr int WSI = BM + 4, XSI = BN + 4;

__device__ __forceinline__ uint4 ld16_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
    const unsigned sh = static_cast<unsigned>(a & 2) * 8;
    const uint32_t w0 = __ldg(w), w1 = __ldg(w + 1), w2 = __ldg(w + 2), w3 = __ldg(w + 3), w4 = __ldg(w + 4);
    return make_uint4(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh), __funnelshift_r(w2, w3, sh), __funnelshift_r(w3, w4, sh));
}

struct Regs6 {
    uint4 A, B, H;      // ql[l0..+15], ql[l0+32..+47], qh[l0..+15] of the half (l0 = 16 * part)
    uint32_t sc;        // 4 int8 scales: sc[8h + 4*part .. +3]
    float d;
    uint4 x0, x1;       // 32 int8 activations of the thread's token
    float dxv;          // scale of the thread's activation 32-block
    int s0, s1;         // that block's two 16-sums
};

__device__ __forceinline__ void load_half6(Regs6& r, const uint8_t* wrow, const int8_t* qa_t, const float* da_t, const int* sa_t, bool xv, int step, int part, int q) {
    const int b = step >> 1, h = step & 1;
    const uint8_t* blk = wrow + static_cast<size_t>(b) * kQ6KBlockBytes;
    r.A = ld16_a2(blk + 64 * h + 16 * part);
    r.B = ld16_a2(blk + 64 * h + 32 + 16 * part);
    r.H = ld16_a2(blk + 128 + 32 * h + 16 * part);
    const unsigned short* scp = reinterpret_cast<const unsigned short*>(blk + 192 + 8 * h + 4 * part);
    r.sc = static_cast<uint32_t>(__ldg(scp)) | (static_cast<uint32_t>(__ldg(scp + 1)) << 16);
    r.d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(blk + 208))));
    const int ab = b * 8 + 4 * h + q;
    if (xv) {
        const int8_t* xp = qa_t + static_cast<size_t>(b) * 256 + 128 * h + 32 * q;
        r.x0 = *reinterpret_cast<const uint4*>(xp);
        r.x1 = *reinterpret_cast<const uint4*>(xp + 16);
        r.dxv = da_t[ab];
        r.s0 = sa_t[2 * ab];
        r.s1 = sa_t[2 * ab + 1];
    } else {
        r.x0 = r.x1 = make_uint4(0, 0, 0, 0);
        r.dxv = 0.0f;
        r.s0 = r.s1 = 0;
    }
}

template <bool MMA>
__global__ void __launch_bounds__(kThreads, 2)
q6k_q8_gemm_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa,
                   float* __restrict__ Y, size_t ldy, int N, int T, int K) {
    __shared__ __align__(16) int Ws[32][WSI];   // the half: [k/4][row]
    __shared__ __align__(16) int Xs[32][XSI];   // [k/4][token]
    __shared__ float Wd[8][BM];                 // d * sc per 16-weight group
    __shared__ float Xd[4][BN];                 // activation scale per 32-block
    __shared__ int Xs16[8][BN];                 // activation int sums per 16

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int lrow = tid >> 1, part = tid & 1;
    const bool wv = row0 + lrow < N;
    const uint8_t* wrow = W + static_cast<size_t>(wv ? row0 + lrow : 0) * row_bytes;
    const int ltok = tid >> 2, q = tid & 3;
    const bool xv = tok0 + ltok < T;
    const size_t t_ld = xv ? static_cast<size_t>(tok0 + ltok) : 0;
    const int8_t* qa_t = qa + t_ld * K;
    const float* da_t = da + t_ld * (K / 32);
    const int* sa_t = sa + t_ld * (K / 16);

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int steps = (K / 256) * 2;
    Regs6 cur;
    load_half6(cur, wrow, qa_t, da_t, sa_t, xv, 0, part, q);
    for (int s = 0; s < steps; ++s) {
        {
            const uint32_t a4[4] = {cur.A.x, cur.A.y, cur.A.z, cur.A.w}, b4[4] = {cur.B.x, cur.B.y, cur.B.z, cur.B.w};
            const uint32_t h4[4] = {cur.H.x, cur.H.y, cur.H.z, cur.H.w};
#pragma unroll
            for (int grp = 0; grp < 4; ++grp)       // k = 32*grp + l: grp 0 = low nibbles of A, 1 = low of B, 2 = high of A, 3 = high of B
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t src = (grp & 1) ? b4[i] : a4[i];
                    const uint32_t nib = ((grp & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu;
                    Ws[8 * grp + 4 * part + i][lrow] = static_cast<int>(nib | (((h4[i] >> (2 * grp)) & 0x03030303u) << 4));
                }
#pragma unroll
            for (int e = 0; e < 4; ++e) Wd[4 * part + e][lrow] = cur.d * static_cast<float>(static_cast<int8_t>((cur.sc >> (8 * e)) & 0xffu));
            const uint32_t xv8[8] = {cur.x0.x, cur.x0.y, cur.x0.z, cur.x0.w, cur.x1.x, cur.x1.y, cur.x1.z, cur.x1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) Xs[8 * q + i][ltok] = static_cast<int>(xv8[i]);
            Xd[q][ltok] = cur.dxv;
            Xs16[2 * q][ltok] = cur.s0;
            Xs16[2 * q + 1][ltok] = cur.s1;
        }
        __syncthreads();
        if (s + 1 < steps) load_half6(cur, wrow, qa_t, da_t, sa_t, xv, s + 1, part, q);
        if constexpr (MMA) {
            // Tensor cores (mma.m8n8k16 s8): warp w owns rows 32*(w&3)..+31 x tokens 32*(w>>2)..+31 as 4 x 4 tiles of 8 x 8; one mma per tile and 16-weight
            // scale group gives exactly the dp4a group sum, and the float combine is the same expression in the same order: same output.
            const int lane = tid & 31, wid = tid >> 5, gid = lane >> 2, tig = lane & 3;
            const int wr = 32 * (wid & 3), wc = 32 * (wid >> 2);
#pragma unroll
            for (int ab = 0; ab < 4; ++ab) {
                float pa[4][8];
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                    for (int j = 0; j < 8; ++j) pa[mi][j] = 0.0f;
#pragma unroll
                for (int g16 = 0; g16 < 2; ++g16) {
                    const int sg = 2 * ab + g16;
                    uint32_t a[4], b[4];
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) a[mi] = static_cast<uint32_t>(Ws[4 * sg + tig][wr + 8 * mi + gid]);
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) b[ni] = static_cast<uint32_t>(Xs[4 * sg + tig][wc + 8 * ni + gid]);
#pragma unroll
                    for (int mi = 0; mi < 4; ++mi) {
                        const float wdv = Wd[sg][wr + 8 * mi + gid];
#pragma unroll
                        for (int ni = 0; ni < 4; ++ni) {
                            int c0 = 0, c1 = 0;
                            asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c0), "+r"(c1) : "r"(a[mi]), "r"(b[ni]));
                            const int t0 = wc + 8 * ni + 2 * tig;
                            pa[mi][2 * ni] = fmaf(wdv, static_cast<float>(c0 - 32 * Xs16[sg][t0]), pa[mi][2 * ni]);
                            pa[mi][2 * ni + 1] = fmaf(wdv, static_cast<float>(c1 - 32 * Xs16[sg][t0 + 1]), pa[mi][2 * ni + 1]);
                        }
                    }
                }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        const float xd = Xd[ab][wc + 8 * ni + 2 * tig + e];
#pragma unroll
                        for (int mi = 0; mi < 4; ++mi) acc[(mi * 8 + 2 * ni + e) / TN][(mi * 8 + 2 * ni + e) % TN] = fmaf(xd, pa[mi][2 * ni + e], acc[(mi * 8 + 2 * ni + e) / TN][(mi * 8 + 2 * ni + e) % TN]);
                    }
            }
        } else {
#pragma unroll
        for (int ab = 0; ab < 4; ++ab) {            // activation 32-block = two 16-weight scale groups
            float pa[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) pa[i][j] = 0.0f;
#pragma unroll
            for (int g16 = 0; g16 < 2; ++g16) {
                const int sg = 2 * ab + g16;
                int isum[TM][TN];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) isum[i][j] = 0;
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int4 a0 = *reinterpret_cast<const int4*>(&Ws[4 * sg + kk][ty * 8]);
                    const int4 a1 = *reinterpret_cast<const int4*>(&Ws[4 * sg + kk][ty * 8 + 4]);
                    const int4 bv = *reinterpret_cast<const int4*>(&Xs[4 * sg + kk][tx * 4]);
                    const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                    const int bb[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                    for (int i = 0; i < TM; ++i)
#pragma unroll
                        for (int j = 0; j < TN; ++j) isum[i][j] = __dp4a(a[i], bb[j], isum[i][j]);
                }
                float wd[TM];
                int xs[TN];
#pragma unroll
                for (int i = 0; i < TM; ++i) wd[i] = Wd[sg][ty * 8 + i];
#pragma unroll
                for (int j = 0; j < TN; ++j) xs[j] = 32 * Xs16[sg][tx * 4 + j];
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) pa[i][j] = fmaf(wd[i], static_cast<float>(isum[i][j] - xs[j]), pa[i][j]);
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const float xd = Xd[ab][tx * 4 + j];
#pragma unroll
                for (int i = 0; i < TM; ++i) acc[i][j] = fmaf(xd, pa[i][j], acc[i][j]);
            }
        }
        }
        __syncthreads();
    }
    if constexpr (MMA) {   // acc[mi][2*ni+e] = row wr+8mi+gid, token wc+8ni+2tig+e
        const int lane = tid & 31, wid = tid >> 5, gid = lane >> 2, tig = lane & 3;
        const int wr = 32 * (wid & 3), wc = 32 * (wid >> 2);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            const int r = row0 + wr + 8 * mi + gid;
            if (r >= N) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int t = tok0 + wc + 8 * ni + 2 * tig + e;
                    if (t < T) Y[static_cast<size_t>(t) * ldy + r] = acc[(mi * 8 + 2 * ni + e) / TN][(mi * 8 + 2 * ni + e) % TN];
                }
        }
        return;
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

bool gemm_q8_use_mma();   // gemm_q8.cu

int gemm_q6k_q8_cuda(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream) {
    if (rows == 0 || T == 0) return 0;
    if (!dW || !scratch || !dy || T < 0 || K == 0 || K % 256 != 0 || ld_blocks < K / 256) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 4 != 0 || reinterpret_cast<uintptr_t>(scratch) % 16 != 0 || reinterpret_cast<uintptr_t>(dy) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base);
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float));
    const dim3 grid(static_cast<unsigned>((rows + BM - 1) / BM), static_cast<unsigned>((T + BN - 1) / BN));
    if (gemm_q8_use_mma())
        q6k_q8_gemm_kernel<true><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ6KBlockBytes, q, d, s, dy, ldy,
                                                                                       static_cast<int>(rows), T, static_cast<int>(K));
    else
        q6k_q8_gemm_kernel<false><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ6KBlockBytes, q, d, s, dy, ldy,
                                                                                        static_cast<int>(rows), T, static_cast<int>(K));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::kernels
