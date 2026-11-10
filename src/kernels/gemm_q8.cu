// Tiled Q4_K / Q5_K x int8-activation GEMM for prefill (dp4a), the prefill counterpart of gemv_q8.cu.
//
// Why: prefill is compute-bound. The FP32 GEMM (gemm_q4k.cu) decodes every weight to float and spends one FMA per weight per token; the GTX 1650 has no
// tensor cores, so FP32 FMA throughput is the ceiling. With the activations quantised to int8 (one scale per 32 values, as in gemv_q8.cu), __dp4a does four
// multiply-adds per instruction on the raw 4/5-bit weight values, and the Q4_K / Q5_K scales factorise per 32-weight sub-block j:
//     Y[t][n] = sum_j dx[t][j] * ( d_n*sc_nj * sum_i q_ni a_ti  -  dmin_n*m_nj * sum_i a_ti )
// The same factorisation llama.cpp's mmq kernels use. LOSSY (8-bit activations): opt-in through DARA_INT8=1 with the decode path.
//
// Tile: 128 output rows x 64 tokens per block, k advanced one Q4_K quant group (64 weights = sub-blocks 2g and 2g+1, which share 32 bytes of nibbles) per
// step. Shared memory holds the group as int8 (4 values per int): weights [16][row], activations [16][token]. Thread (tx, ty) owns rows ty*8..+7 and tokens
// tx*4..+3: per sub-block 8 x 4 int dot products (8 dp4a each), then one float combine per output. The next group is fetched into registers meanwhile.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int BM = 128, BN = 64, TM = 8, TN = 4, kThreads = 256;
constexpr int WSI = BM + 4, XSI = BN + 4;   // padded int rows (16-byte aligned vector loads)

struct Regs {
    uint4 hdr;     // block header of the thread's weight row (d, dmin, scales)
    uint4 qw;      // 16 quant bytes of the row (positions 16*half .. +15 of the group): low nibbles -> sub-block 2g, high -> 2g+1
    uint4 qh;      // Q5_K: the 5th bits of the same positions
    uint4 xa;      // 16 int8 activations of the thread's token
    float dxv;     // part 0/1: activation scale of sub-block 2g / 2g+1
    int sv;        // part 2/3: int sum of the activations of sub-block 2g / 2g+1
};

template <bool Q5>
__device__ __forceinline__ void load_group(Regs& r, const uint8_t* wrow, const int8_t* qa_t, const float* da_t, const int* sa_t, bool xv, int step, int half, int part) {
    const int b = step >> 2, g = step & 3;
    const uint8_t* blk = wrow + static_cast<size_t>(b) * (Q5 ? kQ5KBlockBytes : kQ4KBlockBytes);
    r.hdr = *reinterpret_cast<const uint4*>(blk);
    r.qw = *reinterpret_cast<const uint4*>(blk + (Q5 ? 48 : 16) + 32 * g + 16 * half);
    if (Q5) r.qh = *reinterpret_cast<const uint4*>(blk + 16 + 16 * half);
    const int sb = b * 8 + 2 * g;   // the group's first 32-value activation block
    if (xv) {
        r.xa = *reinterpret_cast<const uint4*>(qa_t + static_cast<size_t>(b) * 256 + 64 * g + 16 * part);
        r.dxv = part < 2 ? da_t[sb + part] : 0.0f;
        r.sv = part >= 2 ? sa_t[(sb + part - 2) * 2] + sa_t[(sb + part - 2) * 2 + 1] : 0;
    } else {
        r.xa = make_uint4(0, 0, 0, 0);
        r.dxv = 0.0f;
        r.sv = 0;
    }
}

template <bool Q5>
__global__ void __launch_bounds__(kThreads, 2)
q_q8_gemm_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa,
                 float* __restrict__ Y, size_t ldy, int N, int T, int K) {
    __shared__ __align__(16) int Ws[16][WSI];   // weights of the group: [k/4][row], k 0..31 = sub-block 2g, 32..63 = 2g+1
    __shared__ __align__(16) int Xs[16][XSI];   // activations:          [k/4][token]
    __shared__ float Wd[2][BM], Wm[2][BM];      // d*sc and dmin*m of each row, per sub-block
    __shared__ float Xd[2][BN];                 // activation scale per token, per sub-block
    __shared__ int Xsum[2][BN];                 // activation int sum per token, per sub-block

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int nblk = K / 256;

    const int lrow = tid >> 1, half = tid & 1;     // weight loading: row and which 16 of the group's 32 quant bytes
    const bool wv = row0 + lrow < N;
    const uint8_t* wrow = W + static_cast<size_t>(wv ? row0 + lrow : 0) * row_bytes;
    const int ltok = tid >> 2, part = tid & 3;     // activation loading: token and which 16 of the group's 64 values
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

    const int steps = nblk * 4;
    Regs cur;
    load_group<Q5>(cur, wrow, qa_t, da_t, sa_t, xv, 0, half, part);
    for (int s = 0; s < steps; ++s) {
        const int g = s & 3;
        // ---- unpack the group held in registers into shared memory
        {
            const uint32_t w[4] = {cur.qw.x, cur.qw.y, cur.qw.z, cur.qw.w};
            const uint32_t h[4] = {cur.qh.x, cur.qh.y, cur.qh.z, cur.qh.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t lo = w[i] & 0x0F0F0F0Fu, hi = (w[i] >> 4) & 0x0F0F0F0Fu;
                if (Q5) {
                    lo |= ((h[i] >> (2 * g)) & 0x01010101u) << 4;
                    hi |= ((h[i] >> (2 * g + 1)) & 0x01010101u) << 4;
                }
                Ws[4 * half + i][lrow] = static_cast<int>(lo);
                Ws[8 + 4 * half + i][lrow] = static_cast<int>(hi);
            }
            if (half == 0) {
                const uint4 hdr = cur.hdr;
                const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
                const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
                const int sh0 = 8 * ((2 * g) & 3), sh1 = 8 * ((2 * g + 1) & 3);
                const bool hp = g >= 2;
                const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
                const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
                const uint32_t s0 = hp ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
                const uint32_t m0 = hp ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
                const uint32_t s1 = hp ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
                const uint32_t m1 = hp ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);
                Wd[0][lrow] = d * static_cast<float>(s0); Wm[0][lrow] = dmin * static_cast<float>(m0);
                Wd[1][lrow] = d * static_cast<float>(s1); Wm[1][lrow] = dmin * static_cast<float>(m1);
            }
            Xs[4 * part + 0][ltok] = static_cast<int>(cur.xa.x);
            Xs[4 * part + 1][ltok] = static_cast<int>(cur.xa.y);
            Xs[4 * part + 2][ltok] = static_cast<int>(cur.xa.z);
            Xs[4 * part + 3][ltok] = static_cast<int>(cur.xa.w);
            if (part < 2) Xd[part][ltok] = cur.dxv;
            else Xsum[part - 2][ltok] = cur.sv;
        }
        __syncthreads();
        if (s + 1 < steps) load_group<Q5>(cur, wrow, qa_t, da_t, sa_t, xv, s + 1, half, part);
#pragma unroll
        for (int sb = 0; sb < 2; ++sb) {
            int isum[TM][TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) isum[i][j] = 0;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int4 a0 = *reinterpret_cast<const int4*>(&Ws[8 * sb + kk][ty * 8]);
                const int4 a1 = *reinterpret_cast<const int4*>(&Ws[8 * sb + kk][ty * 8 + 4]);
                const int4 bv = *reinterpret_cast<const int4*>(&Xs[8 * sb + kk][tx * 4]);
                const int a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const int bb[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
                for (int i = 0; i < TM; ++i)
#pragma unroll
                    for (int j = 0; j < TN; ++j) isum[i][j] = __dp4a(a[i], bb[j], isum[i][j]);
            }
            float wd[TM], wm[TM], xd[TN], xs[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) { wd[i] = Wd[sb][ty * 8 + i]; wm[i] = Wm[sb][ty * 8 + i]; }
#pragma unroll
            for (int j = 0; j < TN; ++j) { xd[j] = Xd[sb][tx * 4 + j]; xs[j] = static_cast<float>(Xsum[sb][tx * 4 + j]); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(xd[j], fmaf(wd[i], static_cast<float>(isum[i][j]), -wm[i] * xs[j]), acc[i][j]);
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

template <bool Q5>
int launch(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream) {
    const size_t block_bytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    if (rows == 0 || T == 0) return 0;
    if (!dW || !scratch || !dy || T < 0 || K == 0 || K % 256 != 0 || ld_blocks < K / 256) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(scratch) % 16 != 0 || reinterpret_cast<uintptr_t>(dy) % 16 != 0)
        return static_cast<int>(cudaErrorMisalignedAddress);
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base);
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float));
    const dim3 grid(static_cast<unsigned>((rows + BM - 1) / BM), static_cast<unsigned>((T + BN - 1) / BN));
    q_q8_gemm_kernel<Q5><<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * block_bytes, q, d, s, dy, ldy,
                                                                                    static_cast<int>(rows), T, static_cast<int>(K));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

int gemm_q6k_q8_cuda(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream);  // gemm_q8_q6k.cu

bool gemm_q8_available(QuantType t) { return t == QuantType::Q4_K || t == QuantType::Q5_K || t == QuantType::Q6_K; }

// Y[t*ldy + r] = W[r] . x_t for the T int8 vectors in `scratch` (filled by quantize_activations_cuda(dx, ldx, K, T, scratch)).
int gemm_q8_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream) {
    switch (t) {
        case QuantType::Q4_K: return launch<false>(dW, ld_blocks, scratch, K, T, dy, ldy, rows, stream);
        case QuantType::Q5_K: return launch<true>(dW, ld_blocks, scratch, K, T, dy, ldy, rows, stream);
        case QuantType::Q6_K: return gemm_q6k_q8_cuda(dW, ld_blocks, scratch, K, T, dy, ldy, rows, stream);
        default: return static_cast<int>(cudaErrorNotSupported);
    }
}

}  // namespace dara::kernels
