// Prefill attention as two tiled GEMMs and a masked softmax, for FP32 and INT8 (Q8) K/V caches:
//     S = scale * Q_h K^T      (per head h, over the key range a query sub-chunk can see)
//     P = softmax(S) with causal and sliding-window masks
//     O_h = P V                (then * sigmoid(gate) for Qwen's gated attention)
// Why: the split-KV kernel (attn_splitkv.cu) is built for decode: every (token, head, 128-position split) block re-reads its K/V range, so a 512-token Nano
// prefill streamed ~150 GB of K/V through L2 (34% of prefill time) and a 9K-token Qwen prefill took ~90 s. Here K/V tiles are staged in shared memory once per
// 64 query rows. Same register tiling as the weight GEMMs (128 x 64 tiles, 8 x 4 outputs per thread), FP32 accumulation. A Q8 cache is dequantised while its
// tile is loaded (code * scale of its 32-value group), so the INT8 cache is consumed directly. Queries are processed in sub-chunks of kSub tokens so the score
// scratch stays bounded at long contexts. Results differ from the split-KV path only by summation order.
#include "model/attn_prefill.h"
#include "model/layer_cuda.h"
#include "kernels/quant.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>

namespace dara::model {
namespace {

constexpr int BM = 128, BN = 64, BK = 32, TM = 8, TN = 4, kThreads = 256;
constexpr int WS = BM + 4, XS = BN + 4;
constexpr int kSub = 64;   // query tokens per sub-chunk (bounds the score scratch)
constexpr size_t kScratchBudgetFloats = size_t{16} << 20;   // 64 MiB: beyond it (long contexts) a sub-chunk takes fewer query tokens
// Tensor-core attention GEMMs: default on compute capability 8.0+ (Ampere, Ada, Blackwell); DARA_MMA=1 forces them on Turing (testing), DARA_MMA=0 off.
bool attn_use_mma() {
    static const bool on = [] {
        if (const char* e = std::getenv("DARA_MMA_ATTN")) return e[0] == '1';   // attention only (A/B against the GEMMs)
        if (const char* e = std::getenv("DARA_MMA")) return e[0] == '1';
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 8;
    }();
    return on;
}
constexpr int kDrainPositions = 16384;   // key ranges longer than this synchronise after each sub-chunk (Windows TDR, see attn_prefill_cuda)

// Weight element (row r, column k) of a row-major matrix with leading dimension ldw: FP32, or an int8 code times the scale of its 32-value group
// (Q8 cache: element offset e -> scale index e / 32; the base pointers passed in are at an offset that is a multiple of 32).
// WF: 0 = FP32, 1 = int8 code * per-32 scale, 2 = FP16, 3 = 4-bit value * per-32 (scale, min) (kv4.h), 4 = Q4 keys: 4-bit with per-channel group
// (scale, min) before the frontier, the Q8 ring after it (aux.base: element offset of W within the cache; Sc: the cache's group scales, not offset)
template <int WF>
__device__ __forceinline__ float wget(const void* W, const float* Sc, size_t e, const KvAux& aux) {
    if (WF == 1) return static_cast<float>(static_cast<const int8_t*>(W)[e]) * Sc[e >> 5];
    if (WF == 2) return __half2float(static_cast<const __half*>(W)[e]);
    if (WF == 3) {
        const float2 sm = __half22float2(reinterpret_cast<const __half2*>(Sc)[e >> 5]);
        return static_cast<float>((static_cast<const uint8_t*>(W)[e >> 1] >> ((e & 1) * 4)) & 15u) * sm.x + sm.y;
    }
    if (WF == 4) {
        const size_t a = aux.base + e;
        const int r = static_cast<int>(a / static_cast<size_t>(aux.kv_width));
        const size_t c = a - static_cast<size_t>(r) * aux.kv_width;
        if (r < kivi_frontier(aux.held)) {
            const float2 sm = __half22float2(reinterpret_cast<const __half2*>(Sc)[static_cast<size_t>(r / kKiviGroup) * aux.kv_width + c]);
            return static_cast<float>((static_cast<const uint8_t*>(W)[e >> 1] >> ((a & 1) * 4)) & 15u) * sm.x + sm.y;
        }
        const size_t i = static_cast<size_t>(r % kKiviRing) * aux.kv_width + c;
        return static_cast<float>(aux.ring[i]) * aux.ring_sc[i >> 5];
    }
    return static_cast<const float*>(W)[e];
}

// Y[t][n] = sum_k W(n, k) X[t][k], k < Kvalid (W and X zero beyond).  TRANS_W = false: W(n, k) at n*ldw + k  (rows n < N)
//                                                                     TRANS_W = true:  W(n, k) at k*ldw + n
template <bool TRANS_W, int WF, bool MMA>
__global__ void __launch_bounds__(kThreads, 2)
attn_gemm_kernel(const void* __restrict__ W, const float* __restrict__ Sc, size_t ldw, const float* __restrict__ X, size_t ldx, float* __restrict__ Y, size_t ldy,
                 int N, int T, int Kpad, int Kvalid, int rpt, int qoff, KvAux aux) {
    // Causal bound: rows of X are queries (token = row / rpt, at key index qoff + token). A tile never needs keys past its last query's index:
    // the scores GEMM (keys are the output rows) skips tiles entirely above it -- the masked softmax ignores those entries -- and the values GEMM
    // (keys are the reduction) stops there, where the probabilities are zero anyway. Halves the work of a causal prefill chunk.
    const int last_key = qoff + (min(T - 1, static_cast<int>(blockIdx.y) * BN + BN - 1)) / rpt;
    if (!TRANS_W && static_cast<int>(blockIdx.x) * BM > last_key) return;
    if (TRANS_W) { Kvalid = min(Kvalid, last_key + 1); Kpad = min(Kpad, (Kvalid + BK - 1) / BK * BK); }
    __shared__ __align__(16) float Ws[BK][WS];
    __shared__ __align__(16) float Xs[BK][XS];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int lq = tid & 3, xt = tid >> 2;
    const bool xv = tok0 + xt < T;
    const float* xp = X + static_cast<size_t>(xv ? tok0 + xt : 0) * ldx;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < Kpad; k0 += BK) {
        if (TRANS_W) {   // 32 k-rows x 128 n: thread -> k = tid / 8, 16 consecutive n
            const int k = tid >> 3, n0 = (tid & 7) * 16;
            const bool kv = k0 + k < Kvalid;
            const size_t base = static_cast<size_t>(kv ? k0 + k : 0) * ldw + row0 + n0;
            if (WF == 0) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                    if (kv && row0 + n0 + 4 * e + 3 < N) v = *reinterpret_cast<const float4*>(static_cast<const float*>(W) + base + 4 * e);
                    *reinterpret_cast<float4*>(&Ws[k][n0 + 4 * e]) = v;
                }
            } else {
#pragma unroll
                for (int e = 0; e < 16; ++e) Ws[k][n0 + e] = (kv && row0 + n0 + e < N) ? wget<WF>(W, Sc, base + e, aux) : 0.0f;
            }
        } else {         // 128 rows x 32 k: thread -> rows tid/4 and tid/4 + 64, 8 consecutive k
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = (tid >> 2) + 64 * i;
                const bool rv = row0 + r < N;
                const size_t base = static_cast<size_t>(rv ? row0 + r : 0) * ldw + k0 + lq * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) Ws[lq * 8 + e][r] = (rv && k0 + lq * 8 + e < Kvalid) ? wget<WF>(W, Sc, base + e, aux) : 0.0f;
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) Xs[lq * 8 + e][xt] = (xv && k0 + lq * 8 + e < Kvalid) ? xp[k0 + lq * 8 + e] : 0.0f;
        __syncthreads();
        if constexpr (MMA) {
            // Tensor cores (mma.m16n8k8, FP16 inputs from the FP32 tiles, FP32 accumulation): warp w owns rows 32*(w&3)..+31 x tokens 32*(w>>2)..+31.
            const int lane = tid & 31, wid = tid >> 5, gid = lane >> 2, tig = lane & 3;
            const int wr = 32 * (wid & 3), wc = 32 * (wid >> 2);
#pragma unroll
            for (int kk = 0; kk < BK; kk += 8) {
                const int k0r = kk + 2 * tig;
                uint32_t a[2][2], b[4];
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = wr + 16 * mi + gid + 8 * h;
                        const __half2 v = __floats2half2_rn(Ws[k0r][r], Ws[k0r + 1][r]);
                        a[mi][h] = *reinterpret_cast<const uint32_t*>(&v);
                    }
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) {
                    const int t = wc + 8 * ni + gid;
                    const __half2 v = __floats2half2_rn(Xs[k0r][t], Xs[k0r + 1][t]);
                    b[ni] = *reinterpret_cast<const uint32_t*>(&v);
                }
#pragma unroll
                for (int mi = 0; mi < 2; ++mi)
#pragma unroll
                    for (int ni = 0; ni < 4; ++ni) {
                        float* c = &acc[(mi * 16 + ni * 4) / TN][(mi * 16 + ni * 4) % TN];
                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[mi][0]), "r"(a[mi][1]), "r"(b[ni]));
                    }
            }
        } else {
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
        }
        __syncthreads();
    }
    if constexpr (MMA) {   // tile (mi, ni) = acc flat [mi*16 + ni*4 .. +3]: {row gid, tokens 2tig, 2tig+1}, {row gid+8, same tokens}
        const int lane = tid & 31, wid = tid >> 5, gid = lane >> 2, tig = lane & 3;
        const int wr = 32 * (wid & 3), wc = 32 * (wid >> 2);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int r = row0 + wr + 16 * mi + gid + 8 * (e >> 1), t = tok0 + wc + 8 * ni + 2 * tig + (e & 1);
                    const int f = mi * 16 + ni * 4 + e;
                    if (r < N && t < T) Y[static_cast<size_t>(t) * ldy + r] = acc[f / TN][f % TN];
                }
        return;
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = tok0 + tx * 4 + j;
        if (t >= T) continue;
        float* yr = Y + static_cast<size_t>(t) * ldy;
#pragma unroll
        for (int i = 0; i < TM; ++i) { const int r = row0 + ty * 8 + i; if (r < N) yr[r] = acc[i][j]; }
    }
}

// ---- int8 query x int8 key scores (dp4a) ---------------------------------------------------------------------------------------------------------
// The Q8 cache already holds the keys as int8 codes with one scale per 32 values. Quantising the queries the same way (what the weight GEMMs do to their
// activations) turns the scores GEMM -- half of attention's FLOPs -- into dp4a (4 multiply-adds per instruction) with one float combine per 32 values:
//     S[r][j] = sum_g dq[r][g] * dk[j][g] * sum_{d in g} q8[r][d] * k8[j][d]
// LOSSY (8-bit queries) but within the quality gate (bench/quality_gate.py, variant int8_q8kv_qk8: Nano and Core 4B, English / code / JSON / 1500 / 8K tokens, top-1
// 126-128 of 128 and KL no worse than the int8 default). On by default, and only where the tensor-core attention is not used (attn_use_mma() false: GTX 16xx and
// RTX 20xx, whose attention was FP32 GEMMs); DARA_ATTN_QK_INT8=0 turns it off. GPUs of compute capability 8.0+ keep their tensor-core path whatever the setting. Same tile shape as the weight GEMM (128 keys x 64 query rows, 8 x 4 per thread),
// the d dimension advanced 64 values (two scale groups) per step with the next step's loads in flight.
int g_qk8_mode = -1;   // -1: environment (default on), 0: off, 1: on (tests)
bool attn_qk_int8() {
    if (g_qk8_mode >= 0) return g_qk8_mode == 1 && !attn_use_mma();
    static const bool on = [] {
        const char* e = std::getenv("DARA_ATTN_QK_INT8");
        return !(e && e[0] == '0');
    }();
    return on && !attn_use_mma();
}

void* qk_scratch(size_t bytes) {   // quantised queries: a few hundred KB, grown when needed (prefill is never inside a CUDA-graph capture)
    static void* p = nullptr;
    static size_t cap = 0;
    if (bytes > cap) {
        if (p) cudaFree(p);
        p = nullptr;
        cap = 0;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        cap = bytes;
    }
    return p;
}

constexpr int QWSI = BM + 4, QXSI = BN + 4;   // padded int rows (16-byte aligned vector loads)

__global__ void __launch_bounds__(kThreads, 2)
attn_qk_i8_kernel(const int8_t* __restrict__ K, const float* __restrict__ Ksc, size_t ldk, const int8_t* __restrict__ Qq, const float* __restrict__ Qd,
                  int hd, float* __restrict__ S, size_t ldS, int P, int R, int rpt, int qoff) {
    // causal bound as in attn_gemm_kernel: a tile of keys entirely above its last query is masked by the softmax, so it is never computed
    const int last_key = qoff + min(R - 1, static_cast<int>(blockIdx.y) * BN + BN - 1) / rpt;
    if (static_cast<int>(blockIdx.x) * BM > last_key) return;
    __shared__ __align__(16) int Ws[16][QWSI];   // keys of the 64-value step: [d/4][key], d 0..31 = scale group 2s, 32..63 = 2s+1
    __shared__ __align__(16) int Xs[16][QXSI];   // queries:                    [d/4][row]
    __shared__ float Wd[2][BM], Xd[2][BN];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int row0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int lrow = tid >> 1, half = tid & 1;        // key loading: key row and which scale group of the step
    const bool wv = row0 + lrow < P;
    const size_t krow_i = wv ? static_cast<size_t>(row0 + lrow) : 0;
    const int8_t* krow = K + krow_i * ldk;
    const float* ksr = Ksc + krow_i * (ldk / 32);
    const int ltok = tid >> 2, part = tid & 3;        // query loading: row and which 16 of the step's 64 values
    const bool xv = tok0 + ltok < R;
    const size_t ql = xv ? static_cast<size_t>(tok0 + ltok) : 0;
    const int8_t* qrow = Qq + ql * hd;
    const float* qdr = Qd + ql * (hd / 32);

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int steps = hd / 64;
    uint4 w0, w1, xa;
    float wdv, xdv;
    auto load = [&](int st) {
        w0 = *reinterpret_cast<const uint4*>(krow + 64 * st + 32 * half);
        w1 = *reinterpret_cast<const uint4*>(krow + 64 * st + 32 * half + 16);
        wdv = wv ? ksr[2 * st + half] : 0.0f;
        xa = xv ? *reinterpret_cast<const uint4*>(qrow + 64 * st + 16 * part) : make_uint4(0, 0, 0, 0);
        xdv = (part < 2 && xv) ? qdr[2 * st + part] : 0.0f;
    };
    load(0);
    for (int st = 0; st < steps; ++st) {
        Ws[8 * half + 0][lrow] = static_cast<int>(w0.x); Ws[8 * half + 1][lrow] = static_cast<int>(w0.y);
        Ws[8 * half + 2][lrow] = static_cast<int>(w0.z); Ws[8 * half + 3][lrow] = static_cast<int>(w0.w);
        Ws[8 * half + 4][lrow] = static_cast<int>(w1.x); Ws[8 * half + 5][lrow] = static_cast<int>(w1.y);
        Ws[8 * half + 6][lrow] = static_cast<int>(w1.z); Ws[8 * half + 7][lrow] = static_cast<int>(w1.w);
        Wd[half][lrow] = wdv;
        Xs[4 * part + 0][ltok] = static_cast<int>(xa.x); Xs[4 * part + 1][ltok] = static_cast<int>(xa.y);
        Xs[4 * part + 2][ltok] = static_cast<int>(xa.z); Xs[4 * part + 3][ltok] = static_cast<int>(xa.w);
        if (part < 2) Xd[part][ltok] = xdv;
        __syncthreads();
        if (st + 1 < steps) load(st + 1);
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
            float wd[TM], xd[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) wd[i] = Wd[sb][ty * 8 + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) xd[j] = Xd[sb][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(wd[i] * xd[j], static_cast<float>(isum[i][j]), acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
        const int t = tok0 + tx * 4 + j;
        if (t >= R) continue;
        float* yr = S + static_cast<size_t>(t) * ldS;
        const int r0 = row0 + ty * 8;
        if (r0 + 7 < P && (ldS % 4 == 0)) {
            *reinterpret_cast<float4*>(yr + r0) = make_float4(acc[0][j], acc[1][j], acc[2][j], acc[3][j]);
            *reinterpret_cast<float4*>(yr + r0 + 4) = make_float4(acc[4][j], acc[5][j], acc[6][j], acc[7][j]);
        } else {
#pragma unroll
            for (int i = 0; i < TM; ++i) if (r0 + i < P) yr[r0 + i] = acc[i][j];
        }
    }
}

// ---- grouped-query prefill: the G query heads that share a K/V head become G rows per token of ONE GEMM pair -------------------------------------------
// Without it a model with several K/V heads (Qwen: 16 query heads over 4) ran one GEMM pair per QUERY head over only m rows: 4x the K/V tile reads, 4x the launches
// and mostly empty tiles at short contexts. Rows are reordered to (kv head, token, head-in-group) so a K/V head's rows are one contiguous [m * G][hd] block (the layout
// Nano's single K/V head already has), then scattered back. Every output element keeps its own accumulation order, so the result is bitwise identical.
int g_group_mode = -1;   // -1: environment (default on), 0: off, 1: on (tests)
bool attn_group_prefill() {
    if (g_group_mode >= 0) return g_group_mode == 1;
    static const bool on = [] {
        const char* e = std::getenv("DARA_ATTN_GROUP_PREFILL");
        return !(e && e[0] == '0');
    }();
    return on;
}

void* grow_buf(void*& p, size_t& cap, size_t bytes) {
    if (bytes > cap) {
        if (p) cudaFree(p);
        p = nullptr;
        cap = 0;
        if (cudaMalloc(&p, bytes) != cudaSuccess) { cudaGetLastError(); return nullptr; }
        cap = bytes;
    }
    return p;
}

// token-major rows (t, h) <-> grouped rows (h / G, t, h % G); one block per row, hd % 4 == 0
__global__ void reorder_heads_kernel(const float* __restrict__ src, float* __restrict__ dst, int m, int heads, int G, int hd, int to_group) {
    const int rt = blockIdx.x, t = rt / heads, h = rt % heads;
    const int rg = (h / G) * (m * G) + t * G + (h % G);
    const float4* s = reinterpret_cast<const float4*>(src + static_cast<size_t>(to_group ? rt : rg) * hd);
    float4* d = reinterpret_cast<float4*>(dst + static_cast<size_t>(to_group ? rg : rt) * hd);
    for (int i = threadIdx.x; i < hd / 4; i += blockDim.x) d[i] = s[i];
}

// Row blockIdx.x of S = (token t, head) with t = blockIdx.x / rows_per_token, query at absolute position pos0 + t, over keys kstart + j, j < ldS:
// masked, scaled softmax in place; masked and padding entries become 0.
__global__ void masked_softmax_kernel(float* __restrict__ S, size_t ldS, int P, int kstart, int pos0, int window, float scale, int rows_per_token) {
    const int t = blockIdx.x / rows_per_token;
    float* row = S + static_cast<size_t>(blockIdx.x) * ldS;
    const int q = pos0 + t;
    const int lo = window > 0 ? max(0, q - window + 1) : 0;
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < P; j += blockDim.x) {
        const int k = kstart + j;
        if (k >= lo && k <= q) m = fmaxf(m, row[j] * scale);
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < nw ? red[lane] : -3.402823466e38f;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) red[0] = m;
    }
    __syncthreads();
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < static_cast<int>(ldS); j += blockDim.x) {
        const int k = kstart + j;
        const float e = (j < P && k >= lo && k <= q) ? expf(row[j] * scale - m) : 0.0f;
        row[j] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < nw ? red[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) red[0] = sum;
    }
    __syncthreads();
    const float inv = 1.0f / red[0];
    for (int j = threadIdx.x; j < static_cast<int>(ldS); j += blockDim.x) row[j] *= inv;
}

// out[i] *= sigmoid(gate[i])   (Qwen's gated attention output; same formula as the split-KV combine kernel)
__global__ void gate_kernel(float* __restrict__ out, const float* __restrict__ gate, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

size_t round_up(size_t v, size_t m) { return (v + m - 1) / m * m; }

template <int KF, int VF>
void launch_pair(const void* Kh, const float* Ksc, const void* Vh, const float* Vsc, size_t kvw, const float* qrows, size_t ldq, float* S, size_t ldS, int P, int R,
                 int hd, int kstart, int pos0, int window, float scale, int rows_per_token, float* out, size_t ldo, cudaStream_t s, const KvAux& aux,
                 const int8_t* qq = nullptr, const float* qd = nullptr) {
    const dim3 g1(static_cast<unsigned>((P + BM - 1) / BM), static_cast<unsigned>((R + BN - 1) / BN));
    if (qq) attn_qk_i8_kernel<<<g1, kThreads, 0, s>>>(static_cast<const int8_t*>(Kh), Ksc, kvw, qq, qd, hd, S, ldS, P, R, rows_per_token, pos0 - kstart);
    else if (attn_use_mma()) attn_gemm_kernel<false, KF, true><<<g1, kThreads, 0, s>>>(Kh, Ksc, kvw, qrows, ldq, S, ldS, P, R, hd, hd, rows_per_token, pos0 - kstart, aux);
    else attn_gemm_kernel<false, KF, false><<<g1, kThreads, 0, s>>>(Kh, Ksc, kvw, qrows, ldq, S, ldS, P, R, hd, hd, rows_per_token, pos0 - kstart, aux);
    masked_softmax_kernel<<<R, 256, 0, s>>>(S, ldS, P, kstart, pos0, window, scale, rows_per_token);
    const dim3 g2(static_cast<unsigned>((hd + BM - 1) / BM), static_cast<unsigned>((R + BN - 1) / BN));
    if (attn_use_mma()) attn_gemm_kernel<true, VF, true><<<g2, kThreads, 0, s>>>(Vh, Vsc, kvw, S, ldS, out, ldo, hd, R, static_cast<int>(ldS), P, rows_per_token, pos0 - kstart, aux);
    else attn_gemm_kernel<true, VF, false><<<g2, kThreads, 0, s>>>(Vh, Vsc, kvw, S, ldS, out, ldo, hd, R, static_cast<int>(ldS), P, rows_per_token, pos0 - kstart, aux);
}

}  // namespace

void attn_prefill_set_qk_int8(int mode) { g_qk8_mode = mode; }
void attn_prefill_set_group(int mode) { g_group_mode = mode; }

size_t attn_prefill_scratch_floats(int n, int pos0, int window, int heads, int kv_heads) {
    const int kstart = window > 0 ? std::max(0, pos0 - window + 1) : 0;
    const int P = pos0 + n - kstart;   // the widest key range of any sub-chunk
    // score rows per query token: the heads that share a K/V head go through one GEMM pair together (sized for that whatever the grouping switch says)
    const size_t per_token = static_cast<size_t>(heads / kv_heads) * round_up(static_cast<size_t>(P), BK);
    const size_t full = static_cast<size_t>(std::min(n, kSub)) * per_token;
    return std::max(per_token, std::min(full, kScratchBudgetFloats));
}

int attn_prefill_cuda(AttnFmt fmt, const float* q, const float* gate, const void* K, const float* Ksc, const void* V, const float* Vsc, int n, int pos0, int heads,
                      int kv_heads, int hd, int window, float scale, float* scratch, size_t scratch_floats, float* out, void* stream, const KvAux& aux_in) {
    if (!q || !K || !V || !scratch || !out || n <= 0 || pos0 < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || hd % 128 != 0 || window < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (fmt == AttnFmt::TQ4) return static_cast<int>(cudaErrorNotSupported);
    // key / value element formats (wget): 0 FP32, 1 int8 * scale, 2 FP16, 3 4-bit values, 4 Q4 keys
    const int kf = fmt == AttnFmt::F32 ? 0 : fmt == AttnFmt::K16V8 ? 2 : fmt == AttnFmt::Q4 ? 4 : 1;
    const int vf = fmt == AttnFmt::F32 ? 0 : (fmt == AttnFmt::K8V4 || fmt == AttnFmt::Q4) ? 3 : 1;
    if ((kf != 0 && kf != 2 && !Ksc) || (vf != 0 && !Vsc)) return static_cast<int>(cudaErrorInvalidValue);
    if (kf == 4 && (!aux_in.ring || !aux_in.ring_sc)) return static_cast<int>(cudaErrorInvalidValue);
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &cap_status);
    const bool qk8 = attn_qk_int8() && cap_status == cudaStreamCaptureStatusNone;
    KvAux aux = aux_in;
    aux.kv_width = kv_heads * hd;
    aux.held = pos0 + n;   // the whole chunk is appended before attention
    aux.pend = nullptr;
    if (scratch_floats < attn_prefill_scratch_floats(n, pos0, window, heads, kv_heads)) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    const size_t kvw = static_cast<size_t>(kv_heads) * hd;
    const int group = heads / kv_heads;
    // byte offset of element `off` in a cache of each element format; scales: per-32 groups (Q4 keys: the unoffset per-channel table + aux.base)
    auto kbytes = [&](size_t off) { return kf == 0 ? off * 4 : kf == 2 ? off * 2 : kf == 4 ? off / 2 : off; };
    auto vbytes = [&](size_t off) { return vf == 0 ? off * 4 : vf == 3 ? off / 2 : off; };
    auto kscales = [&](size_t off) -> const float* { return kf == 1 ? Ksc + off / 32 : kf == 4 ? Ksc : nullptr; };
    auto vscales = [&](size_t off) -> const float* { return vf != 0 ? Vsc + off / 32 : nullptr; };
    auto pair = [&](size_t koff, const void* Kh, const float* Ks, const void* Vh, const float* Vs, const float* qrows, size_t ldq, int P, int R, int kstart, int p0, int rpt,
                    float* o, size_t ldo, size_t ldS) {
        KvAux a = aux;
        a.base = koff;
        const int8_t* qq = nullptr;
        const float* qd = nullptr;
        if (kf == 1 && qk8 && hd % 256 == 0) {   // int8 codes for the keys: score them against int8 queries with dp4a (opt-in, see attn_qk_int8)
            void* qb = qk_scratch(dara::kernels::q8act_bytes(static_cast<size_t>(hd), R));
            if (qb && dara::kernels::quantize_activations_cuda(qrows, ldq, static_cast<size_t>(hd), R, qb, s) == 0) {
                qq = static_cast<const int8_t*>(qb);
                qd = reinterpret_cast<const float*>(static_cast<const uint8_t*>(qb) + static_cast<size_t>(R) * hd);
            }
        }
        if (kf == 0) launch_pair<0, 0>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s, a);
        else if (kf == 2) launch_pair<2, 1>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s, a);
        else if (kf == 4) launch_pair<4, 3>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s, a);
        else if (vf == 3) launch_pair<1, 3>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s, a, qq, qd);
        else launch_pair<1, 1>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s, a, qq, qd);
    };
    const bool grouped = kv_heads > 1 && group > 1 && attn_group_prefill();
    const size_t rows_per_token = kv_heads == 1 ? static_cast<size_t>(heads) : (grouped ? static_cast<size_t>(group) : 1);
    for (int t0 = 0, m = 0; t0 < n; t0 += m) {
        const int p0 = pos0 + t0;
        const int kstart = window > 0 ? std::max(0, p0 - window + 1) : 0;
        m = std::min(kSub, n - t0);
        const size_t fit = scratch_floats / (rows_per_token * round_up(static_cast<size_t>(p0 + m - kstart), BK));   // long key range: fewer query tokens
        m = std::max(1, std::min(m, static_cast<int>(std::min<size_t>(fit, static_cast<size_t>(kSub)))));
        const int P = p0 + m - kstart;
        const size_t ldS = round_up(static_cast<size_t>(P), BK);
        const float* qs = q + static_cast<size_t>(t0) * heads * hd;
        float* os = out + static_cast<size_t>(t0) * heads * hd;
        if (kv_heads == 1) {   // one K/V head: the m*heads query rows (t, h) are evenly strided (hd apart): all heads in ONE pair of GEMMs
            const size_t off = static_cast<size_t>(kstart) * kvw;
            const void* Kh = static_cast<const uint8_t*>(K) + kbytes(off);
            const void* Vh = static_cast<const uint8_t*>(V) + vbytes(off);
            pair(off, Kh, kscales(off), Vh, vscales(off), qs, static_cast<size_t>(hd), P, m * heads, kstart, p0, heads, os, hd, ldS);
        } else if (grouped) {   // several K/V heads: each K/V head's `group` query heads are rows of one GEMM pair
            static void* gq = nullptr; static size_t gq_cap = 0;
            static void* go = nullptr; static size_t go_cap = 0;
            const size_t bytes = static_cast<size_t>(m) * heads * hd * sizeof(float);
            float* qg = static_cast<float*>(grow_buf(gq, gq_cap, bytes));
            float* og = static_cast<float*>(grow_buf(go, go_cap, bytes));
            if (!qg || !og) return static_cast<int>(cudaErrorMemoryAllocation);
            reorder_heads_kernel<<<static_cast<unsigned>(m) * heads, 128, 0, s>>>(qs, qg, m, heads, group, hd, 1);
            const size_t rows = static_cast<size_t>(m) * group;
            for (int g = 0; g < kv_heads; ++g) {
                const size_t off = static_cast<size_t>(kstart) * kvw + static_cast<size_t>(g) * hd;
                const void* Kh = static_cast<const uint8_t*>(K) + kbytes(off);
                const void* Vh = static_cast<const uint8_t*>(V) + vbytes(off);
                pair(off, Kh, kscales(off), Vh, vscales(off), qg + static_cast<size_t>(g) * rows * hd, static_cast<size_t>(hd), P, static_cast<int>(rows), kstart, p0, group,
                     og + static_cast<size_t>(g) * rows * hd, static_cast<size_t>(hd), ldS);
            }
            reorder_heads_kernel<<<static_cast<unsigned>(m) * heads, 128, 0, s>>>(og, os, m, heads, group, hd, 0);
        } else {
            for (int h = 0; h < heads; ++h) {
                const size_t off = static_cast<size_t>(kstart) * kvw + static_cast<size_t>(h / group) * hd;
                const void* Kh = static_cast<const uint8_t*>(K) + kbytes(off);
                const void* Vh = static_cast<const uint8_t*>(V) + vbytes(off);
                const size_t ldq = static_cast<size_t>(heads) * hd;
                pair(off, Kh, kscales(off), Vh, vscales(off), qs + static_cast<size_t>(h) * hd, ldq, P, m, kstart, p0, 1, os + static_cast<size_t>(h) * hd, ldq, ldS);
            }
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return static_cast<int>(e);
        // Long key ranges: let the GPU drain after each sub-chunk. On Windows (WDDM) a long run of queued work at 100K+ positions trips the
        // driver's timeout detection (TDR: the GPU is reset, the process sees "illegal memory access"). Costs ~4% at 111K, nothing below 16K.
        if (P > kDrainPositions) {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) {
                const cudaError_t se = cudaStreamSynchronize(s);
                if (se != cudaSuccess) return static_cast<int>(se);
            }
        }
    }
    if (gate) {
        const size_t total = static_cast<size_t>(n) * heads * hd;
        gate_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, s>>>(out, gate, total);
    }
    return static_cast<int>(cudaGetLastError());
}

int attn_prefill_f32_cuda(const float* q, const float* K, const float* V, int n, int pos0, int heads, int kv_heads, int hd, int window, float scale, float* scratch,
                          size_t scratch_floats, float* out, void* stream) {
    return attn_prefill_cuda(AttnFmt::F32, q, nullptr, K, nullptr, V, nullptr, n, pos0, heads, kv_heads, hd, window, scale, scratch, scratch_floats, out, stream, {});
}

}  // namespace dara::model
