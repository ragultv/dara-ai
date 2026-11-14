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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace dara::model {
namespace {

constexpr int BM = 128, BN = 64, BK = 32, TM = 8, TN = 4, kThreads = 256;
constexpr int WS = BM + 4, XS = BN + 4;
constexpr int kSub = 64;   // query tokens per sub-chunk (bounds the score scratch)

// Weight element (row r, column k) of a row-major matrix with leading dimension ldw: FP32, or an int8 code times the scale of its 32-value group
// (Q8 cache: element offset e -> scale index e / 32; the base pointers passed in are at an offset that is a multiple of 32).
// WF: 0 = FP32, 1 = int8 code * per-32 scale, 2 = FP16
template <int WF>
__device__ __forceinline__ float wget(const void* W, const float* Sc, size_t e) {
    if (WF == 1) return static_cast<float>(static_cast<const int8_t*>(W)[e]) * Sc[e >> 5];
    if (WF == 2) return __half2float(static_cast<const __half*>(W)[e]);
    return static_cast<const float*>(W)[e];
}

// Y[t][n] = sum_k W(n, k) X[t][k], k < Kvalid (W and X zero beyond).  TRANS_W = false: W(n, k) at n*ldw + k  (rows n < N)
//                                                                     TRANS_W = true:  W(n, k) at k*ldw + n
template <bool TRANS_W, int WF>
__global__ void __launch_bounds__(kThreads, 2)
attn_gemm_kernel(const void* __restrict__ W, const float* __restrict__ Sc, size_t ldw, const float* __restrict__ X, size_t ldx, float* __restrict__ Y, size_t ldy,
                 int N, int T, int Kpad, int Kvalid, int rpt, int qoff) {
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
                for (int e = 0; e < 16; ++e) Ws[k][n0 + e] = (kv && row0 + n0 + e < N) ? wget<WF>(W, Sc, base + e) : 0.0f;
            }
        } else {         // 128 rows x 32 k: thread -> rows tid/4 and tid/4 + 64, 8 consecutive k
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = (tid >> 2) + 64 * i;
                const bool rv = row0 + r < N;
                const size_t base = static_cast<size_t>(rv ? row0 + r : 0) * ldw + k0 + lq * 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) Ws[lq * 8 + e][r] = (rv && k0 + lq * 8 + e < Kvalid) ? wget<WF>(W, Sc, base + e) : 0.0f;
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) Xs[lq * 8 + e][xt] = (xv && k0 + lq * 8 + e < Kvalid) ? xp[k0 + lq * 8 + e] : 0.0f;
        __syncthreads();
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
#pragma unroll
        for (int i = 0; i < TM; ++i) { const int r = row0 + ty * 8 + i; if (r < N) yr[r] = acc[i][j]; }
    }
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
                 int hd, int kstart, int pos0, int window, float scale, int rows_per_token, float* out, size_t ldo, cudaStream_t s) {
    attn_gemm_kernel<false, KF><<<dim3(static_cast<unsigned>((P + BM - 1) / BM), static_cast<unsigned>((R + BN - 1) / BN)), kThreads, 0, s>>>(
        Kh, Ksc, kvw, qrows, ldq, S, ldS, P, R, hd, hd, rows_per_token, pos0 - kstart);
    masked_softmax_kernel<<<R, 256, 0, s>>>(S, ldS, P, kstart, pos0, window, scale, rows_per_token);
    attn_gemm_kernel<true, VF><<<dim3(static_cast<unsigned>((hd + BM - 1) / BM), static_cast<unsigned>((R + BN - 1) / BN)), kThreads, 0, s>>>(
        Vh, Vsc, kvw, S, ldS, out, ldo, hd, R, static_cast<int>(ldS), P, rows_per_token, pos0 - kstart);
}

}  // namespace

size_t attn_prefill_scratch_floats(int n, int pos0, int window, int heads) {
    const int kstart = window > 0 ? std::max(0, pos0 - window + 1) : 0;
    const int P = pos0 + n - kstart;   // the widest key range of any sub-chunk
    return static_cast<size_t>(std::min(n, kSub)) * heads * round_up(static_cast<size_t>(P), BK);
}

int attn_prefill_cuda(AttnFmt fmt, const float* q, const float* gate, const void* K, const float* Ksc, const void* V, const float* Vsc, int n, int pos0, int heads,
                      int kv_heads, int hd, int window, float scale, float* scratch, size_t scratch_floats, float* out, void* stream) {
    if (!q || !K || !V || !scratch || !out || n <= 0 || pos0 < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || hd % 128 != 0 || window < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (fmt == AttnFmt::TQ4) return static_cast<int>(cudaErrorNotSupported);
    const int kf = fmt == AttnFmt::Q8 ? 1 : fmt == AttnFmt::K16V8 ? 2 : 0, vf = fmt == AttnFmt::F32 ? 0 : 1;
    if ((kf == 1 && !Ksc) || (vf == 1 && !Vsc)) return static_cast<int>(cudaErrorInvalidValue);
    if (scratch_floats < attn_prefill_scratch_floats(n, pos0, window, heads)) return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    const size_t kvw = static_cast<size_t>(kv_heads) * hd;
    const int group = heads / kv_heads;
    const size_t kesz = kf == 0 ? 4 : kf == 1 ? 1 : 2, vesz = vf == 0 ? 4 : 1;
    auto pair = [&](const void* Kh, const float* Ks, const void* Vh, const float* Vs, const float* qrows, size_t ldq, int P, int R, int kstart, int p0, int rpt,
                    float* o, size_t ldo, size_t ldS) {
        if (kf == 0) launch_pair<0, 0>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s);
        else if (kf == 1) launch_pair<1, 1>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s);
        else launch_pair<2, 1>(Kh, Ks, Vh, Vs, kvw, qrows, ldq, scratch, ldS, P, R, hd, kstart, p0, window, scale, rpt, o, ldo, s);
    };
    for (int t0 = 0; t0 < n; t0 += kSub) {
        const int m = std::min(kSub, n - t0), p0 = pos0 + t0;
        const int kstart = window > 0 ? std::max(0, p0 - window + 1) : 0;
        const int P = p0 + m - kstart;
        const size_t ldS = round_up(static_cast<size_t>(P), BK);
        const float* qs = q + static_cast<size_t>(t0) * heads * hd;
        float* os = out + static_cast<size_t>(t0) * heads * hd;
        if (kv_heads == 1) {   // one K/V head: the m*heads query rows (t, h) are evenly strided (hd apart): all heads in ONE pair of GEMMs
            const size_t off = static_cast<size_t>(kstart) * kvw;
            const void* Kh = static_cast<const uint8_t*>(K) + off * kesz;
            const void* Vh = static_cast<const uint8_t*>(V) + off * vesz;
            pair(Kh, kf == 1 ? Ksc + off / 32 : nullptr, Vh, vf == 1 ? Vsc + off / 32 : nullptr, qs, static_cast<size_t>(hd), P, m * heads, kstart, p0, heads, os, hd, ldS);
        } else {
            for (int h = 0; h < heads; ++h) {
                const size_t off = static_cast<size_t>(kstart) * kvw + static_cast<size_t>(h / group) * hd;
                const void* Kh = static_cast<const uint8_t*>(K) + off * kesz;
                const void* Vh = static_cast<const uint8_t*>(V) + off * vesz;
                const size_t ldq = static_cast<size_t>(heads) * hd;
                pair(Kh, kf == 1 ? Ksc + off / 32 : nullptr, Vh, vf == 1 ? Vsc + off / 32 : nullptr, qs + static_cast<size_t>(h) * hd, ldq, P, m, kstart, p0, 1,
                     os + static_cast<size_t>(h) * hd, ldq, ldS);
            }
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return static_cast<int>(e);
    }
    if (gate) {
        const size_t total = static_cast<size_t>(n) * heads * hd;
        gate_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0, s>>>(out, gate, total);
    }
    return static_cast<int>(cudaGetLastError());
}

int attn_prefill_f32_cuda(const float* q, const float* K, const float* V, int n, int pos0, int heads, int kv_heads, int hd, int window, float scale, float* scratch,
                          size_t scratch_floats, float* out, void* stream) {
    return attn_prefill_cuda(AttnFmt::F32, q, nullptr, K, nullptr, V, nullptr, n, pos0, heads, kv_heads, hd, window, scale, scratch, scratch_floats, out, stream);
}

}  // namespace dara::model
