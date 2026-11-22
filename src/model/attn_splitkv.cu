#include "model/attn_splitkv.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

#include "model/layer_cuda.h"
#include "model/tq_kv.h"

namespace dara::model {
namespace {

constexpr int kThreads = 128;  // 4 warps; one block handles kSplitChunk = 128 positions of one (token, head)
__constant__ float k_cent[16];
bool g_init_done = false;

__device__ __forceinline__ float sign_of(int i) {  // must match tq_kv.cu
    uint32_t h = static_cast<uint32_t>(i) * 0x9E3779B1u + 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h & 1u) ? -1.0f : 1.0f;
}
__device__ __forceinline__ void fwht(float* s, int d) {
    for (int h = 1; h < d; h <<= 1) {
        for (int idx = threadIdx.x; idx < d / 2; idx += blockDim.x) {
            const int i = (idx / h) * 2 * h + (idx % h);
            const float a = s[i], b = s[i + h];
            s[i] = a + b;
            s[i + h] = a - b;
        }
        __syncthreads();
    }
}

// TQ4 stage 0: qrot = H D q for every (token, head), once, instead of once per split block.
__global__ void rotate_q_kernel(const float* __restrict__ q, float* __restrict__ qrot, int heads, int d) {
    extern __shared__ float s[];
    const size_t base = (static_cast<size_t>(blockIdx.y) * heads + blockIdx.x) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s[i] = q[base + i] * sign_of(i);
    __syncthreads();
    fwht(s, d);
    for (int i = threadIdx.x; i < d; i += blockDim.x) qrot[base + i] = s[i];
}

// Stage 1: grid (splits, heads, tokens). Partial layout per (token, head, split): [m, l, o[0..hd)].
template <AttnFmt FMT>
__global__ void __launch_bounds__(kThreads)
partial_kernel(const float* __restrict__ q, const void* __restrict__ Kv, const float* __restrict__ Kn, const void* __restrict__ Vv, const float* __restrict__ Vn,
               const int64_t* __restrict__ pos, int heads, int kv_heads, int hd, int window, float scale, int splits, float* __restrict__ part, KvAux aux) {
    extern __shared__ float sq[];             // hd floats: the query (F32) or rotated query (TQ4)
    __shared__ float sc[kSplitChunk];         // scores of this chunk
    __shared__ float s_c[16];
    __shared__ float red[8];
    const int split = blockIdx.x, h = blockIdx.y, t = blockIdx.z;
    const int p = static_cast<int>(pos[t]);
    const int start = window > 0 ? max(0, p - window + 1) : 0;
    const int lo = start + split * kSplitChunk;
    if (lo > p) return;                       // beyond the current context (the grid is sized for the maximum)
    const int hi = min(lo + kSplitChunk, p + 1), cnt = hi - lo;
    const int kvh = h / (heads / kv_heads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const size_t qbase = (static_cast<size_t>(t) * heads + h) * hd;
    if (FMT == AttnFmt::TQ4 && threadIdx.x < 16) s_c[threadIdx.x] = k_cent[threadIdx.x];
    for (int i = threadIdx.x; i < hd; i += blockDim.x) sq[i] = q[qbase + i];
    __syncthreads();

    // scores: one warp per position
    for (int j = warp; j < cnt; j += kThreads / 32) {
        const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
        float acc = 0.0f;
        if (FMT == AttnFmt::F32) {
            const float* kr = static_cast<const float*>(Kv) + row * hd;
            for (int i = lane * 4; i < hd; i += 128) {
                const float4 kk = *reinterpret_cast<const float4*>(kr + i);
                const float4 qq = *reinterpret_cast<const float4*>(sq + i);
                acc = fmaf(qq.x, kk.x, acc); acc = fmaf(qq.y, kk.y, acc); acc = fmaf(qq.z, kk.z, acc); acc = fmaf(qq.w, kk.w, acc);
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) sc[j] = acc * scale;
        } else if (FMT == AttnFmt::K16V8) {
            const __half* kr = static_cast<const __half*>(Kv) + row * hd;
            for (int i = lane * 4; i < hd; i += 128) {
                const uint2 raw = *reinterpret_cast<const uint2*>(kr + i);
                const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&raw.x)), k23 = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
                const float4 qq = *reinterpret_cast<const float4*>(sq + i);
                acc = fmaf(qq.x, k01.x, acc); acc = fmaf(qq.y, k01.y, acc); acc = fmaf(qq.z, k23.x, acc); acc = fmaf(qq.w, k23.y, acc);
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) sc[j] = acc * scale;
        } else if (FMT == AttnFmt::Q4) {
            const int r = lo + j;
            const int frontier = kivi_frontier(aux.pend ? static_cast<int>(*aux.pend) + 1 : aux.held);
            if (r < frontier) {   // 4-bit, per-channel (scale, min) of the position's group
                const uint8_t* kc = static_cast<const uint8_t*>(Kv) + (row * hd >> 1);
                const __half2* ks = reinterpret_cast<const __half2*>(Kn) + static_cast<size_t>(r / kKiviGroup) * kv_heads * hd + static_cast<size_t>(kvh) * hd;
                for (int i = lane * 4; i < hd; i += 128) {
                    const uint16_t w = *reinterpret_cast<const uint16_t*>(kc + (i >> 1));
                    const float4 qq = *reinterpret_cast<const float4*>(sq + i);
                    const float2 s0 = __half22float2(ks[i]), s1 = __half22float2(ks[i + 1]), s2 = __half22float2(ks[i + 2]), s3 = __half22float2(ks[i + 3]);
                    acc = fmaf(qq.x, fmaf(static_cast<float>(w & 15u), s0.x, s0.y), acc);
                    acc = fmaf(qq.y, fmaf(static_cast<float>((w >> 4) & 15u), s1.x, s1.y), acc);
                    acc = fmaf(qq.z, fmaf(static_cast<float>((w >> 8) & 15u), s2.x, s2.y), acc);
                    acc = fmaf(qq.w, fmaf(static_cast<float>((w >> 12) & 15u), s3.x, s3.y), acc);
                }
            } else {              // the ring: Q8
                const size_t srow = static_cast<size_t>(r % kKiviRing) * kv_heads + kvh;
                const int8_t* kr = aux.ring + srow * hd;
                const float* ks = aux.ring_sc + srow * (hd / 32);
                for (int i = lane * 4; i < hd; i += 128) {
                    const char4 kk = *reinterpret_cast<const char4*>(kr + i);
                    const float4 qq = *reinterpret_cast<const float4*>(sq + i);
                    float d = qq.x * static_cast<float>(kk.x);
                    d = fmaf(qq.y, static_cast<float>(kk.y), d); d = fmaf(qq.z, static_cast<float>(kk.z), d); d = fmaf(qq.w, static_cast<float>(kk.w), d);
                    acc = fmaf(ks[i >> 5], d, acc);
                }
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) sc[j] = acc * scale;
        } else if (FMT == AttnFmt::Q8 || FMT == AttnFmt::K8V4) {
            const int8_t* kr = static_cast<const int8_t*>(Kv) + row * hd;
            const float* ks = Kn + row * (hd / 32);
            for (int i = lane * 4; i < hd; i += 128) {
                const char4 kk = *reinterpret_cast<const char4*>(kr + i);
                const float4 qq = *reinterpret_cast<const float4*>(sq + i);
                float d = qq.x * static_cast<float>(kk.x);
                d = fmaf(qq.y, static_cast<float>(kk.y), d); d = fmaf(qq.z, static_cast<float>(kk.z), d); d = fmaf(qq.w, static_cast<float>(kk.w), d);
                acc = fmaf(ks[i >> 5], d, acc);
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) sc[j] = acc * scale;
        } else {
            const uint8_t* kc = static_cast<const uint8_t*>(Kv) + row * (hd / 2);
            const int groups = hd >= 256 ? hd / 256 : 1;
            const bool active = hd >= 256 || lane < hd / 8;
            if (active) {
                for (int g = 0; g < groups; ++g) {
                    const uint32_t w = *reinterpret_cast<const uint32_t*>(kc + g * 128 + lane * 4);
#pragma unroll
                    for (int b = 0; b < 8; ++b) acc = fmaf(sq[g * 256 + lane * 8 + b], s_c[(w >> (4 * b)) & 15u], acc);
                }
            }
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
            if (lane == 0) sc[j] = acc * (Kn[row] / static_cast<float>(hd)) * scale;
        }
    }
    __syncthreads();

    // local softmax statistics
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, sc[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    const float mloc = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < cnt; j += blockDim.x) { const float e = expf(sc[j] - mloc); sc[j] = e; sum += e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    const float lloc = red[0] + red[1] + red[2] + red[3];

    // local unnormalised output
    float* pr = part + ((static_cast<size_t>(t) * heads + h) * splits + split) * (hd + 2);
    if (threadIdx.x == 0) { pr[0] = mloc; pr[1] = lloc; }
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.0f;
        if (FMT == AttnFmt::F32) {
            const float* vc = static_cast<const float*>(Vv) + static_cast<size_t>(kvh) * hd + i;
            for (int j = 0; j < cnt; ++j) acc = fmaf(sc[j], vc[static_cast<size_t>(lo + j) * kv_heads * hd], acc);
        } else if (FMT == AttnFmt::Q8 || FMT == AttnFmt::K16V8) {
            const int8_t* vc = static_cast<const int8_t*>(Vv);
            for (int j = 0; j < cnt; ++j) {
                const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
                acc = fmaf(sc[j] * Vn[row * (hd / 32) + (i >> 5)], static_cast<float>(vc[row * hd + i]), acc);
            }
        } else if (FMT == AttnFmt::K8V4 || FMT == AttnFmt::Q4) {
            const uint8_t* vc = static_cast<const uint8_t*>(Vv);
            const __half2* vs = reinterpret_cast<const __half2*>(Vn);
            const int shift = (i & 1) * 4;
            for (int j = 0; j < cnt; ++j) {
                const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
                const float2 sm = __half22float2(vs[row * (hd / 32) + (i >> 5)]);
                const float code = static_cast<float>((vc[(row * hd + i) >> 1] >> shift) & 15u);
                acc = fmaf(sc[j], fmaf(code, sm.x, sm.y), acc);
            }
        } else {
            const uint8_t* vc = static_cast<const uint8_t*>(Vv);
            for (int j = 0; j < cnt; ++j) {
                const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
                const uint8_t byte = vc[row * (hd / 2) + (i >> 1)];
                acc = fmaf(sc[j] * Vn[row], s_c[(i & 1) ? (byte >> 4) : (byte & 15)], acc);
            }
        }
        pr[2 + i] = acc;
    }
}

// Stage 1 for decode with grouped-query attention (G query heads share one K/V head: Nano 8:1, Qwen 4:1): grid (splits, kv_heads, tokens). partial_kernel runs one
// block per QUERY head, so the G blocks of a group each stream the same K/V rows: at a 64K context Nano read ~8x the cache it holds and decode fell from 74 to 19
// tok/s. Here one block scores and sums all G heads of a group, reading every K/V row ONCE. Every per-head operation and its order are exactly partial_kernel's
// (same warp-per-position scores and shuffle trees, same softmax reductions, same sequential V sum), so the partials -- and the output -- are bitwise identical.
// F32 and Q8 caches only (the default formats); G in {2, 4, 8}; HD (= hd) 256 or 512.
// Memory-level parallelism matters here (decode is bandwidth-bound and a block has only 4 warps): a warp scores TWO positions per step with every load of both
// rows issued before any arithmetic, and the V sum gives each thread HD / 128 contiguous elements so a whole V row is one coalesced load per step. Which thread
// computes which element does not change any element's arithmetic (each element's sum over positions stays sequential), so the result is unchanged.
template <AttnFmt FMT, int G, int HD>
__global__ void __launch_bounds__(kThreads)
partial_gqa_kernel(const float* __restrict__ q, const void* __restrict__ Kv, const float* __restrict__ Kn, const void* __restrict__ Vv, const float* __restrict__ Vn,
                   const int64_t* __restrict__ pos, int heads, int kv_heads, int hd, int window, float scale, int splits, float* __restrict__ part) {
    extern __shared__ float sq[];             // G * hd floats: the group's queries
    __shared__ float sc[G][kSplitChunk];      // scores of this chunk, per head
    __shared__ float red[8];
    const int split = blockIdx.x, kvh = blockIdx.y, t = blockIdx.z;
    const int p = static_cast<int>(pos[t]);
    const int start = window > 0 ? max(0, p - window + 1) : 0;
    const int lo = start + split * kSplitChunk;
    if (lo > p) return;                       // beyond the current context (the grid is sized for the maximum)
    const int hi = min(lo + kSplitChunk, p + 1), cnt = hi - lo;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int g = 0; g < G; ++g) {
        const size_t qbase = (static_cast<size_t>(t) * heads + static_cast<size_t>(kvh) * G + g) * hd;
        for (int i = threadIdx.x; i < hd; i += blockDim.x) sq[g * hd + i] = q[qbase + i];
    }
    __syncthreads();

    // scores: a warp takes positions j and j + 4 together (the K rows read once for all G heads)
    constexpr int KI = HD / 128;   // 128-element steps per row: each lane owns elements 4 * lane + 128 * k .. +3
    for (int j0 = warp; j0 < cnt; j0 += 2 * (kThreads / 32)) {
        const int jb = j0 + kThreads / 32;
        const bool hb = jb < cnt;
        const size_t rowA = static_cast<size_t>(lo + j0) * kv_heads + kvh;
        const size_t rowB = static_cast<size_t>(lo + (hb ? jb : j0)) * kv_heads + kvh;
        float acc[2][G];
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int g = 0; g < G; ++g) acc[r][g] = 0.0f;
        if (FMT == AttnFmt::F32) {
            float4 kk[2][KI];
#pragma unroll
            for (int k = 0; k < KI; ++k) {
                kk[0][k] = *reinterpret_cast<const float4*>(static_cast<const float*>(Kv) + rowA * HD + lane * 4 + 128 * k);
                kk[1][k] = *reinterpret_cast<const float4*>(static_cast<const float*>(Kv) + rowB * HD + lane * 4 + 128 * k);
            }
#pragma unroll
            for (int k = 0; k < KI; ++k)
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float4 qq = *reinterpret_cast<const float4*>(sq + g * HD + lane * 4 + 128 * k);
                        acc[r][g] = fmaf(qq.x, kk[r][k].x, acc[r][g]); acc[r][g] = fmaf(qq.y, kk[r][k].y, acc[r][g]);
                        acc[r][g] = fmaf(qq.z, kk[r][k].z, acc[r][g]); acc[r][g] = fmaf(qq.w, kk[r][k].w, acc[r][g]);
                    }
        } else {
            char4 kk[2][KI];
            float ksc[2][KI];
#pragma unroll
            for (int k = 0; k < KI; ++k) {
                kk[0][k] = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(Kv) + rowA * HD + lane * 4 + 128 * k);
                kk[1][k] = *reinterpret_cast<const char4*>(static_cast<const int8_t*>(Kv) + rowB * HD + lane * 4 + 128 * k);
                ksc[0][k] = Kn[rowA * (HD / 32) + ((lane * 4 + 128 * k) >> 5)];
                ksc[1][k] = Kn[rowB * (HD / 32) + ((lane * 4 + 128 * k) >> 5)];
            }
#pragma unroll
            for (int k = 0; k < KI; ++k)
#pragma unroll
                for (int r = 0; r < 2; ++r)
#pragma unroll
                    for (int g = 0; g < G; ++g) {
                        const float4 qq = *reinterpret_cast<const float4*>(sq + g * HD + lane * 4 + 128 * k);
                        float d = qq.x * static_cast<float>(kk[r][k].x);
                        d = fmaf(qq.y, static_cast<float>(kk[r][k].y), d); d = fmaf(qq.z, static_cast<float>(kk[r][k].z), d); d = fmaf(qq.w, static_cast<float>(kk[r][k].w), d);
                        acc[r][g] = fmaf(ksc[r][k], d, acc[r][g]);
                    }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (r == 1 && !hb) break;
#pragma unroll
            for (int g = 0; g < G; ++g) {
                float a = acc[r][g];
                for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
                if (lane == 0) sc[g][r == 0 ? j0 : jb] = a * scale;
            }
        }
    }
    __syncthreads();

    // local softmax statistics per head (the same reductions as partial_kernel, one head after another)
    float mloc[G], lloc[G];
    for (int g = 0; g < G; ++g) {
        float m = -3.402823466e38f;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, sc[g][j]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (lane == 0) red[warp] = m;
        __syncthreads();
        const float mg = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        __syncthreads();
        float sum = 0.0f;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) { const float e = expf(sc[g][j] - mg); sc[g][j] = e; sum += e; }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) red[warp] = sum;
        __syncthreads();
        mloc[g] = mg; lloc[g] = red[0] + red[1] + red[2] + red[3];
        __syncthreads();
    }

    // local unnormalised outputs: the V row read once for all G heads
    float* pr0 = part + ((static_cast<size_t>(t) * heads + static_cast<size_t>(kvh) * G) * splits + split) * (hd + 2);
    const size_t head_stride = static_cast<size_t>(splits) * (hd + 2);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int g = 0; g < G; ++g) { pr0[g * head_stride] = mloc[g]; pr0[g * head_stride + 1] = lloc[g]; }
    }
    constexpr int EPT = HD / 128;   // contiguous elements per thread (HD 512: 4, 256: 2)
    const int i0 = threadIdx.x * EPT;
    float acc[EPT][G];
#pragma unroll
    for (int e = 0; e < EPT; ++e)
#pragma unroll
        for (int g = 0; g < G; ++g) acc[e][g] = 0.0f;
    if (FMT == AttnFmt::F32) {
        const float* vc = static_cast<const float*>(Vv) + static_cast<size_t>(kvh) * HD + i0;
#pragma unroll 4
        for (int j = 0; j < cnt; ++j) {
            const float* vp = vc + static_cast<size_t>(lo + j) * kv_heads * HD;
            float v[EPT];
            if constexpr (EPT == 4) { const float4 t4 = *reinterpret_cast<const float4*>(vp); v[0] = t4.x; v[1] = t4.y; v[2] = t4.z; v[3] = t4.w; }
            else { const float2 t2 = *reinterpret_cast<const float2*>(vp); v[0] = t2.x; v[1] = EPT > 1 ? t2.y : 0.0f; }
#pragma unroll
            for (int e = 0; e < EPT; ++e)
#pragma unroll
                for (int g = 0; g < G; ++g) acc[e][g] = fmaf(sc[g][j], v[e], acc[e][g]);
        }
    } else {
        const int8_t* vc = static_cast<const int8_t*>(Vv);
#pragma unroll 4
        for (int j = 0; j < cnt; ++j) {
            const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
            float v[EPT];
            if constexpr (EPT == 4) { const char4 c4 = *reinterpret_cast<const char4*>(vc + row * HD + i0); v[0] = c4.x; v[1] = c4.y; v[2] = c4.z; v[3] = c4.w; }
            else { const char2 c2 = *reinterpret_cast<const char2*>(vc + row * HD + i0); v[0] = c2.x; v[1] = EPT > 1 ? c2.y : 0; }
            const float vs = Vn[row * (HD / 32) + (i0 >> 5)];
#pragma unroll
            for (int e = 0; e < EPT; ++e)
#pragma unroll
                for (int g = 0; g < G; ++g) acc[e][g] = fmaf(sc[g][j] * vs, v[e], acc[e][g]);
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int e = 0; e < EPT; ++e) pr0[g * head_stride + 2 + i0 + e] = acc[e][g];
}

// Stage 1 for verification steps (2..kMultiTokens tokens at consecutive positions, full attention): grid (splits, heads). One block reads each
// K / V entry of its chunk ONCE and applies it to every token, instead of one block per token re-reading the chunk (with many draft tokens at a long
// context the K / V reads dominated the step). Per token, every operation and its order are those of partial_kernel, so the partials -- and the
// output -- are bitwise identical to the per-token kernel.
constexpr int kMultiTokens = 8;
template <AttnFmt FMT>
__global__ void __launch_bounds__(kThreads)
partial_multi_kernel(const float* __restrict__ q, const void* __restrict__ Kv, const float* __restrict__ Kn, const void* __restrict__ Vv, const float* __restrict__ Vn,
                     const int64_t* __restrict__ pos, int n, int heads, int kv_heads, int hd, float scale, int splits, float* __restrict__ part, KvAux aux) {
    extern __shared__ float sq[];                    // [n][hd] queries
    __shared__ float sc[kMultiTokens][kSplitChunk];  // scores, then probabilities, per token
    __shared__ float red[8];
    __shared__ int s_cnt[kMultiTokens];
    const int split = blockIdx.x, h = blockIdx.y;
    const int lo = split * kSplitChunk;
    int pmax = 0;
    for (int t = 0; t < n; ++t) pmax = max(pmax, static_cast<int>(pos[t]));
    if (lo > pmax) return;
    const int cnt_max = min(lo + kSplitChunk, pmax + 1) - lo;
    const int kvh = h / (heads / kv_heads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (threadIdx.x < kMultiTokens) s_cnt[threadIdx.x] = threadIdx.x < n ? max(0, min(lo + kSplitChunk, static_cast<int>(pos[threadIdx.x]) + 1) - lo) : 0;
    for (int t = 0; t < n; ++t)
        for (int i = threadIdx.x; i < hd; i += blockDim.x) sq[t * hd + i] = q[(static_cast<size_t>(t) * heads + h) * hd + i];
    __syncthreads();
    const int frontier = FMT == AttnFmt::Q4 ? kivi_frontier(aux.pend ? static_cast<int>(*aux.pend) + 1 : aux.held) : 0;

    // scores: one warp per position, every token's dot product from the same K reads
    for (int j = warp; j < cnt_max; j += kThreads / 32) {
        const int r = lo + j;
        const size_t row = static_cast<size_t>(r) * kv_heads + kvh;
        float acc[kMultiTokens];
#pragma unroll
        for (int t = 0; t < kMultiTokens; ++t) acc[t] = 0.0f;
        for (int i = lane * 4; i < hd; i += 128) {
            if (FMT == AttnFmt::F32) {
                const float4 kk = *reinterpret_cast<const float4*>(static_cast<const float*>(Kv) + row * hd + i);
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) {
                    if (t >= n) break;
                    const float4 qq = *reinterpret_cast<const float4*>(sq + t * hd + i);
                    acc[t] = fmaf(qq.x, kk.x, acc[t]); acc[t] = fmaf(qq.y, kk.y, acc[t]); acc[t] = fmaf(qq.z, kk.z, acc[t]); acc[t] = fmaf(qq.w, kk.w, acc[t]);
                }
            } else if (FMT == AttnFmt::K16V8) {
                const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const __half*>(Kv) + row * hd + i);
                const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&raw.x)), k23 = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) {
                    if (t >= n) break;
                    const float4 qq = *reinterpret_cast<const float4*>(sq + t * hd + i);
                    acc[t] = fmaf(qq.x, k01.x, acc[t]); acc[t] = fmaf(qq.y, k01.y, acc[t]); acc[t] = fmaf(qq.z, k23.x, acc[t]); acc[t] = fmaf(qq.w, k23.y, acc[t]);
                }
            } else if (FMT == AttnFmt::Q4 && r < frontier) {
                const uint16_t w = *reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(Kv) + (row * hd >> 1) + (i >> 1));
                const __half2* ks = reinterpret_cast<const __half2*>(Kn) + static_cast<size_t>(r / kKiviGroup) * kv_heads * hd + static_cast<size_t>(kvh) * hd;
                const float2 s0 = __half22float2(ks[i]), s1 = __half22float2(ks[i + 1]), s2 = __half22float2(ks[i + 2]), s3 = __half22float2(ks[i + 3]);
                const float k0 = fmaf(static_cast<float>(w & 15u), s0.x, s0.y), k1 = fmaf(static_cast<float>((w >> 4) & 15u), s1.x, s1.y);
                const float k2 = fmaf(static_cast<float>((w >> 8) & 15u), s2.x, s2.y), k3 = fmaf(static_cast<float>((w >> 12) & 15u), s3.x, s3.y);
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) {
                    if (t >= n) break;
                    const float4 qq = *reinterpret_cast<const float4*>(sq + t * hd + i);
                    acc[t] = fmaf(qq.x, k0, acc[t]); acc[t] = fmaf(qq.y, k1, acc[t]); acc[t] = fmaf(qq.z, k2, acc[t]); acc[t] = fmaf(qq.w, k3, acc[t]);
                }
            } else {   // Q8 / K8V4 keys, or the Q4 ring (Q8)
                const int8_t* kr;
                const float* ks;
                if (FMT == AttnFmt::Q4) {
                    const size_t srow = static_cast<size_t>(r % kKiviRing) * kv_heads + kvh;
                    kr = aux.ring + srow * hd; ks = aux.ring_sc + srow * (hd / 32);
                } else {
                    kr = static_cast<const int8_t*>(Kv) + row * hd; ks = Kn + row * (hd / 32);
                }
                const char4 kk = *reinterpret_cast<const char4*>(kr + i);
                const float sk = ks[i >> 5];
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) {
                    if (t >= n) break;
                    const float4 qq = *reinterpret_cast<const float4*>(sq + t * hd + i);
                    float d = qq.x * static_cast<float>(kk.x);
                    d = fmaf(qq.y, static_cast<float>(kk.y), d); d = fmaf(qq.z, static_cast<float>(kk.z), d); d = fmaf(qq.w, static_cast<float>(kk.w), d);
                    acc[t] = fmaf(sk, d, acc[t]);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kMultiTokens; ++t) {
            if (t >= n) break;
            float a = acc[t];
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
            if (lane == 0 && j < s_cnt[t]) sc[t][j] = a * scale;
        }
    }
    __syncthreads();

    // per token: local softmax statistics (as partial_kernel), unnormalised probabilities in place
    float mloc[kMultiTokens], lloc[kMultiTokens];
    for (int t = 0; t < n; ++t) {
        const int cnt = s_cnt[t];
        float m = -3.402823466e38f;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) m = fmaxf(m, sc[t][j]);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if (lane == 0) red[warp] = m;
        __syncthreads();
        mloc[t] = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
        __syncthreads();
        float sum = 0.0f;
        for (int j = threadIdx.x; j < cnt; j += blockDim.x) { const float e = expf(sc[t][j] - mloc[t]); sc[t][j] = e; sum += e; }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
        if (lane == 0) red[warp] = sum;
        __syncthreads();
        lloc[t] = red[0] + red[1] + red[2] + red[3];
        __syncthreads();
    }

    // local unnormalised outputs: each V entry read once, applied to every token that sees it
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc[kMultiTokens];
#pragma unroll
        for (int t = 0; t < kMultiTokens; ++t) acc[t] = 0.0f;
        for (int j = 0; j < cnt_max; ++j) {
            const size_t row = static_cast<size_t>(lo + j) * kv_heads + kvh;
            if (FMT == AttnFmt::F32) {
                const float v = static_cast<const float*>(Vv)[row * hd + i];
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) if (t < n && j < s_cnt[t]) acc[t] = fmaf(sc[t][j], v, acc[t]);
            } else if (FMT == AttnFmt::K8V4 || FMT == AttnFmt::Q4) {
                const float2 sm = __half22float2(reinterpret_cast<const __half2*>(Vn)[row * (hd / 32) + (i >> 5)]);
                const float code = static_cast<float>((static_cast<const uint8_t*>(Vv)[(row * hd + i) >> 1] >> ((i & 1) * 4)) & 15u);
                const float v = fmaf(code, sm.x, sm.y);
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) if (t < n && j < s_cnt[t]) acc[t] = fmaf(sc[t][j], v, acc[t]);
            } else {   // Q8 / K16V8 values
                const float vs = Vn[row * (hd / 32) + (i >> 5)];
                const float v = static_cast<float>(static_cast<const int8_t*>(Vv)[row * hd + i]);
#pragma unroll
                for (int t = 0; t < kMultiTokens; ++t) if (t < n && j < s_cnt[t]) acc[t] = fmaf(sc[t][j] * vs, v, acc[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < kMultiTokens; ++t) {
            if (t >= n || s_cnt[t] == 0) continue;
            float* pr = part + ((static_cast<size_t>(t) * heads + h) * splits + split) * (hd + 2);
            if (i == threadIdx.x && threadIdx.x == 0) { pr[0] = mloc[t]; pr[1] = lloc[t]; }
            pr[2 + i] = acc[t];
        }
    }
}

// Stage 2: grid (heads, tokens): merge the partials of one (token, head).
template <AttnFmt FMT>
__global__ void __launch_bounds__(256)
combine_kernel(const float* __restrict__ part, const float* __restrict__ gate, const int64_t* __restrict__ pos, int heads, int hd, int window, int splits,
               float* __restrict__ out) {
    extern __shared__ float sw[];  // hd floats (TQ4 inverse rotation)
    const int h = blockIdx.x, t = blockIdx.y;
    const int p = static_cast<int>(pos[t]);
    const int start = window > 0 ? max(0, p - window + 1) : 0;
    const int ns = (p - start) / kSplitChunk + 1;  // valid splits for this token
    const float* base = part + (static_cast<size_t>(t) * heads + h) * splits * (hd + 2);
    float M = -3.402823466e38f;
    for (int s = 0; s < ns; ++s) M = fmaxf(M, base[static_cast<size_t>(s) * (hd + 2)]);
    float L = 0.0f;
    for (int s = 0; s < ns; ++s) L += base[static_cast<size_t>(s) * (hd + 2) + 1] * expf(base[static_cast<size_t>(s) * (hd + 2)] - M);
    const float invL = 1.0f / L;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.0f;
        for (int s = 0; s < ns; ++s) o += base[static_cast<size_t>(s) * (hd + 2) + 2 + i] * expf(base[static_cast<size_t>(s) * (hd + 2)] - M);
        o *= invL;
        if (FMT == AttnFmt::TQ4) sw[i] = o;
        else {
            if (gate) { const float g = gate[(static_cast<size_t>(t) * heads + h) * hd + i]; o *= 1.0f / (1.0f + expf(-g)); }
            out[(static_cast<size_t>(t) * heads + h) * hd + i] = o;
        }
    }
    if (FMT == AttnFmt::TQ4) {
        __syncthreads();
        fwht(sw, hd);
        for (int i = threadIdx.x; i < hd; i += blockDim.x) {
            float o = sign_of(i) * sw[i] / static_cast<float>(hd);
            if (gate) { const float g = gate[(static_cast<size_t>(t) * heads + h) * hd + i]; o *= 1.0f / (1.0f + expf(-g)); }
            out[(static_cast<size_t>(t) * heads + h) * hd + i] = o;
        }
    }
}

// INT8 K/V append: one warp per 32-value group of one (token, kv head).
__global__ void q8kv_append_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos, int kv_heads, int hd, int8_t* __restrict__ codes, float* __restrict__ scales) {
    const int groups = hd / 32;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   // group index within the token's kv_heads * groups
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    if (g >= kv_heads * groups) return;
    const float v = x[(static_cast<size_t>(t) * kv_heads * hd) + static_cast<size_t>(g) * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t p = static_cast<size_t>(pos[t]);
    codes[(p * kv_heads * hd) + static_cast<size_t>(g) * 32 + lane] = static_cast<int8_t>(__float2int_rn(v * inv));
    if (lane == 0) scales[p * kv_heads * groups + g] = amax / 127.0f;
}

__global__ void f16kv_append_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos, int width, __half* __restrict__ out) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    out[static_cast<size_t>(pos[t]) * width + i] = __float2half_rn(x[static_cast<size_t>(t) * width + i]);
}

}  // namespace

int f16kv_append_cuda(const float* x, const int64_t* pos, int T, int kv_heads, int hd, uint8_t* out, void* stream) {
    if (!x || !pos || !out || T < 0 || kv_heads <= 0 || hd <= 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    const int width = kv_heads * hd;
    f16kv_append_kernel<<<dim3((width + 255) / 256, T), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, pos, width, reinterpret_cast<__half*>(out));
    return static_cast<int>(cudaGetLastError());
}

int q8kv_append_cuda(const float* x, const int64_t* pos, int T, int kv_heads, int hd, uint8_t* codes, float* scales, void* stream) {
    if (!x || !pos || !codes || !scales || T < 0 || kv_heads <= 0 || hd <= 0 || hd % 32 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    const int groups = kv_heads * (hd / 32);
    q8kv_append_kernel<<<dim3((groups + 7) / 8, T), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, pos, kv_heads, hd, reinterpret_cast<int8_t*>(codes), scales);
    return static_cast<int>(cudaGetLastError());
}

int attn_splitkv_init() {
    if (g_init_done) return 0;
    const cudaError_t e = cudaMemcpyToSymbol(k_cent, tq_centroids(), 16 * sizeof(float));
    if (e != cudaSuccess) return static_cast<int>(e);
    g_init_done = true;
    return 0;
}

static int splits_for(int max_positions) { return std::max(1, (max_positions + kSplitChunk - 1) / kSplitChunk); }

// Grouped-query decode kernel on/off: DARA_GQA_ATTN=0 (or attn_splitkv_set_gqa(0)) selects the per-head kernels, which give bitwise identical output.
static int g_gqa_mode = -1;   // -1: environment (default on), 0: off, 1: on
void attn_splitkv_set_gqa(int mode) { g_gqa_mode = mode; }
static bool gqa_enabled() {
    if (g_gqa_mode >= 0) return g_gqa_mode == 1;
    static const bool on = [] { const char* e = std::getenv("DARA_GQA_ATTN"); return !(e && e[0] == '0'); }();
    return on;
}

size_t attn_splitkv_scratch_floats_per_token(AttnFmt fmt, int heads, int hd, int max_positions) {
    const size_t partials = static_cast<size_t>(heads) * splits_for(max_positions) * (hd + 2);
    return partials + (fmt == AttnFmt::TQ4 ? static_cast<size_t>(heads) * hd : 0);
}

int attn_splitkv_cuda(AttnFmt fmt, const float* q, const float* gate, const void* K, const float* Kn, const void* V, const float* Vn, const int64_t* pos, int T,
                      int heads, int kv_heads, int hd, int window, float scale, int max_positions, float* scratch, size_t scratch_floats, float* out, void* stream,
                      const KvAux& aux) {
    if (fmt == AttnFmt::Q4 && (!aux.ring || !aux.ring_sc)) return static_cast<int>(cudaErrorInvalidValue);
    if (!q || !K || !V || !pos || !out || !scratch || T < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || max_positions <= 0 || window < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (fmt != AttnFmt::TQ4 ? (hd % 128 != 0 || hd > 512) : !tq_supported_head_dim(hd)) return static_cast<int>(cudaErrorInvalidValue);
    if ((fmt == AttnFmt::Q8 || fmt == AttnFmt::TQ4) && !Kn) return static_cast<int>(cudaErrorInvalidValue);
    if (fmt != AttnFmt::F32 && !Vn) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    auto s = static_cast<cudaStream_t>(stream);
    const int splits = splits_for(max_positions);
    const size_t per_token = attn_splitkv_scratch_floats_per_token(fmt, heads, hd, max_positions);
    if (scratch_floats < per_token) return static_cast<int>(cudaErrorInvalidValue);
    const int group = static_cast<int>(std::min<size_t>(static_cast<size_t>(T), scratch_floats / per_token));
    for (int t0 = 0; t0 < T; t0 += group) {
        const int n = std::min(group, T - t0);
        const float* qg = q + static_cast<size_t>(t0) * heads * hd;
        float* og = out + static_cast<size_t>(t0) * heads * hd;
        const float* gg = gate ? gate + static_cast<size_t>(t0) * heads * hd : nullptr;
        const int64_t* pg = pos + t0;
        float* partials = scratch;
        const float* qsrc = qg;
        if (fmt == AttnFmt::TQ4) {
            float* qrot = scratch + static_cast<size_t>(n) * heads * splits * (hd + 2);
            rotate_q_kernel<<<dim3(heads, n), 256, static_cast<size_t>(hd) * sizeof(float), s>>>(qg, qrot, heads, hd);
            qsrc = qrot;
        }
        const size_t smem = static_cast<size_t>(hd) * sizeof(float);
        const dim3 grid(splits, heads, n);
        static const bool multi_on = [] { const char* e = std::getenv("DARA_MULTI_VERIFY"); return !(e && e[0] == '0'); }();
        if (multi_on && n >= 2 && n <= kMultiTokens && window == 0 && fmt != AttnFmt::TQ4) {
            // several tokens, full attention: one block per (chunk, head) reads K/V once for all of them (bitwise identical partials)
            const dim3 g2(splits, heads);
            const size_t sm2 = static_cast<size_t>(n) * hd * sizeof(float);
            switch (fmt) {
                case AttnFmt::F32: partial_multi_kernel<AttnFmt::F32><<<g2, kThreads, sm2, s>>>(qsrc, K, Kn, V, Vn, pg, n, heads, kv_heads, hd, scale, splits, partials, aux); break;
                case AttnFmt::Q8: partial_multi_kernel<AttnFmt::Q8><<<g2, kThreads, sm2, s>>>(qsrc, K, Kn, V, Vn, pg, n, heads, kv_heads, hd, scale, splits, partials, aux); break;
                case AttnFmt::K16V8: partial_multi_kernel<AttnFmt::K16V8><<<g2, kThreads, sm2, s>>>(qsrc, K, Kn, V, Vn, pg, n, heads, kv_heads, hd, scale, splits, partials, aux); break;
                case AttnFmt::K8V4: partial_multi_kernel<AttnFmt::K8V4><<<g2, kThreads, sm2, s>>>(qsrc, K, Kn, V, Vn, pg, n, heads, kv_heads, hd, scale, splits, partials, aux); break;
                case AttnFmt::Q4: partial_multi_kernel<AttnFmt::Q4><<<g2, kThreads, sm2, s>>>(qsrc, K, Kn, V, Vn, pg, n, heads, kv_heads, hd, scale, splits, partials, aux); break;
                default: break;
            }
            combine_kernel<AttnFmt::F32><<<dim3(heads, n), 256, smem, s>>>(partials, gg, pg, heads, hd, window, splits, og);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) return static_cast<int>(e);
            continue;
        }
        // Only full-attention layers: a sliding-window layer reads at most `window` positions (4-5 chunks), so the per-head kernels' G-times-more blocks run it faster.
        if (gqa_enabled() && window == 0 && (fmt == AttnFmt::F32 || fmt == AttnFmt::Q8) && (hd == 256 || hd == 512)) {
            const int G = heads / kv_heads;
            if (G == 2 || G == 4 || G == 8) {
                // grouped-query decode: one block per (chunk, kv head, token) serves all G query heads of the group (K/V read once; bitwise identical partials)
                const dim3 gq(splits, kv_heads, n);
                const size_t sg = static_cast<size_t>(G) * hd * sizeof(float);
#define DARA_GQA_LAUNCH(F, GG, HH) partial_gqa_kernel<F, GG, HH><<<gq, kThreads, sg, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials)
#define DARA_GQA_BY_G(F, HH) { if (G == 2) DARA_GQA_LAUNCH(F, 2, HH); else if (G == 4) DARA_GQA_LAUNCH(F, 4, HH); else DARA_GQA_LAUNCH(F, 8, HH); }
                if (fmt == AttnFmt::F32) { if (hd == 512) DARA_GQA_BY_G(AttnFmt::F32, 512) else DARA_GQA_BY_G(AttnFmt::F32, 256) }
                else { if (hd == 512) DARA_GQA_BY_G(AttnFmt::Q8, 512) else DARA_GQA_BY_G(AttnFmt::Q8, 256) }
#undef DARA_GQA_BY_G
#undef DARA_GQA_LAUNCH
                combine_kernel<AttnFmt::F32><<<dim3(heads, n), 256, smem, s>>>(partials, gg, pg, heads, hd, window, splits, og);
                const cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) return static_cast<int>(e);
                continue;
            }
        }
        switch (fmt) {
            case AttnFmt::F32: partial_kernel<AttnFmt::F32><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
            case AttnFmt::Q8: partial_kernel<AttnFmt::Q8><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
            case AttnFmt::K16V8: partial_kernel<AttnFmt::K16V8><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
            case AttnFmt::K8V4: partial_kernel<AttnFmt::K8V4><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
            case AttnFmt::Q4: partial_kernel<AttnFmt::Q4><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
            case AttnFmt::TQ4: partial_kernel<AttnFmt::TQ4><<<grid, kThreads, smem, s>>>(qsrc, K, Kn, V, Vn, pg, heads, kv_heads, hd, window, scale, splits, partials, aux); break;
        }
        if (fmt != AttnFmt::TQ4)   // F32 and Q8 partials are in the plain (unrotated) domain
            combine_kernel<AttnFmt::F32><<<dim3(heads, n), 256, smem, s>>>(partials, gg, pg, heads, hd, window, splits, og);
        else
            combine_kernel<AttnFmt::TQ4><<<dim3(heads, n), 256, smem, s>>>(partials, gg, pg, heads, hd, window, splits, og);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return static_cast<int>(e);
    }
    return 0;
}

}  // namespace dara::model
