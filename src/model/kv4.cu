#include "model/kv4.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "model/layer_cuda.h"

namespace dara::model {
namespace {

// One warp per 32-value group of one (token, kv head): asymmetric 4-bit with an fp16 (scale, min). Codes are computed against the fp16-rounded
// scale and minimum, so decoding (c * s + min with the stored halves) is the closest representable value.
__global__ void v4_append_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos, int width, uint8_t* __restrict__ codes, __half2* __restrict__ sc) {
    const int groups = width / 32;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    if (g >= groups) return;
    const float v = x[static_cast<size_t>(t) * width + static_cast<size_t>(g) * 32 + lane];
    float mn = v, mx = v;
    for (int off = 16; off > 0; off >>= 1) { mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, off)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off)); }
    const __half hs = __float2half_rn((mx - mn) / 15.0f), hm = __float2half_rn(mn);
    const float s = __half2float(hs), m = __half2float(hm);
    const int c = s > 0.0f ? min(15, max(0, __float2int_rn((v - m) / s))) : 0;
    const int hi = __shfl_down_sync(0xffffffffu, c, 1);
    const size_t p = static_cast<size_t>(pos[t]);
    if ((lane & 1) == 0) codes[(p * width + static_cast<size_t>(g) * 32 + lane) >> 1] = static_cast<uint8_t>(c | (hi << 4));
    if (lane == 0) sc[p * groups + g] = __halves2half2(hs, hm);
}

// Q8 append into the key ring (slot = position % kKiviRing): one warp per 32-value group.
__global__ void ring_append_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos, int width, int8_t* __restrict__ ring, float* __restrict__ ring_sc) {
    const int groups = width / 32;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int t = blockIdx.y, lane = threadIdx.x & 31;
    if (g >= groups) return;
    const float v = x[static_cast<size_t>(t) * width + static_cast<size_t>(g) * 32 + lane];
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t slot = static_cast<size_t>(pos[t] % kKiviRing);
    ring[slot * width + static_cast<size_t>(g) * 32 + lane] = static_cast<int8_t>(__float2int_rn(v * inv));
    if (lane == 0) ring_sc[slot * groups + g] = amax / 127.0f;
}

// One block per candidate group: quantizes group g (positions g*32 .. g*32+31) from the ring, per channel. Each thread owns a channel pair (one byte
// of nibbles per position).
__global__ void kivi_quantize_kernel(const int64_t* __restrict__ pos, int T, int width, const int8_t* __restrict__ ring, const float* __restrict__ ring_sc,
                                     uint8_t* __restrict__ codes, __half2* __restrict__ ksc) {
    const int p_start = static_cast<int>(pos[0]), p_end = static_cast<int>(pos[T - 1]) + 1;
    const int g = kivi_frontier(p_start) / kKiviGroup + static_cast<int>(blockIdx.x);
    if (g >= kivi_frontier(p_end) / kKiviGroup) return;
    const int p0 = g * kKiviGroup;
    for (int c = 2 * threadIdx.x; c < width; c += 2 * blockDim.x) {
        float a[kKiviGroup], b[kKiviGroup];
        float amn = 3.4e38f, amx = -3.4e38f, bmn = 3.4e38f, bmx = -3.4e38f;
#pragma unroll
        for (int j = 0; j < kKiviGroup; ++j) {
            const size_t e = static_cast<size_t>((p0 + j) % kKiviRing) * width + c;
            const float sc = ring_sc[e >> 5];   // c and c + 1 share their 32-value group (c is even)
            a[j] = static_cast<float>(ring[e]) * sc;
            b[j] = static_cast<float>(ring[e + 1]) * sc;
            amn = fminf(amn, a[j]); amx = fmaxf(amx, a[j]); bmn = fminf(bmn, b[j]); bmx = fmaxf(bmx, b[j]);
        }
        const __half as = __float2half_rn((amx - amn) / 15.0f), am = __float2half_rn(amn), bs = __float2half_rn((bmx - bmn) / 15.0f), bm = __float2half_rn(bmn);
        const float fas = __half2float(as), fam = __half2float(am), fbs = __half2float(bs), fbm = __half2float(bm);
#pragma unroll
        for (int j = 0; j < kKiviGroup; ++j) {
            const int ca = fas > 0.0f ? min(15, max(0, __float2int_rn((a[j] - fam) / fas))) : 0;
            const int cb = fbs > 0.0f ? min(15, max(0, __float2int_rn((b[j] - fbm) / fbs))) : 0;
            codes[(static_cast<size_t>(p0 + j) * width + c) >> 1] = static_cast<uint8_t>(ca | (cb << 4));
        }
        ksc[static_cast<size_t>(g) * width + c] = __halves2half2(as, am);
        ksc[static_cast<size_t>(g) * width + c + 1] = __halves2half2(bs, bm);
    }
}

// Re-expands positions [p0, p0 + gridDim.y) from their 4-bit groups into the ring: one warp per 32-value group of one position.
__global__ void kivi_unquantize_kernel(int p0, int width, const uint8_t* __restrict__ codes, const __half2* __restrict__ ksc, int8_t* __restrict__ ring,
                                       float* __restrict__ ring_sc) {
    const int groups = width / 32;
    const int g = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (g >= groups) return;
    const int p = p0 + static_cast<int>(blockIdx.y);
    const int c = g * 32 + lane;
    const uint8_t byte = codes[(static_cast<size_t>(p) * width + c) >> 1];
    const float2 sm = __half22float2(ksc[static_cast<size_t>(p / kKiviGroup) * width + c]);
    const float v = static_cast<float>((c & 1) ? (byte >> 4) : (byte & 15)) * sm.x + sm.y;
    float amax = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t slot = static_cast<size_t>(p % kKiviRing);
    ring[slot * width + c] = static_cast<int8_t>(__float2int_rn(v * inv));
    if (lane == 0) ring_sc[slot * groups + g] = amax / 127.0f;
}

}  // namespace

int v4_append_cuda(const float* x, const int64_t* pos, int T, int kv_heads, int hd, uint8_t* codes, float* sc, void* stream) {
    if (!x || !pos || !codes || !sc || T < 0 || kv_heads <= 0 || hd % 32 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    const int width = kv_heads * hd, groups = width / 32;
    v4_append_kernel<<<dim3((groups + 7) / 8, T), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, pos, width, codes, reinterpret_cast<__half2*>(sc));
    return static_cast<int>(cudaGetLastError());
}

int kivi_ring_append_cuda(const float* x, const int64_t* pos, int T, int kv_heads, int hd, uint8_t* ring, float* ring_sc, void* stream) {
    if (!x || !pos || !ring || !ring_sc || T < 0 || kv_heads <= 0 || hd % 32 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T > kKiviRing - kKiviResidual - kKiviGroup) return static_cast<int>(cudaErrorInvalidValue);   // the ring must hold the call plus the residual
    if (T == 0) return 0;
    const int width = kv_heads * hd, groups = width / 32;
    ring_append_kernel<<<dim3((groups + 7) / 8, T), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, pos, width, reinterpret_cast<int8_t*>(ring), ring_sc);
    return static_cast<int>(cudaGetLastError());
}

int kivi_quantize_cuda(const int64_t* pos, int T, int width, const uint8_t* ring, const float* ring_sc, uint8_t* codes, float* ksc, void* stream) {
    if (!pos || !ring || !ring_sc || !codes || !ksc || T < 0 || width % 64 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    const int blocks = (T + kKiviGroup - 1) / kKiviGroup + 1;   // groups an append of T positions can move behind the frontier
    kivi_quantize_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(pos, T, width, reinterpret_cast<const int8_t*>(ring), ring_sc, codes,
                                                                              reinterpret_cast<__half2*>(ksc));
    return static_cast<int>(cudaGetLastError());
}

int kivi_unquantize_cuda(int p0, int p1, int width, const uint8_t* codes, const float* ksc, uint8_t* ring, float* ring_sc, void* stream) {
    if (!codes || !ksc || !ring || !ring_sc || p0 < 0 || width % 32 != 0) return static_cast<int>(cudaErrorInvalidValue);
    if (p1 <= p0) return 0;
    const int groups = width / 32;
    kivi_unquantize_kernel<<<dim3((groups + 7) / 8, p1 - p0), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        p0, width, codes, reinterpret_cast<const __half2*>(ksc), reinterpret_cast<int8_t*>(ring), ring_sc);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
