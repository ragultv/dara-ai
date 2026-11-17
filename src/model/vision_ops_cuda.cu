#include "model/vision_ops_cuda.h"

#include <cuda_runtime.h>

#include <cmath>

namespace dara::model {
namespace {

unsigned grid_for(size_t n) { return static_cast<unsigned>((n + 255) / 256); }

__global__ void clamp_kernel(const float* __restrict__ x, float* __restrict__ out, size_t n, float lo, float hi) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fminf(fmaxf(x[i], lo), hi);
}

// one thread per (token, head, rotated pair)
__global__ void rope2d_kernel(float* __restrict__ x, int tokens, int heads, int hd, const int* __restrict__ px, const int* __restrict__ py, float theta) {
    const int quarter = hd / 4;   // pairs per spatial half
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(tokens) * heads * 2 * quarter;
    if (idx >= total) return;
    const int i = static_cast<int>(idx % quarter);
    const int half = static_cast<int>((idx / quarter) % 2);
    const size_t th = idx / (2 * quarter);   // token * heads + head
    const int t = static_cast<int>(th / heads);
    const float p = static_cast<float>(half == 0 ? px[t] : py[t]);
    // inv_freq exactly as the reference: 1 / theta^(arange(0, spatial, 2) / spatial), spatial = hd / 2
    const float inv = 1.0f / powf(theta, static_cast<float>(2 * i) / static_cast<float>(hd / 2));
    const float ang = p * inv;
    const float c = cosf(ang), s = sinf(ang);
    float* base = x + th * hd + static_cast<size_t>(half) * (hd / 2);
    const float a = base[i], b = base[i + quarter];
    base[i] = a * c - b * s;
    base[i + quarter] = b * c + a * s;
}

// Flash-style attention, head_dim 64: a block holds 64 queries of one head (4 threads per query, 16 channels each); keys and values go
// through shared memory in tiles of 32 with an online softmax.
constexpr int kQ = 64, kKT = 32, kHD = 64;
__global__ void __launch_bounds__(256) vattn_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, int tokens, int heads,
                                                    float scale, float* __restrict__ out) {
    __shared__ float ks[kKT][kHD];
    __shared__ float vs[kKT][kHD];
    const int head = blockIdx.y;
    const int qi = blockIdx.x * kQ + threadIdx.x / 4;
    const int sub = threadIdx.x % 4;   // channels [16*sub, 16*sub + 16)
    const size_t ld = static_cast<size_t>(heads) * kHD;
    float qr[16], acc[16];
    const bool valid = qi < tokens;
    for (int j = 0; j < 16; ++j) { qr[j] = valid ? q[static_cast<size_t>(qi) * ld + head * kHD + sub * 16 + j] : 0.0f; acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int k0 = 0; k0 < tokens; k0 += kKT) {
        const int nk = min(kKT, tokens - k0);
        __syncthreads();
        for (int e = threadIdx.x; e < kKT * kHD; e += blockDim.x) {
            const int r = e / kHD, c = e % kHD;
            const bool in = r < nk;
            ks[r][c] = in ? k[static_cast<size_t>(k0 + r) * ld + head * kHD + c] : 0.0f;
            vs[r][c] = in ? v[static_cast<size_t>(k0 + r) * ld + head * kHD + c] : 0.0f;
        }
        __syncthreads();
        for (int r = 0; r < nk; ++r) {
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < 16; ++j) d = fmaf(qr[j], ks[r][sub * 16 + j], d);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            const float sc = d * scale;
            const float mn = fmaxf(m, sc);
            const float corr = expf(m - mn), p = expf(sc - mn);
            l = l * corr + p;
#pragma unroll
            for (int j = 0; j < 16; ++j) acc[j] = fmaf(p, vs[r][sub * 16 + j], acc[j] * corr);
            m = mn;
        }
    }
    if (!valid) return;
    const float inv = 1.0f / l;
    for (int j = 0; j < 16; ++j) out[static_cast<size_t>(qi) * ld + head * kHD + sub * 16 + j] = acc[j] * inv;
}

}  // namespace

int clamp_copy_cuda(const float* d_x, float* d_out, size_t n, float lo, float hi, void* stream) {
    if (!d_x || !d_out) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    clamp_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, d_out, n, lo, hi);
    return static_cast<int>(cudaGetLastError());
}

int clamp_cuda(float* d_x, size_t n, float lo, float hi, void* stream) { return clamp_copy_cuda(d_x, d_x, n, lo, hi, stream); }

int rope2d_cuda(float* d_x, int tokens, int heads, int hd, const int* d_px, const int* d_py, float theta, void* stream) {
    if (!d_x || !d_px || !d_py || tokens < 0 || heads <= 0 || hd <= 0 || hd % 4 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const size_t total = static_cast<size_t>(tokens) * heads * (hd / 2);
    if (total == 0) return 0;
    rope2d_kernel<<<grid_for(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, tokens, heads, hd, d_px, d_py, theta);
    return static_cast<int>(cudaGetLastError());
}

int vision_attention_cuda(const float* d_q, const float* d_k, const float* d_v, int tokens, int heads, float scale, float* d_out, void* stream) {
    if (!d_q || !d_k || !d_v || !d_out || tokens < 0 || heads <= 0) return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    vattn_kernel<<<dim3(static_cast<unsigned>((tokens + kQ - 1) / kQ), static_cast<unsigned>(heads)), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        d_q, d_k, d_v, tokens, heads, scale, d_out);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
