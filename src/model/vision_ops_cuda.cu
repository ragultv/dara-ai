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

// Flash-style attention: a block holds 64 queries of one head (4 threads per query, HD/4 channels each); keys and values go through shared
// memory in tiles of 32 with an online softmax.
constexpr int kQ = 64, kKT = 32;
template <int HD>
__global__ void __launch_bounds__(256) vattn_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, int tokens, int heads,
                                                    float scale, size_t ld, float* __restrict__ out) {
    constexpr int PT = HD / 4;
    __shared__ float ks[kKT][HD];
    __shared__ float vs[kKT][HD];
    const int head = blockIdx.y;
    const int qi = blockIdx.x * kQ + threadIdx.x / 4;
    const int sub = threadIdx.x % 4;
    const size_t ldo = static_cast<size_t>(heads) * HD;
    float qr[PT], acc[PT];
    const bool valid = qi < tokens;
    for (int j = 0; j < PT; ++j) { qr[j] = valid ? q[static_cast<size_t>(qi) * ld + head * HD + sub * PT + j] : 0.0f; acc[j] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (int k0 = 0; k0 < tokens; k0 += kKT) {
        const int nk = min(kKT, tokens - k0);
        __syncthreads();
        for (int e = threadIdx.x; e < kKT * HD; e += blockDim.x) {
            const int r = e / HD, c = e % HD;
            const bool in = r < nk;
            ks[r][c] = in ? k[static_cast<size_t>(k0 + r) * ld + head * HD + c] : 0.0f;
            vs[r][c] = in ? v[static_cast<size_t>(k0 + r) * ld + head * HD + c] : 0.0f;
        }
        __syncthreads();
        for (int r = 0; r < nk; ++r) {
            float d = 0.0f;
#pragma unroll
            for (int j = 0; j < PT; ++j) d = fmaf(qr[j], ks[r][sub * PT + j], d);
            d += __shfl_xor_sync(0xffffffffu, d, 1);
            d += __shfl_xor_sync(0xffffffffu, d, 2);
            const float sc = d * scale;
            const float mn = fmaxf(m, sc);
            const float corr = expf(m - mn), p = expf(sc - mn);
            l = l * corr + p;
#pragma unroll
            for (int j = 0; j < PT; ++j) acc[j] = fmaf(p, vs[r][sub * PT + j], acc[j] * corr);
            m = mn;
        }
    }
    if (!valid) return;
    const float inv = 1.0f / l;
    for (int j = 0; j < PT; ++j) out[static_cast<size_t>(qi) * ldo + head * HD + sub * PT + j] = acc[j] * inv;
}

// one thread per (token, head, i < hd/2)
__global__ void rope2d_rh_kernel(float* __restrict__ x, size_t ld, int tokens, int heads, int hd, const int* __restrict__ row, const int* __restrict__ col,
                                 float theta) {
    const int half = hd / 2, quarter = hd / 4;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(tokens) * heads * half) return;
    const int i = static_cast<int>(idx % half);
    const int h = static_cast<int>((idx / half) % heads);
    const int t = static_cast<int>(idx / (static_cast<size_t>(half) * heads));
    const int kf = i < quarter ? i : i - quarter;
    const float p = static_cast<float>(i < quarter ? row[t] : col[t]);
    const float inv = 1.0f / powf(theta, static_cast<float>(2 * kf) / static_cast<float>(half));
    const float ang = p * inv;
    const float c = cosf(ang), sn = sinf(ang);
    float* base = x + static_cast<size_t>(t) * ld + static_cast<size_t>(h) * hd;
    const float a = base[i], b = base[i + half];
    base[i] = a * c - b * sn;
    base[i + half] = b * c + a * sn;
}

__device__ float block_sum256(float v, float* red, float* bc) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    if (lane == 0) red[wp] = v;
    __syncthreads();
    if (wp == 0) {
        v = lane < static_cast<int>(blockDim.x >> 5) ? red[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) *bc = v;
    }
    __syncthreads();
    const float r = *bc;
    __syncthreads();
    return r;
}

__global__ void layernorm_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ b, int n, float eps, float* __restrict__ out) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
    float* yr = out + static_cast<size_t>(blockIdx.x) * n;
    __shared__ float red[32];
    __shared__ float bc;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum256(s, red, &bc) / static_cast<float>(n);
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = xr[i] - mean; v += d * d; }
    const float rstd = rsqrtf(block_sum256(v, red, &bc) / static_cast<float>(n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

__global__ void bias_add_kernel(float* __restrict__ y, const float* __restrict__ b, size_t total, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) y[i] += b[i % n];
}

__global__ void gelu_tanh_kernel(float* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}

__global__ void gelu_erf_kernel(float* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
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

int vision_attention_cuda(const float* d_q, const float* d_k, const float* d_v, int tokens, int heads, float scale, float* d_out, void* stream, int hd,
                          size_t ld_qkv) {
    if (!d_q || !d_k || !d_v || !d_out || tokens < 0 || heads <= 0 || (hd != 64 && hd != 72)) return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    const size_t ld = ld_qkv ? ld_qkv : static_cast<size_t>(heads) * hd;
    const dim3 grid(static_cast<unsigned>((tokens + kQ - 1) / kQ), static_cast<unsigned>(heads));
    auto st = static_cast<cudaStream_t>(stream);
    if (hd == 64) vattn_kernel<64><<<grid, 256, 0, st>>>(d_q, d_k, d_v, tokens, heads, scale, ld, d_out);
    else vattn_kernel<72><<<grid, 256, 0, st>>>(d_q, d_k, d_v, tokens, heads, scale, ld, d_out);
    return static_cast<int>(cudaGetLastError());
}

int rope2d_rh_cuda(float* d_x, size_t ld, int tokens, int heads, int hd, const int* d_row, const int* d_col, float theta, void* stream) {
    if (!d_x || !d_row || !d_col || tokens < 0 || heads <= 0 || hd % 4 != 0) return static_cast<int>(cudaErrorInvalidValue);
    const size_t total = static_cast<size_t>(tokens) * heads * (hd / 2);
    if (total == 0) return 0;
    rope2d_rh_kernel<<<grid_for(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, ld, tokens, heads, hd, d_row, d_col, theta);
    return static_cast<int>(cudaGetLastError());
}

int layernorm_cuda(const float* d_x, const float* d_w, const float* d_b, int rows, int n, float eps, float* d_out, void* stream) {
    if (!d_x || !d_w || !d_b || !d_out || rows < 0 || n <= 0) return static_cast<int>(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    layernorm_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, d_w, d_b, n, eps, d_out);
    return static_cast<int>(cudaGetLastError());
}

int bias_add_cuda(float* d_y, const float* d_b, int rows, int n, void* stream) {
    if (!d_y || !d_b || rows < 0 || n <= 0) return static_cast<int>(cudaErrorInvalidValue);
    const size_t total = static_cast<size_t>(rows) * n;
    if (total == 0) return 0;
    bias_add_kernel<<<grid_for(total), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_y, d_b, total, n);
    return static_cast<int>(cudaGetLastError());
}

int gelu_tanh_cuda(float* d_x, size_t n, void* stream) {
    if (!d_x) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    gelu_tanh_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, n);
    return static_cast<int>(cudaGetLastError());
}

int gelu_erf_cuda(float* d_x, size_t n, void* stream) {
    if (!d_x) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    gelu_erf_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(d_x, n);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
