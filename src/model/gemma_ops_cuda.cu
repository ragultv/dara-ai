#include "model/gemma_ops_cuda.h"

#include <cuda_runtime.h>

namespace dara::model {
namespace {

__device__ __forceinline__ float gelu_tanh(float x) {
    // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))  -- PyTorch's "tanh" approximation, in fp32.
    const float k = 0.7978845608028654f;
    const float inner = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void geglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[i] = gelu_tanh(g[i]) * u[i];
}

__global__ void fill_positions_kernel(int64_t* p, int64_t first, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = first + i;
}

__global__ void kv_append_kernel(float* __restrict__ Kc, float* __restrict__ Vc, const float* __restrict__ k, const float* __restrict__ v,
                                 const int64_t* __restrict__ pos, int kv_width) {
    const int t = blockIdx.x;
    const size_t dst = static_cast<size_t>(pos[t]) * kv_width, src = static_cast<size_t>(t) * kv_width;
    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) { Kc[dst + i] = k[src + i]; Vc[dst + i] = v[src + i]; }
}

__global__ void scale_kernel(float* __restrict__ x, float s, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) x[i] *= s;
}

__global__ void softcap_kernel(float* __restrict__ x, float cap, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        x[i] = cap * tanhf(x[i] / cap);
}

__global__ void combine_kernel(const float* __restrict__ proj, const float* __restrict__ tok, float* __restrict__ out, int T, int layers, int width, float scale) {
    const size_t total = static_cast<size_t>(T) * layers * width;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int j = static_cast<int>(i % width);
        const int l = static_cast<int>((i / width) % layers);
        const int t = static_cast<int>(i / (static_cast<size_t>(width) * layers));
        const float v = proj[i] + (tok ? tok[i] : 0.0f);
        out[(static_cast<size_t>(l) * T + t) * width + j] = v * scale;
    }
}

// One block per (token, query head). Phase 1: every warp computes scores for a strided subset of positions (a warp-wide dot product
// over head_dim); phase 2: block-wide softmax over the scores in shared memory; phase 3: each thread accumulates output dimensions
// over all attended positions (coalesced V reads).
constexpr int kAttnThreads = 128;

__global__ void __launch_bounds__(kAttnThreads)
gemma_attention_kernel(const float* __restrict__ q, const float* __restrict__ K, const float* __restrict__ V, const int64_t* __restrict__ pos, int heads, int kv_heads,
                       int head_dim, int window, float* __restrict__ out) {
    extern __shared__ float scores[];
    const int t = blockIdx.x, h = blockIdx.y;
    const int p = static_cast<int>(pos[t]);                  // absolute position of this query
    const int kvh = h / (heads / kv_heads);
    const int start = window > 0 ? max(0, p - window + 1) : 0;
    const int n = p - start + 1;                              // attended positions: start .. p
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    const float* qh = q + (static_cast<size_t>(t) * heads + h) * head_dim;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    constexpr int kWarps = kAttnThreads / 32;

    for (int j = warp; j < n; j += kWarps) {
        const float* kj = K + static_cast<size_t>(start + j) * kv_stride + static_cast<size_t>(kvh) * head_dim;
        float s = 0.0f;
        for (int d = lane; d < head_dim; d += 32) s = fmaf(qh[d], kj[d], s);
        for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
        if (lane == 0) scores[j] = s;  // scaling is 1.0
    }
    __syncthreads();

    __shared__ float red[kWarps];
    __shared__ float s_max, s_sum;
    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, scores[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float mm = red[0]; for (int w = 1; w < kWarps; ++w) mm = fmaxf(mm, red[w]); s_max = mm; }
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) { const float e = expf(scores[j] - s_max); scores[j] = e; sum += e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    __syncthreads();
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) { float ss = 0.0f; for (int w = 0; w < kWarps; ++w) ss += red[w]; s_sum = ss; }
    __syncthreads();
    const float inv = 1.0f / s_sum;

    float* oh = out + (static_cast<size_t>(t) * heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        const float* vcol = V + static_cast<size_t>(start) * kv_stride + static_cast<size_t>(kvh) * head_dim + d;
        for (int j = 0; j < n; ++j) acc = fmaf(scores[j], vcol[static_cast<size_t>(j) * kv_stride], acc);
        oh[d] = acc * inv;
    }
}

inline unsigned grid_for(size_t n) { const size_t b = (n + 255) / 256; return static_cast<unsigned>(b < 4096 ? b : 4096); }


// ---- fused decode kernels (fewer launches per layer; each reproduces the separate kernels' arithmetic in the same order) -------------------------
// Block-wide sum of squares with the same partition and reduction order as rmsnorm_kernel (ops_cuda.cu).
__device__ __forceinline__ float block_rms_inv(const float* xr, int n, float eps) {
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss = fmaf(xr[i], xr[i], ss);
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, off);
    __shared__ float warp_sums[32];
    __shared__ float s_r;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp] = ss;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total += warp_sums[w];
        s_r = rsqrtf(total / static_cast<float>(n) + eps);
    }
    __syncthreads();
    const float r = s_r;
    __syncthreads();   // s_r / warp_sums may be reused by a following call in the same kernel
    return r;
}

// out = (res + rmsnorm(y) * w) * scale     (post-attention / post-feed-forward / per-layer norms followed by the residual add [and the layer scalar])
__global__ void norm_residual_kernel(const float* __restrict__ y, const float* __restrict__ w, int n, float eps, const float* __restrict__ res, float scale,
                                     bool scaled, float* __restrict__ out) {
    const float* yr = y + static_cast<size_t>(blockIdx.x) * n;
    const float* rr = res + static_cast<size_t>(blockIdx.x) * n;
    float* orow = out + static_cast<size_t>(blockIdx.x) * n;
    const float r = block_rms_inv(yr, n, eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = rr[i] + yr[i] * r * w[i];
        orow[i] = scaled ? v * scale : v;
    }
}

// per (token, head): x <- rope(rmsnorm(x) * w) with rot_dim == head_dim (Gemma's q path); the head vector staged in shared memory
__global__ void norm_rope_kernel(float* __restrict__ x, const float* __restrict__ w, int heads, int hd, float eps, const int64_t* __restrict__ pos,
                                 const float* __restrict__ inv_freq) {
    extern __shared__ float sh[];
    const int t = blockIdx.x / heads;
    float* p = x + static_cast<size_t>(blockIdx.x) * hd;
    const float r = block_rms_inv(p, hd, eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) sh[i] = p[i] * r * w[i];
    __syncthreads();
    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float angle = static_cast<float>(pos[t]) * inv_freq[i];
        float s, c;
        sincosf(angle, &s, &c);
        const float a = sh[i], b = sh[i + half];
        p[i] = a * c - b * s;
        p[i + half] = b * c + a * s;
    }
}

// INT8 K/V finish for one (token, kv head): k <- rope(rmsnorm(k) * k_norm), v <- rmsnorm(v) (parameter-free), then both quantised per 32 values exactly as
// q8kv_append_kernel and written at the token's position. Replaces k_norm, rope k, v_norm and the two appends.
__global__ void kv_finish_q8_kernel(const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ kw, int kv_heads, int hd, float eps,
                                    const int64_t* __restrict__ pos, const float* __restrict__ inv_freq, int8_t* __restrict__ Kq, float* __restrict__ Ks,
                                    int8_t* __restrict__ Vq, float* __restrict__ Vs) {
    extern __shared__ float sh[];   // hd floats (k after norm), then hd floats (v after norm), then hd (k after rope)
    float* sk = sh;
    float* sv = sh + hd;
    float* skr = sh + 2 * hd;
    const int t = blockIdx.x / kv_heads, h = blockIdx.x % kv_heads;
    const float* kp = k + static_cast<size_t>(blockIdx.x) * hd;
    const float* vp = v + static_cast<size_t>(blockIdx.x) * hd;
    const float rk = block_rms_inv(kp, hd, eps);
    const float rv = block_rms_inv(vp, hd, eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) { sk[i] = kp[i] * rk * kw[i]; sv[i] = vp[i] * rv * 1.0f; }
    __syncthreads();
    const int half = hd / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float angle = static_cast<float>(pos[t]) * inv_freq[i];
        float s, c;
        sincosf(angle, &s, &c);
        const float a = sk[i], b = sk[i + half];
        skr[i] = a * c - b * s;
        skr[i + half] = b * c + a * s;
    }
    __syncthreads();
    const size_t p = static_cast<size_t>(pos[t]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int g = warp; g < hd / 32; g += nw) {   // one warp per 32-value group, as q8kv_append_kernel
        const size_t base = (p * kv_heads + h) * hd + static_cast<size_t>(g) * 32 + lane;
        const size_t sbase = (p * kv_heads + h) * (hd / 32) + g;
        {
            const float val = skr[g * 32 + lane];
            float amax = fabsf(val);
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            Kq[base] = static_cast<int8_t>(__float2int_rn(val * inv));
            if (lane == 0) Ks[sbase] = amax / 127.0f;
        }
        {
            const float val = sv[g * 32 + lane];
            float amax = fabsf(val);
            for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            Vq[base] = static_cast<int8_t>(__float2int_rn(val * inv));
            if (lane == 0) Vs[sbase] = amax / 127.0f;
        }
    }
}

}  // namespace

int kv_append_cuda(float* Kc, float* Vc, const float* k, const float* v, const int64_t* d_pos, int T, int kv_width, void* stream) {
    if (!Kc || !Vc || !k || !v || !d_pos || T < 0 || kv_width <= 0) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    kv_append_kernel<<<T, 256, 0, static_cast<cudaStream_t>(stream)>>>(Kc, Vc, k, v, d_pos, kv_width);
    return static_cast<int>(cudaGetLastError());
}

int fill_positions_cuda(int64_t* p, int64_t first, int n, void* stream) {
    if (!p || n < 0) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    fill_positions_kernel<<<(n + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(p, first, n);
    return static_cast<int>(cudaGetLastError());
}

int geglu_cuda(const float* g, const float* u, float* out, size_t n, void* stream) {
    if (!g || !u || !out) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    geglu_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(g, u, out, n);
    return static_cast<int>(cudaGetLastError());
}

int scale_cuda(float* x, float s, size_t n, void* stream) {
    if (!x) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    scale_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, s, n);
    return static_cast<int>(cudaGetLastError());
}

int softcap_cuda(float* x, float cap, size_t n, void* stream) {
    if (!x || cap <= 0.0f) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    softcap_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, cap, n);
    return static_cast<int>(cudaGetLastError());
}

int combine_per_layer_inputs_cuda(const float* proj, const float* tok, float* out, int T, int layers, int width, float scale, void* stream) {
    if (!proj || !out || T < 0 || layers <= 0 || width <= 0) return static_cast<int>(cudaErrorInvalidValue);
    const size_t n = static_cast<size_t>(T) * layers * width;
    if (n == 0) return 0;
    combine_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(proj, tok, out, T, layers, width, scale);
    return static_cast<int>(cudaGetLastError());
}

int gemma_attention_cuda(const float* q, const float* K, const float* V, const int64_t* d_pos, int T, int heads, int kv_heads, int head_dim, int window,
                         int smem_positions, float* out, void* stream) {
    if (!q || !K || !V || !d_pos || !out || T < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || head_dim <= 0 || head_dim % 32 != 0 || head_dim > 512 ||
        smem_positions <= 0 || window < 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    const int max_n = smem_positions;
    if (max_n > kGemmaMaxContext) return static_cast<int>(cudaErrorInvalidValue);
    const size_t smem = static_cast<size_t>(max_n) * sizeof(float);
    if (smem > 48 * 1024) {  // opt in to more than the default 48 KiB of dynamic shared memory (all supported GPUs allow at least 64 KiB)
        const cudaError_t e = cudaFuncSetAttribute(gemma_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (e != cudaSuccess) return static_cast<int>(e);
    }
    gemma_attention_kernel<<<dim3(T, heads), kAttnThreads, smem, static_cast<cudaStream_t>(stream)>>>(q, K, V, d_pos, heads, kv_heads, head_dim, window, out);
    return static_cast<int>(cudaGetLastError());
}


int norm_residual_cuda(const float* y, const float* w, int rows, int n, float eps, const float* res, float scale, bool scaled, float* out, void* stream) {
    if (!y || !w || !res || !out || rows < 0 || n <= 0) return static_cast<int>(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    norm_residual_kernel<<<rows, n >= 1024 ? 256 : 128, 0, static_cast<cudaStream_t>(stream)>>>(y, w, n, eps, res, scale, scaled, out);
    return static_cast<int>(cudaGetLastError());
}

int norm_rope_cuda(float* x, const float* w, int tokens, int heads, int hd, float eps, const int64_t* d_pos, const float* inv_freq, void* stream) {
    if (!x || !w || !d_pos || !inv_freq || tokens < 0 || heads <= 0 || hd <= 0 || hd % 2) return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    norm_rope_kernel<<<tokens * heads, hd >= 1024 ? 256 : 128, static_cast<size_t>(hd) * sizeof(float), static_cast<cudaStream_t>(stream)>>>(x, w, heads, hd, eps, d_pos,
                                                                                                                                         inv_freq);
    return static_cast<int>(cudaGetLastError());
}

int kv_finish_q8_cuda(const float* k, const float* v, const float* k_norm, int tokens, int kv_heads, int hd, float eps, const int64_t* d_pos, const float* inv_freq,
                      uint8_t* Kq, float* Ks, uint8_t* Vq, float* Vs, void* stream) {
    if (!k || !v || !k_norm || !d_pos || !inv_freq || !Kq || !Ks || !Vq || !Vs || tokens < 0 || kv_heads <= 0 || hd % 32) return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    kv_finish_q8_kernel<<<tokens * kv_heads, hd >= 1024 ? 256 : 128, 3 * static_cast<size_t>(hd) * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        k, v, k_norm, kv_heads, hd, eps, d_pos, inv_freq, reinterpret_cast<int8_t*>(Kq), Ks, reinterpret_cast<int8_t*>(Vq), Vs);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
