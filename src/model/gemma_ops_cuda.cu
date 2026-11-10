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

}  // namespace dara::model
