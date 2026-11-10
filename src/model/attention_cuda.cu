#include <cmath>

#include <cuda_runtime.h>

#include "model/attention_cuda.h"

namespace dara::model {
namespace {

__global__ void split_qgate_kernel(const float* __restrict__ qg, float* __restrict__ q, float* __restrict__ gate, int heads, int hd) {
    const int t = blockIdx.x, h = blockIdx.y;
    const float* src = qg + (static_cast<size_t>(t) * heads + h) * 2 * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        q[(static_cast<size_t>(t) * heads + h) * hd + i] = src[i];
        gate[(static_cast<size_t>(t) * heads + h) * hd + i] = src[hd + i];
    }
}

// Block-wide reductions over `nthreads` (a multiple of 32, <= 1024). red must hold 32 floats.
__device__ __forceinline__ float block_max(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    float m = red[0];
    for (int i = 1; i < (blockDim.x + 31) / 32; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    return m;
}
__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i) s += red[i];
    __syncthreads();
    return s;
}

// grid (heads, nq); blockDim.x == head_dim (256). One block computes one (query, head).
__global__ void attention_kernel(const float* __restrict__ q, const float* __restrict__ gate, const float* __restrict__ Kc,
                                 const float* __restrict__ Vc, float* __restrict__ out, int heads, int kv_heads, int hd, int pos0,
                                 float scale, const int64_t* __restrict__ d_pos) {
    extern __shared__ float sc[];  // scores / probabilities for positions 0..n-1
    __shared__ float qv[256];
    __shared__ float red[32];
    const int h = blockIdx.x, qi = blockIdx.y;
    const int kh = h / (heads / kv_heads);
    // pos0 comes from device memory when d_pos is given (a replayed CUDA graph cannot change launch arguments)
    const int n = (d_pos ? static_cast<int>(d_pos[0]) : pos0) + qi + 1;  // causal: this query sees positions 0 .. pos0+qi
    const int kvw = kv_heads * hd;
    const size_t qbase = (static_cast<size_t>(qi) * heads + h) * hd;

    qv[threadIdx.x] = q[qbase + threadIdx.x];
    __syncthreads();

    // scores: thread-per-key (correctness-first; not coalesced)
    float local_max = __int_as_float(0xff800000);  // -inf
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float* kj = Kc + static_cast<size_t>(j) * kvw + static_cast<size_t>(kh) * hd;
        float s = 0.0f;
#pragma unroll 8
        for (int i = 0; i < 256; ++i) s = fmaf(qv[i], kj[i], s);
        s *= scale;
        sc[j] = s;
        local_max = fmaxf(local_max, s);
    }
    const float mx = block_max(local_max, red);
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const float p = expf(sc[j] - mx);
        sc[j] = p;
        local_sum += p;
    }
    const float denom = block_sum(local_sum, red);

    // output: thread d owns output dimension d (V rows are contiguous across d: coalesced)
    float o = 0.0f;
    const float* vcol = Vc + static_cast<size_t>(kh) * hd + threadIdx.x;
    for (int j = 0; j < n; ++j) o = fmaf(sc[j], vcol[static_cast<size_t>(j) * kvw], o);
    o /= denom;
    const float g = gate[qbase + threadIdx.x];
    out[qbase + threadIdx.x] = o * (1.0f / (1.0f + expf(-g)));
}

}  // namespace

int attn_split_qgate_cuda(const AttentionDims& d, const float* d_qg, int T, float* d_q, float* d_gate, void* stream) {
    if (T < 0 || !d_qg || !d_q || !d_gate) return static_cast<int>(cudaErrorInvalidValue);
    if (T == 0) return 0;
    split_qgate_kernel<<<dim3(T, d.heads), 128, 0, static_cast<cudaStream_t>(stream)>>>(d_qg, d_q, d_gate, d.heads, d.head_dim);
    return static_cast<int>(cudaGetLastError());
}

int attn_kv_append_cuda(const AttentionDims& d, float* d_Kcache, float* d_Vcache, const float* d_k, const float* d_v, int pos0, int n,
                        void* stream) {
    if (pos0 < 0 || n < 0 || !d_Kcache || !d_Vcache || !d_k || !d_v) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    const size_t bytes = static_cast<size_t>(n) * d.kv_width() * sizeof(float);
    auto s = static_cast<cudaStream_t>(stream);
    cudaError_t e = cudaMemcpyAsync(d_Kcache + static_cast<size_t>(pos0) * d.kv_width(), d_k, bytes, cudaMemcpyDeviceToDevice, s);
    if (e != cudaSuccess) return static_cast<int>(e);
    return static_cast<int>(cudaMemcpyAsync(d_Vcache + static_cast<size_t>(pos0) * d.kv_width(), d_v, bytes, cudaMemcpyDeviceToDevice, s));
}

int attention_core_cuda(const AttentionDims& d, const float* d_q, const float* d_gate, const float* d_Kcache, const float* d_Vcache,
                        int pos0, int nq, float* d_out, void* stream) {
    if (d.head_dim != 256 || d.heads % d.kv_heads != 0 || pos0 < 0 || nq < 0 || !d_q || !d_gate || !d_Kcache || !d_Vcache || !d_out)
        return static_cast<int>(cudaErrorInvalidValue);
    if (nq == 0) return 0;
    if (pos0 + nq > kAttentionMaxKvLen) return static_cast<int>(cudaErrorInvalidConfiguration);  // needs split-KV
    const size_t smem = static_cast<size_t>(pos0 + nq) * sizeof(float);
    attention_kernel<<<dim3(d.heads, nq), 256, smem, static_cast<cudaStream_t>(stream)>>>(
        d_q, d_gate, d_Kcache, d_Vcache, d_out, d.heads, d.kv_heads, d.head_dim, pos0, 1.0f / std::sqrt(static_cast<float>(d.head_dim)), nullptr);
    return static_cast<int>(cudaGetLastError());
}

int attention_core_dev_cuda(const AttentionDims& d, const float* d_q, const float* d_gate, const float* d_Kcache, const float* d_Vcache,
                            const int64_t* d_pos, int nq, int smem_positions, float* d_out, void* stream) {
    if (d.head_dim != 256 || d.heads % d.kv_heads != 0 || smem_positions <= 0 || nq < 0 || !d_q || !d_gate || !d_Kcache || !d_Vcache || !d_pos || !d_out)
        return static_cast<int>(cudaErrorInvalidValue);
    if (nq == 0) return 0;
    if (smem_positions > kAttentionMaxKvLen) return static_cast<int>(cudaErrorInvalidConfiguration);
    const size_t smem = static_cast<size_t>(smem_positions) * sizeof(float);
    attention_kernel<<<dim3(d.heads, nq), 256, smem, static_cast<cudaStream_t>(stream)>>>(
        d_q, d_gate, d_Kcache, d_Vcache, d_out, d.heads, d.kv_heads, d.head_dim, 0, 1.0f / std::sqrt(static_cast<float>(d.head_dim)), d_pos);
    return static_cast<int>(cudaGetLastError());
}

namespace {
__global__ void kv_append_dev_kernel(float* __restrict__ Kc, float* __restrict__ Vc, const float* __restrict__ k, const float* __restrict__ v,
                                     const int64_t* __restrict__ pos, int kvw) {
    const size_t dst = static_cast<size_t>(pos[blockIdx.x]) * kvw, src = static_cast<size_t>(blockIdx.x) * kvw;
    for (int i = threadIdx.x; i < kvw; i += blockDim.x) { Kc[dst + i] = k[src + i]; Vc[dst + i] = v[src + i]; }
}
}  // namespace

int attn_kv_append_dev_cuda(const AttentionDims& d, float* d_Kcache, float* d_Vcache, const float* d_k, const float* d_v, const int64_t* d_pos, int n,
                            void* stream) {
    if (n < 0 || !d_Kcache || !d_Vcache || !d_k || !d_v || !d_pos) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    kv_append_dev_kernel<<<n, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_Kcache, d_Vcache, d_k, d_v, d_pos, d.kv_width());
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
