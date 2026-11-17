#include <algorithm>
#include <climits>
#include <cmath>

#include <cuda_runtime.h>

#include "model/ops_cuda.h"

namespace dara::model {
namespace {

// One block per row. Sum of squares in FP32 with a fixed reduction order (deterministic).
__global__ void rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ scale, float* __restrict__ y,
                               int n, float eps) {
    const float* xr = x + static_cast<size_t>(blockIdx.x) * n;
    float* yr = y + static_cast<size_t>(blockIdx.x) * n;
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
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * scale[i];
}

// grid (tokens, heads); thread i rotates the pair (i, i + rot/2).
__global__ void rope_kernel(float* __restrict__ x, int heads, int head_dim, int rot_dim, const int64_t* __restrict__ pos,
                            const float* __restrict__ inv_freq) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int half = rot_dim / 2;
    if (i >= half) return;
    const float angle = static_cast<float>(pos[t]) * inv_freq[i];  // fp32 angle, as the reference
    float s, c;
    sincosf(angle, &s, &c);  // full-precision (not the __sinf intrinsic)
    float* p = x + (static_cast<size_t>(t) * heads + h) * head_dim;
    const float a = p[i], b = p[i + half];
    p[i] = a * c - b * s;
    p[i + half] = b * c + a * s;
}

__global__ void rope3_kernel(float* __restrict__ x, int heads, int head_dim, int rot_dim, const int64_t* __restrict__ pos3,
                             const float* __restrict__ inv_freq, int sec_h, int sec_w) {
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const int half = rot_dim / 2;
    if (i >= half) return;
    const int s = (i % 3 == 1 && i < 3 * sec_h) ? 1 : (i % 3 == 2 && i < 3 * sec_w) ? 2 : 0;
    const float angle = static_cast<float>(pos3[static_cast<size_t>(t) * 3 + s]) * inv_freq[i];  // same arithmetic as rope_kernel
    float sn, c;
    sincosf(angle, &sn, &c);
    float* p = x + (static_cast<size_t>(t) * heads + h) * head_dim;
    const float a = p[i], b = p[i + half];
    p[i] = a * c - b * sn;
    p[i + half] = b * c + a * sn;
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x)
        out[i] = a[i] + b[i];
}

__global__ void swiglu_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float x = g[i];
        out[i] = x / (1.0f + expf(-x)) * u[i];
    }
}

}  // namespace

int add_cuda(const float* d_a, const float* d_b, float* d_out, size_t n, void* stream) {
    if (!d_a || !d_b || !d_out) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 4096));
    add_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_a, d_b, d_out, n);
    return static_cast<int>(cudaGetLastError());
}

int swiglu_cuda(const float* d_gate, const float* d_up, float* d_out, size_t n, void* stream) {
    if (!d_gate || !d_up || !d_out) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((n + 255) / 256, 4096));
    swiglu_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(d_gate, d_up, d_out, n);
    return static_cast<int>(cudaGetLastError());
}

int rmsnorm_cuda(const float* d_x, const float* d_scale, int rows, int n, float eps, float* d_y, void* stream) {
    if (rows < 0 || n <= 0 || !d_x || !d_scale || !d_y) return static_cast<int>(cudaErrorInvalidValue);
    if (rows == 0) return 0;
    const int threads = n >= 1024 ? 256 : 128;
    rmsnorm_kernel<<<rows, threads, 0, static_cast<cudaStream_t>(stream)>>>(d_x, d_scale, d_y, n, eps);
    return static_cast<int>(cudaGetLastError());
}

int rope_cuda(float* d_x, int tokens, int heads, int head_dim, int rot_dim, const int64_t* d_pos, const float* d_inv_freq,
              void* stream) {
    if (tokens < 0 || heads <= 0 || rot_dim <= 0 || rot_dim % 2 != 0 || rot_dim > head_dim || rot_dim / 2 > 1024 ||
        !d_x || !d_pos || !d_inv_freq)
        return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    rope_kernel<<<dim3(tokens, heads), rot_dim / 2, 0, static_cast<cudaStream_t>(stream)>>>(d_x, heads, head_dim, rot_dim, d_pos, d_inv_freq);
    return static_cast<int>(cudaGetLastError());
}

int rope3_cuda(float* d_x, int tokens, int heads, int head_dim, int rot_dim, const int64_t* d_pos3, const float* d_inv_freq, int sec_h, int sec_w,
               void* stream) {
    if (tokens < 0 || heads <= 0 || rot_dim <= 0 || rot_dim % 2 != 0 || rot_dim > head_dim || rot_dim / 2 > 1024 || !d_x || !d_pos3 || !d_inv_freq)
        return static_cast<int>(cudaErrorInvalidValue);
    if (tokens == 0) return 0;
    rope3_kernel<<<dim3(tokens, heads), rot_dim / 2, 0, static_cast<cudaStream_t>(stream)>>>(d_x, heads, head_dim, rot_dim, d_pos3, d_inv_freq, sec_h, sec_w);
    return static_cast<int>(cudaGetLastError());
}

// ---- greedy token selection on the device -------------------------------------------------------------------------------------------
// Two passes: kArgmaxBlocks blocks each reduce a strided slice to (value, index), then one block reduces those. Ties go to the LOWEST index and NaNs never win,
// which is exactly what the host loop `if (p[i] > p[best]) best = i` returns, so the chosen token is identical to the host argmax of the same logits.
namespace {
constexpr int kArgmaxBlocks = 64, kArgmaxThreads = 256;
__device__ __forceinline__ bool am_better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi) || (bv != bv && v == v); }
__device__ void am_block_reduce(float& v, int& i) {
    __shared__ float sv[kArgmaxThreads / 32];
    __shared__ int si[kArgmaxThreads / 32];
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (am_better(ov, oi, v, i)) { v = ov; i = oi; }
    }
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { sv[w] = v; si[w] = i; }
    __syncthreads();
    if (w == 0) {
        v = lane < kArgmaxThreads / 32 ? sv[lane] : -INFINITY;
        i = lane < kArgmaxThreads / 32 ? si[lane] : INT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_down_sync(0xffffffffu, v, off);
            const int oi = __shfl_down_sync(0xffffffffu, i, off);
            if (am_better(ov, oi, v, i)) { v = ov; i = oi; }
        }
    }
}
__global__ void argmax_pass1(const float* __restrict__ x, int n, float* pv, int* pi) {
    float v = -INFINITY; int i = INT_MAX;
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += gridDim.x * blockDim.x) {
        const float xv = x[k];
        if (am_better(xv, k, v, i)) { v = xv; i = k; }
    }
    am_block_reduce(v, i);
    if (threadIdx.x == 0) { pv[blockIdx.x] = v; pi[blockIdx.x] = i; }
}
__global__ void argmax_pass2(const float* pv, const int* pi, int* out) {
    float v = -INFINITY; int i = INT_MAX;
    for (int k = threadIdx.x; k < kArgmaxBlocks; k += blockDim.x) if (am_better(pv[k], pi[k], v, i)) { v = pv[k]; i = pi[k]; }
    am_block_reduce(v, i);
    if (threadIdx.x == 0) *out = (i == INT_MAX) ? 0 : i;
}
}  // namespace

size_t argmax_scratch_bytes() { return kArgmaxBlocks * (sizeof(float) + sizeof(int)) + 16; }

int argmax_cuda(const float* d_x, int n, void* d_scratch, int* d_out, void* stream) {
    if (!d_x || !d_scratch || !d_out || n <= 0) return static_cast<int>(cudaErrorInvalidValue);
    float* pv = static_cast<float*>(d_scratch);
    int* pi = reinterpret_cast<int*>(pv + kArgmaxBlocks);
    auto s = static_cast<cudaStream_t>(stream);
    argmax_pass1<<<kArgmaxBlocks, kArgmaxThreads, 0, s>>>(d_x, n, pv, pi);
    argmax_pass2<<<1, kArgmaxThreads, 0, s>>>(pv, pi, d_out);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
