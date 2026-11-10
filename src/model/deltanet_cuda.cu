// Gated DeltaNet decode step (see deltanet_cuda.h and deltanet.h for the equations).
//
// Kernel 1 (conv): one thread per channel: window = [3 stored inputs, current input], tap 0 on the oldest,
//   silu, shift the state.
// Kernel 2 (recurrence): one block per V head, 128 threads, thread v owns column v of the head's 128x128 state
//   S[:, v] in REGISTERS. It loads the column once, applies decay, forms kv = S^T k, the delta update and the
//   output o = S^T q in registers, and stores the column once: the 64 KiB state per head is read and written
//   exactly once per token. q/k of the head's key head are l2-normalised in shared memory first (the key head
//   follows the V-head ORDER: v/ratio for checkpoint order, v % k_heads for GGUF order). The gated RMSNorm
//   (norm_w * rmsnorm(o) * silu(z)) is fused at the end.
#include <climits>
#include <cmath>

#include <cuda_runtime.h>

#include "model/deltanet_cuda.h"

namespace dara::model {
namespace {

constexpr int kDim = 128;  // k_dim == v_dim == block size

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x * sigmoidf_(x); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__global__ void conv_kernel(const float* __restrict__ w, float* __restrict__ state, const float* __restrict__ qkv,
                            float* __restrict__ out, int channels, int taps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int hist = taps - 1;
    float* h = state + static_cast<size_t>(c) * hist;
    const float* wc = w + static_cast<size_t>(c) * taps;
    float cur = qkv[c];
    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) acc += wc[k] * h[k];
    acc += wc[taps - 1] * cur;
    out[c] = siluf_(acc);
    for (int k = 0; k + 1 < hist; ++k) h[k] = h[k + 1];
    h[hist - 1] = cur;
}

__device__ __forceinline__ float block_sum_128(float v, float* red /* 4 floats */) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const float total = red[0] + red[1] + red[2] + red[3];
    __syncthreads();  // red is reused by the next reduction
    return total;
}

__global__ void __launch_bounds__(kDim)
recurrence_kernel(const float* __restrict__ conv_out, const float* __restrict__ z, const float* __restrict__ b,
                  const float* __restrict__ a, const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
                  const float* __restrict__ norm_w, float* __restrict__ S_all, float* __restrict__ normed,
                  int k_heads, int v_heads, int key_dim, int gguf_order, float eps) {
    const int h = blockIdx.x;
    const int v = threadIdx.x;
    const int kh = gguf_order ? (h % k_heads) : (h / (v_heads / k_heads));

    __shared__ float qs[kDim], ks[kDim], red[4];
    __shared__ float s_beta, s_decay;

    const float* q_raw = conv_out + static_cast<size_t>(kh) * kDim;
    const float* k_raw = conv_out + key_dim + static_cast<size_t>(kh) * kDim;
    const float qv = q_raw[v], kv_ = k_raw[v];
    const float qsum = block_sum_128(qv * qv, red);
    const float ksum = block_sum_128(kv_ * kv_, red);
    qs[v] = qv * rsqrtf(qsum + 1e-6f) * rsqrtf(static_cast<float>(kDim));  // l2norm, then the 1/sqrt(k_dim) scale
    ks[v] = kv_ * rsqrtf(ksum + 1e-6f);
    if (v == 0) {
        s_beta = sigmoidf_(b[h]);
        s_decay = expf(ssm_a[h] * softplusf_(a[h] + dt_bias[h]));
    }
    __syncthreads();

    float* S = S_all + static_cast<size_t>(h) * kDim * kDim;  // [k][v]
    float col[kDim];
#pragma unroll
    for (int k = 0; k < kDim; ++k) col[k] = S[static_cast<size_t>(k) * kDim + v];

    const float decay = s_decay, beta = s_beta;
    float kvm = 0.0f;
#pragma unroll
    for (int k = 0; k < kDim; ++k) {
        col[k] *= decay;
        kvm = fmaf(col[k], ks[k], kvm);
    }
    const float vt = conv_out[2 * key_dim + static_cast<size_t>(h) * kDim + v];
    const float delta = (vt - kvm) * beta;
    float o = 0.0f;
#pragma unroll
    for (int k = 0; k < kDim; ++k) {
        col[k] = fmaf(ks[k], delta, col[k]);
        o = fmaf(col[k], qs[k], o);
        S[static_cast<size_t>(k) * kDim + v] = col[k];
    }

    // Gated RMSNorm over this head's 128 outputs: norm_w * (o * rsqrt(mean(o^2) + eps)) * silu(z)
    const float ms = block_sum_128(o * o, red) / static_cast<float>(kDim);
    const float r = rsqrtf(ms + eps);
    const size_t idx = static_cast<size_t>(h) * kDim + v;
    normed[idx] = norm_w[v] * (o * r) * siluf_(z[idx]);
}

}  // namespace

size_t deltanet_conv_state_floats(const DeltaNetDims& d) { return static_cast<size_t>(d.conv_dim()) * (d.conv_kernel - 1); }
size_t deltanet_S_floats(const DeltaNetDims& d) { return static_cast<size_t>(d.v_heads) * d.k_dim * d.v_dim; }

int deltanet_step_cuda(const DeltaNetDims& d, VHeadOrder order, const DeltaNetDeviceParams& p, float* d_conv_state,
                       float* d_S, const float* d_qkv, const float* d_z, const float* d_b, const float* d_a,
                       float* d_conv_scratch, float* d_normed, void* stream) {
    if (d.k_dim != kDim || d.v_dim != kDim || d.conv_kernel < 2 || d.conv_kernel > 8 || d.v_heads % d.k_heads != 0)
        return static_cast<int>(cudaErrorInvalidValue);
    auto s = static_cast<cudaStream_t>(stream);
    const int ch = d.conv_dim();
    conv_kernel<<<(ch + 255) / 256, 256, 0, s>>>(p.conv_w, d_conv_state, d_qkv, d_conv_scratch, ch, d.conv_kernel);
    recurrence_kernel<<<d.v_heads, kDim, 0, s>>>(d_conv_scratch, d_z, d_b, d_a, p.ssm_a, p.dt_bias, p.norm_w, d_S,
                                                 d_normed, d.k_heads, d.v_heads, d.key_dim(),
                                                 order == VHeadOrder::Gguf ? 1 : 0, d.eps);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
