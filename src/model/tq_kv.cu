#include "model/tq_kv.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <mutex>

namespace dara::model {
namespace {

constexpr int kThreads = 256;
__constant__ float c_cent[16];   // centroids, ascending
__constant__ float c_bnd[15];    // decision boundaries between neighbouring centroids

float g_cent[16];
float g_bnd[15];
double g_mse = 0;
std::once_flag g_once;

// Lloyd-Max for the standard normal: alternate boundaries = midpoints and centroids = conditional means (closed form via phi / Phi).
void build_codebook() {
    const double kPi = 3.14159265358979323846;
    auto phi = [&](double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi); };
    auto Phi = [&](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    double c[16], b[15];
    for (int k = 0; k < 16; ++k) c[k] = -2.4 + 4.8 * (k + 0.5) / 16.0;  // start spread over the bulk of the distribution
    for (int it = 0; it < 2000; ++it) {
        for (int k = 0; k < 15; ++k) b[k] = 0.5 * (c[k] + c[k + 1]);
        for (int k = 0; k < 16; ++k) {
            const double lo = k == 0 ? -INFINITY : b[k - 1], hi = k == 15 ? INFINITY : b[k];
            const double plo = k == 0 ? 0.0 : phi(lo), phi_hi = k == 15 ? 0.0 : phi(hi);
            const double mass = (k == 15 ? 1.0 : Phi(hi)) - (k == 0 ? 0.0 : Phi(lo));
            c[k] = (plo - phi_hi) / mass;
        }
    }
    for (int k = 0; k < 15; ++k) b[k] = 0.5 * (c[k] + c[k + 1]);
    // MSE of the quantiser on N(0,1), by numerical integration over [-8, 8]
    double mse = 0;
    const int N = 400000;
    for (int i = 0; i < N; ++i) {
        const double x = -8.0 + 16.0 * (i + 0.5) / N;
        int k = 0;
        while (k < 15 && x > b[k]) ++k;
        mse += (x - c[k]) * (x - c[k]) * phi(x) * (16.0 / N);
    }
    for (int k = 0; k < 16; ++k) g_cent[k] = static_cast<float>(c[k]);
    for (int k = 0; k < 15; ++k) g_bnd[k] = static_cast<float>(b[k]);
    g_mse = mse;
}

// Fixed +-1 diagonal D: a hash of the coordinate index (the same for every head, layer and run).
__device__ __forceinline__ float sign_of(int i) {
    uint32_t h = static_cast<uint32_t>(i) * 0x9E3779B1u + 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h & 1u) ? -1.0f : 1.0f;
}

// In-place unnormalised Walsh-Hadamard transform of s[0..d) by the whole block (d a power of two).
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

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < blockDim.x / 32; ++i) t += red[i];
    __syncthreads();
    return t;
}

// grid (n, kv_heads): quantise one head vector.
__global__ void __launch_bounds__(kThreads)
tq_append_kernel(const float* __restrict__ x, const int64_t* __restrict__ pos, int kv_heads, int d, uint8_t* __restrict__ codes, float* __restrict__ norms) {
    extern __shared__ float s[];  // d floats
    __shared__ float red[32];
    __shared__ uint8_t sc[512];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* xr = x + (static_cast<size_t>(t) * kv_heads + h) * d;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) { const float v = xr[i]; s[i] = v; ss = fmaf(v, v, ss); }
    const float n = sqrtf(block_reduce_sum(ss, red));
    const float inv = n > 1e-30f ? 1.0f / n : 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) s[i] = s[i] * inv * sign_of(i);
    __syncthreads();
    fwht(s, d);  // z = H D u ~ N(0,1) per coordinate
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float z = s[i];
        int k = 0;
#pragma unroll
        for (int b = 0; b < 15; ++b) k += z > c_bnd[b];
        sc[i] = static_cast<uint8_t>(k);
    }
    __syncthreads();
    const size_t p = static_cast<size_t>(pos[t]);
    uint8_t* out = codes + (p * kv_heads + h) * (d / 2);
    for (int b = threadIdx.x; b < d / 2; b += blockDim.x) out[b] = static_cast<uint8_t>(sc[2 * b] | (sc[2 * b + 1] << 4));
    if (threadIdx.x == 0) norms[p * kv_heads + h] = n;
}

// grid (heads, n): one block per (query head, query token); scores live in dynamic shared memory.
__global__ void __launch_bounds__(kThreads)
tq_attention_kernel(const float* __restrict__ q, const float* __restrict__ gate, const uint8_t* __restrict__ Kc, const float* __restrict__ Kn,
                    const uint8_t* __restrict__ Vc, const float* __restrict__ Vn, const int64_t* __restrict__ pos, int heads, int kv_heads, int d,
                    int window, float scale, float* __restrict__ out) {
    extern __shared__ float scores[];            // smem_positions floats, then nothing else dynamic
    __shared__ float sq[512];                    // rotated query, later the rotated output accumulator
    __shared__ float s_c[16];
    __shared__ float red[32];
    __shared__ float s_max, s_sum;
    const int h = blockIdx.x, t = blockIdx.y;
    const int kvh = h / (heads / kv_heads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    const int p = static_cast<int>(pos[t]);
    const int start = window > 0 ? max(0, p - window + 1) : 0;
    const int n = p - start + 1;
    const int half = d / 2;

    if (threadIdx.x < 16) s_c[threadIdx.x] = c_cent[threadIdx.x];
    const float* qh = q + (static_cast<size_t>(t) * heads + h) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sq[i] = qh[i] * sign_of(i);
    __syncthreads();
    fwht(sq, d);  // sq = H D q

    // each lane owns coordinates [8*lane + 256*g, +8) of every 256-wide group g (d/256 groups; d == 64..128 uses a partial group)
    const int groups = d >= 256 ? d / 256 : 1;
    const bool lane_active = d >= 256 || lane < d / 8;
    float qr[2][8];
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int j = 0; j < 8; ++j) qr[g][j] = (g < groups && lane_active) ? sq[g * 256 + lane * 8 + j] : 0.0f;
    __syncthreads();

    for (int j = warp; j < n; j += nwarps) {
        const size_t row = (static_cast<size_t>(start + j) * kv_heads + kvh);
        const uint8_t* kc = Kc + row * half;
        float acc = 0.0f;
        if (lane_active) {
            for (int g = 0; g < groups; ++g) {
                const uint32_t w = *reinterpret_cast<const uint32_t*>(kc + g * 128 + lane * 4);
#pragma unroll
                for (int b = 0; b < 8; ++b) acc = fmaf(qr[g][b], s_c[(w >> (4 * b)) & 15u], acc);
            }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) scores[j] = acc * (Kn[row] / static_cast<float>(d)) * scale;
    }
    __syncthreads();

    float m = -3.402823466e38f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, scores[j]);
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0) { float mm = red[0]; for (int w = 1; w < nwarps; ++w) mm = fmaxf(mm, red[w]); s_max = mm; }
    __syncthreads();
    float sum = 0.0f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) { const float e = expf(scores[j] - s_max); scores[j] = e; sum += e; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, off);
    __syncthreads();
    if (lane == 0) red[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) { float ss = 0.0f; for (int w = 0; w < nwarps; ++w) ss += red[w]; s_sum = ss; }
    __syncthreads();
    const float inv = 1.0f / s_sum;

    // rotated-domain value accumulation: w_i = sum_j p_j n_j c[code_j,i]
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; ++j) {
            const size_t row = (static_cast<size_t>(start + j) * kv_heads + kvh);
            const uint8_t byte = Vc[row * half + (i >> 1)];
            acc = fmaf(scores[j] * Vn[row], s_c[(i & 1) ? (byte >> 4) : (byte & 15)], acc);
        }
        sq[i] = acc;
    }
    __syncthreads();
    fwht(sq, d);  // (H w)
    float* oh = out + (static_cast<size_t>(t) * heads + h) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float o = sign_of(i) * sq[i] * (inv / static_cast<float>(d));
        if (gate) { const float g = gate[(static_cast<size_t>(t) * heads + h) * d + i]; o *= 1.0f / (1.0f + expf(-g)); }
        oh[i] = o;
    }
}

}  // namespace

bool tq_supported_head_dim(int hd) { return hd >= 64 && hd <= 512 && (hd & (hd - 1)) == 0; }

int tq_init() {
    std::call_once(g_once, build_codebook);
    cudaError_t e = cudaMemcpyToSymbol(c_cent, g_cent, sizeof g_cent);
    if (e != cudaSuccess) { std::fprintf(stderr, "tq_init: centroids upload failed: %s\n", cudaGetErrorString(e)); return static_cast<int>(e); }
    e = cudaMemcpyToSymbol(c_bnd, g_bnd, sizeof g_bnd);
    if (e != cudaSuccess) std::fprintf(stderr, "tq_init: boundaries upload failed: %s\n", cudaGetErrorString(e));
    return static_cast<int>(e);
}

const float* tq_centroids() { std::call_once(g_once, build_codebook); return g_cent; }
double tq_codebook_mse() { std::call_once(g_once, build_codebook); return g_mse; }
const float* tq_boundaries() { std::call_once(g_once, build_codebook); return g_bnd; }

int tq_append_cuda(const float* x, const int64_t* pos, int n, int kv_heads, int d, uint8_t* codes, float* norms, void* stream) {
    if (!x || !pos || !codes || !norms || n < 0 || kv_heads <= 0 || !tq_supported_head_dim(d)) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    tq_append_kernel<<<dim3(n, kv_heads), kThreads, static_cast<size_t>(d) * sizeof(float), static_cast<cudaStream_t>(stream)>>>(x, pos, kv_heads, d, codes, norms);
    return static_cast<int>(cudaGetLastError());
}

int tq_attention_cuda(const float* q, const float* gate, const uint8_t* Kc, const float* Kn, const uint8_t* Vc, const float* Vn, const int64_t* pos, int n,
                      int heads, int kv_heads, int d, int window, float scale, int smem_positions, float* out, void* stream) {
    if (!q || !Kc || !Kn || !Vc || !Vn || !pos || !out || n < 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || !tq_supported_head_dim(d) ||
        window < 0 || smem_positions <= 0 || smem_positions > kTqMaxPositions)
        return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0) return 0;
    tq_attention_kernel<<<dim3(heads, n), kThreads, static_cast<size_t>(smem_positions) * sizeof(float), static_cast<cudaStream_t>(stream)>>>(
        q, gate, Kc, Kn, Vc, Vn, pos, heads, kv_heads, d, window, scale, out);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace dara::model
