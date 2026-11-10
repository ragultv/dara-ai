// int8-activation decode GEMV (dp4a) with 16-byte weight loads, for Q4_K and Q5_K.
//
// Why: on the GTX 1650 the FP32 dequantise-and-FMA kernels cost ~4-6 ALU instructions per weight, which caps them near ~90 GB/s although a streaming read reaches
// ~182 GB/s (tests/bw_probe.cu: the exact weight-load pattern of this kernel, without arithmetic, reaches 182 GB/s). Quantising the ACTIVATION vector to int8
// (one scale per 32 values) lets __dp4a do four weight x activation multiply-adds per instruction directly on the unpacked nibbles, and the Q4_K / Q5_K
// arithmetic factorises per 32-weight sub-block j (sub-blocks line up with the activation blocks):
//       sum_i w_i x_i  =  sum_j  dx_j * ( d*sc_j * sum_i q_i a_i  -  dmin*m_j * sum_i a_i )          w_i = d*sc_j*q_i - dmin*m_j,  x_i ~= dx_j * a_i
// The sums of the int8 activations (the dmin term) are computed once per vector, not per row. llama.cpp's mmvq kernels use the same factorisation with 4- or
// 8-byte loads and a second dp4a for the activation sum; this is LOSSY (8-bit activations) and is an opt-in fast mode gated on measured quality.
//
// Activation layout per vector (K a multiple of 256):  q[K] int8 | d[K/32] float scale per 32 values | s[K/16] int32 sum of the int8 values per 16 values.
#include <climits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/q6k.h"
#include "kernels/quant.h"

namespace dara::kernels {
namespace {

constexpr int kWarpsPerBlock = 8;
constexpr float k2p23 = 8388608.0f;
__device__ __forceinline__ float small2f(uint32_t v) { return __uint_as_float(0x4B000000u | v) - k2p23; }

// One warp per 32-value block of one vector: scale = max|x| / 127, q = round(x / scale), plus the int sums of each 16 values.
__global__ void quantize_act_kernel(const float* __restrict__ x, size_t ldx, int K, int8_t* __restrict__ q, float* __restrict__ d, int* __restrict__ s, int T) {
    const int lane = threadIdx.x & 31;
    const int blk = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);   // 32-value block index within the vector
    const int t = blockIdx.y;
    if (blk >= K / 32) return;
    const float v = x[static_cast<size_t>(t) * ldx + blk * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float scale = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int a = __float2int_rn(v * inv);
    q[static_cast<size_t>(t) * K + blk * 32 + lane] = static_cast<int8_t>(a);
    int sum = a;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);   // sum over the 16 lanes of this half
    if ((lane & 15) == 0) s[static_cast<size_t>(t) * (K / 16) + blk * 2 + (lane >> 4)] = sum;
    if (lane == 0) d[static_cast<size_t>(t) * (K / 32) + blk] = scale;
}

// y[row] = sum over blocks; lane (bi, q): block b = bi + 4*iter, 16-byte chunk q of the block's 128 quant bytes (group g = q/2, half = q%2).
template <bool Q5>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q_q8_gemv_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa,
                 float* __restrict__ y, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    constexpr int kBlockBytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    constexpr int kQsOff = Q5 ? 48 : 16;
    const uint8_t* rp = W + row * row_bytes;
    const int bi = lane >> 3, q = lane & 7, g = q >> 1, half = q & 1;
    const int sh0 = 8 * ((2 * g) & 3), sh1 = 8 * ((2 * g + 1) & 3);
    const bool hi_pair = g >= 2;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = bi; b < nblk; b += 4) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kBlockBytes;
        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));
        const uint4 qs = __ldg(reinterpret_cast<const uint4*>(blk + kQsOff + 16 * q));
        uint4 qh = make_uint4(0, 0, 0, 0);
        if (Q5) qh = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * half));   // 5th bits of the same 16 positions
        const int8_t* ap = qa + static_cast<size_t>(b) * 256 + 64 * g + 16 * half;
        const uint4 alo = *reinterpret_cast<const uint4*>(ap), ahi = *reinterpret_cast<const uint4*>(ap + 32);
        const float dlo = da[b * 8 + 2 * g], dhi = da[b * 8 + 2 * g + 1];
        const int slo = sa[b * 16 + 4 * g + half], shi = sa[b * 16 + 4 * g + 2 + half];

        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
        const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
        const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
        const uint32_t s0 = hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
        const uint32_t m0 = hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
        const uint32_t s1 = hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
        const uint32_t m1 = hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);

        const uint32_t w[4] = {qs.x, qs.y, qs.z, qs.w};
        const uint32_t h4[4] = {qh.x, qh.y, qh.z, qh.w};
        const uint32_t al[4] = {alo.x, alo.y, alo.z, alo.w}, ah[4] = {ahi.x, ahi.y, ahi.z, ahi.w};
        int plo = 0, phi = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t lo = w[i] & 0x0F0F0F0Fu, hi = (w[i] >> 4) & 0x0F0F0F0Fu;
            if (Q5) {  // bit 2g / 2g+1 of each qh byte is the 5th bit of the low / high weight at that position
                lo |= ((h4[i] >> (2 * g)) & 0x01010101u) << 4;
                hi |= ((h4[i] >> (2 * g + 1)) & 0x01010101u) << 4;
            }
            plo = __dp4a(static_cast<int>(lo), static_cast<int>(al[i]), plo);
            phi = __dp4a(static_cast<int>(hi), static_cast<int>(ah[i]), phi);
        }
        acc = fmaf(dlo, fmaf(d * small2f(s0), static_cast<float>(plo), -(dmin * small2f(m0)) * static_cast<float>(slo)), acc);
        acc = fmaf(dhi, fmaf(d * small2f(s1), static_cast<float>(phi), -(dmin * small2f(m1)) * static_cast<float>(shi)), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

template <bool Q5>
int launch_q8(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int t, float* dy, size_t rows, void* stream) {
    constexpr size_t block_bytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    if (rows == 0) return 0;
    if (!dW || !scratch || !dy || K % 256 != 0 || ld_blocks < K / 256 || t < 0 || t >= T) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base) + static_cast<size_t>(t) * K;
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K) + static_cast<size_t>(t) * (K / 32);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float)) + static_cast<size_t>(t) * (K / 16);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    q_q8_gemv_kernel<Q5><<<blocks, kWarpsPerBlock * 32, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * block_bytes, q, d, s, dy, rows,
                                                                                            static_cast<int>(K / 256));
    return static_cast<int>(cudaGetLastError());
}

// ---- Q6_K: 210-byte blocks are only 2-byte aligned ------------------------------------------------------------------------------------
// 16 bytes at a 2-byte-aligned address from five 4-byte-aligned loads and funnel shifts (branch-free: the shift is 0 or 16 bits), so the weights need no
// repacking and keep their GGUF layout (the FP32 kernels and the host staging read the same bytes).
__device__ __forceinline__ uint4 ld16_a2(const uint8_t* p) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t* w = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
    const unsigned sh = static_cast<unsigned>(a & 2) * 8;
    const uint32_t w0 = __ldg(w), w1 = __ldg(w + 1), w2 = __ldg(w + 2), w3 = __ldg(w + 3), w4 = __ldg(w + 4);
    return make_uint4(__funnelshift_r(w0, w1, sh), __funnelshift_r(w1, w2, sh), __funnelshift_r(w2, w3, sh), __funnelshift_r(w3, w4, sh));
}

// Lane mapping: 4 lanes per block (half = which 128-weight half, l0 = 0 or 16 = which 16 positions of the 32), 8 blocks per warp step. A lane reads 16 bytes of
// ql at l0, 16 at l0+32 and 16 of qh, i.e. the 64 weights at positions l0.., +32, +64, +96 of its half: four groups k of 16 weights, each with one int8 scale
// and lying inside one 32-value activation block, so  sum = d * sum_k sc_k * dx_k * (dp4a(q, a) - 32 * sum16(a)).
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
q6k_q8_gemv_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const int8_t* __restrict__ qa, const float* __restrict__ da, const int* __restrict__ sa,
                   float* __restrict__ y, size_t rows, int nblk) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint8_t* rp = W + row * row_bytes;
    const int bi = lane >> 2, sub = lane & 3, half = sub >> 1, part = sub & 1, l0 = 16 * part;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = bi; b < nblk; b += 8) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kQ6KBlockBytes;
        const uint4 A = ld16_a2(blk + 64 * half + l0);            // ql[l0 .. l0+15]       (low nibbles: k=0, high nibbles: k=2)
        const uint4 B = ld16_a2(blk + 64 * half + 32 + l0);       // ql[l0+32 .. l0+47]    (low nibbles: k=1, high nibbles: k=3)
        const uint4 Hh = ld16_a2(blk + 128 + 32 * half + l0);     // qh[l0 .. l0+15]       (2 bits per k)
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192 + 8 * half + part);
        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const unsigned short*>(blk + 208))));
        const uint32_t a4[4] = {A.x, A.y, A.z, A.w}, b4[4] = {B.x, B.y, B.z, B.w}, h4[4] = {Hh.x, Hh.y, Hh.z, Hh.w};
        const int ab = b * 8 + 4 * half;                           // activation 32-block of group k is ab + k
        float tmp = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const uint4 av = *reinterpret_cast<const uint4*>(qa + static_cast<size_t>(ab + k) * 32 + l0);
            const uint32_t a[4] = {av.x, av.y, av.z, av.w};
            int p = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t src = (k & 1) ? b4[i] : a4[i];
                const uint32_t nib = ((k & 2) ? (src >> 4) : src) & 0x0F0F0F0Fu;
                const uint32_t q = nib | (((h4[i] >> (2 * k)) & 0x03030303u) << 4);
                p = __dp4a(static_cast<int>(q), static_cast<int>(a[i]), p);
            }
            const int corr = p - 32 * sa[(ab + k) * 2 + part];
            tmp = fmaf(da[ab + k] * static_cast<float>(__ldg(sc + 2 * k)), static_cast<float>(corr), tmp);
        }
        acc = fmaf(d, tmp, acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) y[row] = acc;
}

int launch_q6k_q8(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int t, float* dy, size_t rows, void* stream) {
    if (rows == 0) return 0;
    if (!dW || !scratch || !dy || K % 256 != 0 || ld_blocks < K / 256 || t < 0 || t >= T) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 4 != 0) return static_cast<int>(cudaErrorMisalignedAddress);   // rows then stay 2-byte aligned (210 = 2 * 105)
    auto base = static_cast<const uint8_t*>(scratch);
    const int8_t* q = reinterpret_cast<const int8_t*>(base) + static_cast<size_t>(t) * K;
    const float* d = reinterpret_cast<const float*>(base + static_cast<size_t>(T) * K) + static_cast<size_t>(t) * (K / 32);
    const int* s = reinterpret_cast<const int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float)) + static_cast<size_t>(t) * (K / 16);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    q6k_q8_gemv_kernel<<<blocks, kWarpsPerBlock * 32, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(dW), ld_blocks * kQ6KBlockBytes, q, d, s, dy,
                                                                                           rows, static_cast<int>(K / 256));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

size_t q8act_bytes(size_t K, int T) { return static_cast<size_t>(T) * (K + (K / 32) * sizeof(float) + (K / 16) * sizeof(int)); }

// Quantises T activation vectors (x[t*ldx ..]) into `scratch` (q8act_bytes(K, T) bytes, 16-byte aligned). Layout: all q, then all d, then all s.
int quantize_activations_cuda(const float* dx, size_t ldx, size_t K, int T, void* scratch, void* stream) {
    if (!dx || !scratch || K == 0 || K % 256 != 0 || T <= 0) return static_cast<int>(cudaErrorInvalidValue);
    auto base = static_cast<uint8_t*>(scratch);
    int8_t* q = reinterpret_cast<int8_t*>(base);
    float* d = reinterpret_cast<float*>(base + static_cast<size_t>(T) * K);
    int* s = reinterpret_cast<int*>(base + static_cast<size_t>(T) * K + static_cast<size_t>(T) * (K / 32) * sizeof(float));
    const int blocks_per_vec = static_cast<int>(K / 32);
    quantize_act_kernel<<<dim3((blocks_per_vec + 7) / 8, T), 256, 0, static_cast<cudaStream_t>(stream)>>>(dx, ldx, static_cast<int>(K), q, d, s, T);
    return static_cast<int>(cudaGetLastError());
}

bool q8_supported(QuantType t) { return t == QuantType::Q4_K || t == QuantType::Q5_K || t == QuantType::Q6_K; }

// GEMV of a quantised weight matrix with the t-th of T activation vectors held in a scratch filled by quantize_activations_cuda.
int gemv_q8_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int vec, float* dy, size_t rows, void* stream) {
    switch (t) {
        case QuantType::Q4_K: return launch_q8<false>(dW, ld_blocks, scratch, K, T, vec, dy, rows, stream);
        case QuantType::Q5_K: return launch_q8<true>(dW, ld_blocks, scratch, K, T, vec, dy, rows, stream);
        case QuantType::Q6_K: return launch_q6k_q8(dW, ld_blocks, scratch, K, T, vec, dy, rows, stream);
        default: return static_cast<int>(cudaErrorNotSupported);
    }
}

int gemv_q4k_q8_cuda(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int t, float* dy, size_t rows, void* stream) {
    return launch_q8<false>(dW, ld_blocks, scratch, K, T, t, dy, rows, stream);
}

}  // namespace dara::kernels
