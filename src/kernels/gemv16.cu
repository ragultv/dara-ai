// Wide-load decode GEMV for Q4_K and Q5_K.
//
// Measured on the GTX 1650 (tests/bw_probe.cu): a warp-per-row streaming read reaches ~105-128 GB/s with 4-byte loads per lane but ~182 GB/s (97% of the
// 188 GB/s theoretical peak) with 16-byte loads. The original GEMV kernels use 4-byte loads, so they were capped far below the memory system. Here a lane
// loads 16 bytes of quantised weights at a time: 8 lanes cover one 256-weight block (its header plus the 8 sixteen-byte chunks of quant data) and the warp's
// 32 lanes cover 4 consecutive blocks of the row, so a warp reads contiguous memory 16 bytes per lane.
//
// Weight decoding is the same arithmetic as the original kernels (fmaf(d*sc, q, -dmin*m), exact conversions); only the order of accumulation differs
// (4 independent chains per lane, then a lane reduction), so results agree to ~1e-6 relative. DARA_GEMV_REF=1 selects the original kernels.
#include <climits>
#include <cstdlib>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/quant.h"

namespace dara::kernels {

int gemv_q4k_cuda_ref(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols, void* stream);
int gemv_q5k_cuda_ref(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols, void* stream);

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr float k2p23 = 8388608.0f;

__device__ __forceinline__ float small2f(uint32_t v) { return __uint_as_float(0x4B000000u | v) - k2p23; }
template <int K>
__device__ __forceinline__ float nib(uint32_t packed) { return __uint_as_float(__byte_perm(packed, 0x4B000000u, 0x7650 | K)) - k2p23; }

// One lane, one 16-byte chunk of quant data: 16 low-nibble weights (sub-block 2g) and 16 high-nibble weights (sub-block 2g+1), 32 FMAs on 4 chains.
template <bool Q5>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
gemv16_kernel(const uint8_t* __restrict__ W, size_t row_bytes, const float* __restrict__ xg, float* __restrict__ y, size_t rows, int nblk) {
    extern __shared__ float xs[];   // the activation vector, staged once per thread block (every row of the block uses the same x)
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    {
        const float4* src = reinterpret_cast<const float4*>(xg);
        float4* dst = reinterpret_cast<float4*>(xs);
        for (int i = threadIdx.x; i < nblk * 64; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    const float* x = xs;
    const size_t row = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const uint8_t* rp = W + row * row_bytes;
    constexpr int kBlockBytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    constexpr int kQsOff = Q5 ? 48 : 16;

    const int bi = lane >> 3;      // which of the warp's 4 blocks
    const int q = lane & 7;        // which 16-byte chunk of the block's 128 quant bytes
    const int g = q >> 1;          // quant group (32 bytes): sub-blocks 2g (low nibbles) and 2g+1 (high nibbles)
    const int half = q & 1;        // first or second 16 bytes of the group
    const int sh0 = 8 * ((2 * g) & 3), sh1 = 8 * ((2 * g + 1) & 3);
    const bool hi_pair = g >= 2;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll 2
    for (int b = bi; b < nblk; b += 4) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * kBlockBytes;
        const uint4 hdr = __ldg(reinterpret_cast<const uint4*>(blk));                       // d, dmin, scales[12]
        const uint4 qs = __ldg(reinterpret_cast<const uint4*>(blk + kQsOff + 16 * q));      // 16 quant bytes
        uint4 qh = make_uint4(0, 0, 0, 0);
        if (Q5) qh = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * half));           // 5th bits of the same 16 positions
        const float* xb = x + static_cast<size_t>(b) * 256 + 64 * g + 16 * half;
        const float4 xl0 = *reinterpret_cast<const float4*>(xb), xl1 = *reinterpret_cast<const float4*>(xb + 4);
        const float4 xl2 = *reinterpret_cast<const float4*>(xb + 8), xl3 = *reinterpret_cast<const float4*>(xb + 12);
        const float4 xh0 = *reinterpret_cast<const float4*>(xb + 32), xh1 = *reinterpret_cast<const float4*>(xb + 36);
        const float4 xh2 = *reinterpret_cast<const float4*>(xb + 40), xh3 = *reinterpret_cast<const float4*>(xb + 44);

        const float d = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x & 0xffffu)));
        const float dmin = __half2float(__ushort_as_half(static_cast<unsigned short>(hdr.x >> 16)));
        const uint32_t a0 = (hdr.y >> sh0) & 0xffu, a1 = (hdr.z >> sh0) & 0xffu, a2 = (hdr.w >> sh0) & 0xffu;
        const uint32_t b0 = (hdr.y >> sh1) & 0xffu, b1 = (hdr.z >> sh1) & 0xffu, b2 = (hdr.w >> sh1) & 0xffu;
        const uint32_t s0 = hi_pair ? ((a2 & 0xFu) | ((a0 >> 6) << 4)) : (a0 & 63u);
        const uint32_t m0 = hi_pair ? ((a2 >> 4) | ((a1 >> 6) << 4)) : (a1 & 63u);
        const uint32_t s1 = hi_pair ? ((b2 & 0xFu) | ((b0 >> 6) << 4)) : (b0 & 63u);
        const uint32_t m1v = hi_pair ? ((b2 >> 4) | ((b1 >> 6) << 4)) : (b1 & 63u);
        const float d1 = d * small2f(s0), m1 = dmin * small2f(m0);
        const float d2 = d * small2f(s1), m2 = dmin * small2f(m1v);

        const uint32_t w4[4] = {qs.x, qs.y, qs.z, qs.w};
        const uint32_t h4[4] = {qh.x, qh.y, qh.z, qh.w};
        const float4 xlo[4] = {xl0, xl1, xl2, xl3}, xhi[4] = {xh0, xh1, xh2, xh3};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint32_t lo = w4[i] & 0x0F0F0F0Fu, hi = (w4[i] >> 4) & 0x0F0F0F0Fu;
            if (Q5) {  // bit (2g) / (2g+1) of each qh byte is the 5th bit of the low / high weight
                lo |= ((h4[i] >> (2 * g)) & 0x01010101u) << 4;
                hi |= ((h4[i] >> (2 * g + 1)) & 0x01010101u) << 4;
            }
            float a = acc[i];
            a = fmaf(fmaf(d1, nib<0>(lo), -m1), xlo[i].x, a);
            a = fmaf(fmaf(d2, nib<0>(hi), -m2), xhi[i].x, a);
            a = fmaf(fmaf(d1, nib<1>(lo), -m1), xlo[i].y, a);
            a = fmaf(fmaf(d2, nib<1>(hi), -m2), xhi[i].y, a);
            a = fmaf(fmaf(d1, nib<2>(lo), -m1), xlo[i].z, a);
            a = fmaf(fmaf(d2, nib<2>(hi), -m2), xhi[i].z, a);
            a = fmaf(fmaf(d1, nib<3>(lo), -m1), xlo[i].w, a);
            a = fmaf(fmaf(d2, nib<3>(hi), -m2), xhi[i].w, a);
            acc[i] = a;
        }
    }
    float total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(0xffffffffu, total, off);
    if (lane == 0) y[row] = total;
}

// The wide-load FP32 kernel above measured SLOWER than the original (0.4-0.6x: ~47 vs ~88 GB/s): with 16-byte loads the kernel becomes instruction-bound, not
// bandwidth-bound (tests/bw_probe.cu shows the loads alone reach 182 GB/s). It is kept as an experiment: the ORIGINAL kernels are the default and
// DARA_GEMV16=1 selects this one. (The fast path for decode is the int8 dp4a kernel in gemv_q8.cu.)
bool use_reference() {
    static const bool r = [] { const char* e = std::getenv("DARA_GEMV16"); return !(e && e[0] == '1'); }();
    return r;
}

template <bool Q5>
int launch16(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols, void* stream) {
    constexpr size_t block_bytes = Q5 ? kQ5KBlockBytes : kQ4KBlockBytes;
    auto s = static_cast<cudaStream_t>(stream);
    if (rows == 0) return 0;
    if (!dW || !dx || !dy || cols == 0 || cols % 256 != 0 || cols > static_cast<size_t>(INT_MAX) || ld_blocks < cols / 256) return static_cast<int>(cudaErrorInvalidValue);
    if (reinterpret_cast<uintptr_t>(dW) % 16 != 0 || reinterpret_cast<uintptr_t>(dx) % 16 != 0) return static_cast<int>(cudaErrorMisalignedAddress);
    const unsigned blocks = static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const size_t smem = static_cast<size_t>(cols) * sizeof(float);
    if (smem > 48 * 1024) return static_cast<int>(cudaErrorInvalidValue);   // K <= 12288; wider rows need a different kernel
    gemv16_kernel<Q5><<<blocks, kWarpsPerBlock * 32, smem, s>>>(static_cast<const uint8_t*>(dW), ld_blocks * block_bytes, dx, dy, rows, static_cast<int>(cols / 256));
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

int gemv_q4k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols, void* stream) {
    return use_reference() ? gemv_q4k_cuda_ref(dW, ld_blocks, dx, dy, rows, cols, stream) : launch16<false>(dW, ld_blocks, dx, dy, rows, cols, stream);
}
int gemv_q5k_cuda(const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows, size_t cols, void* stream) {
    return use_reference() ? gemv_q5k_cuda_ref(dW, ld_blocks, dx, dy, rows, cols, stream) : launch16<true>(dW, ld_blocks, dx, dy, rows, cols, stream);
}

}  // namespace dara::kernels
