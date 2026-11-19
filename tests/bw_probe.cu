// Memory-bandwidth probe: how fast can this GPU stream weights with the access patterns a GEMV uses? Reads a large buffer organised as rows (like a quantised
// matrix: one warp per row) with different per-lane load widths and unroll depths, and reports GB/s. The ceiling for a bandwidth-bound decode kernel.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

template <typename V, int UNROLL>
__global__ void __launch_bounds__(256) read_rows(const V* __restrict__ p, size_t row_vecs, size_t rows, unsigned* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (row >= rows) return;
    const V* r = p + row * row_vecs;
    unsigned acc = 0;
#pragma unroll UNROLL
    for (size_t i = lane; i < row_vecs; i += 32) {
        const V v = __ldg(r + i);
        if constexpr (sizeof(V) == 4) acc ^= *reinterpret_cast<const unsigned*>(&v);
        else if constexpr (sizeof(V) == 8) { const uint2 u = *reinterpret_cast<const uint2*>(&v); acc ^= u.x ^ u.y; }
        else { const uint4 u = *reinterpret_cast<const uint4*>(&v); acc ^= u.x ^ u.y ^ u.z ^ u.w; }
    }
    for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0 && acc == 0x12345678u) out[0] = acc;  // keeps the loads alive
}

template <typename V, int UNROLL>
float run(const void* buf, size_t bytes, size_t row_bytes, const char* name) {
    const size_t rows = bytes / row_bytes, row_vecs = row_bytes / sizeof(V);
    unsigned* out; cudaMalloc(&out, 4);
    const unsigned blocks = static_cast<unsigned>((rows + 7) / 8);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 0;
    for (int rep = 0; rep < 6; ++rep) {
        cudaEventRecord(a);
        for (int it = 0; it < 5; ++it) read_rows<V, UNROLL><<<blocks, 256>>>(static_cast<const V*>(buf), row_vecs, rows, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        best = std::max(best, static_cast<float>(5.0 * rows * row_bytes / (ms / 1000.0) / 1e9));
    }
    std::printf("  %-34s row %5zu B: %6.1f GB/s\n", name, row_bytes, best);
    cudaFree(out);
    return best;
}

// The exact weight-load pattern of the wide-load Q4_K kernel (header + 16-byte quant chunk per lane, 4 blocks per warp step), with no arithmetic beyond a xor.
template <int UNROLL>
__global__ void __launch_bounds__(256) read_q4k_pattern(const uint8_t* __restrict__ W, size_t row_bytes, size_t rows, int nblk, unsigned* out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * 8 + warp;
    if (row >= rows) return;
    const uint8_t* rp = W + row * row_bytes;
    const int bi = lane >> 3, q = lane & 7;
    unsigned acc = 0;
#pragma unroll UNROLL
    for (int b = bi; b < nblk; b += 4) {
        const uint8_t* blk = rp + static_cast<size_t>(b) * 144;
        const uint4 h = __ldg(reinterpret_cast<const uint4*>(blk));
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(blk + 16 + 16 * q));
        acc ^= h.x ^ h.y ^ v.x ^ v.y ^ v.z ^ v.w;
    }
    for (int off = 16; off > 0; off >>= 1) acc ^= __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0 && acc == 0x12345678u) out[0] = acc;
}

template <int UNROLL>
float run_q4k_pattern(const void* buf, size_t bytes, int nblk) {
    const size_t row_bytes = static_cast<size_t>(nblk) * 144, rows = bytes / row_bytes;
    unsigned* out; cudaMalloc(&out, 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 0;
    for (int rep = 0; rep < 6; ++rep) {
        cudaEventRecord(a);
        for (int it = 0; it < 5; ++it) read_q4k_pattern<UNROLL><<<static_cast<unsigned>((rows + 7) / 8), 256>>>(static_cast<const uint8_t*>(buf), row_bytes, rows, nblk, out);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        best = std::max(best, static_cast<float>(5.0 * rows * row_bytes / (ms / 1000.0) / 1e9));
    }
    std::printf("  Q4_K load pattern, %2d blocks/row, unroll %d: %6.1f GB/s\n", nblk, UNROLL, best);
    cudaFree(out);
    return best;
}

int main() {
    const size_t bytes = 512ull << 20;
    void* buf;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMemset(buf, 1, bytes));
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int mem_khz = 0, bus_bits = 0;
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, 0);
    const double peak = 2.0 * mem_khz * 1e3 * bus_bits / 8 / 1e9;
    std::printf("%s: %d SMs, memory %d MHz x %d-bit (theoretical peak about %.0f GB/s)\n", prop.name, prop.multiProcessorCount, mem_khz / 1000, bus_bits, peak);
    float best_stream = 0, best_q4k = 0;
    std::printf("streaming a 512 MiB buffer (far larger than L2), one warp per row:\n");
    for (size_t row : {1440ul, 11520ul}) {   // 10 and 80 Q4_K blocks per row (2560 / 20480 columns)
        run<unsigned, 1>(buf, bytes, row, "4 B per lane, no unroll");
        run<unsigned, 4>(buf, bytes, row, "4 B per lane, unroll 4");
        run<uint2, 4>(buf, bytes, row, "8 B per lane, unroll 4");
        run<uint4, 1>(buf, bytes, row, "16 B per lane, no unroll");
        best_stream = std::max(best_stream, run<uint4, 4>(buf, bytes, row, "16 B per lane, unroll 4"));
    }
    std::printf("the wide-load Q4_K kernel's exact weight-load pattern, no arithmetic:\n");
    for (int nblk : {10, 36})
        best_q4k = std::max({best_q4k, run_q4k_pattern<1>(buf, bytes, nblk), run_q4k_pattern<2>(buf, bytes, nblk), run_q4k_pattern<4>(buf, bytes, nblk)});
    // one line for scripts (bench/decode_quick.py)
    std::printf("PROBE gpu=\"%s\" sm=%d%d theoretical_gbs=%.1f stream_gbs=%.1f q4k_pattern_gbs=%.1f\n", prop.name, prop.major, prop.minor, peak, best_stream, best_q4k);
    cudaFree(buf);
    return 0;
}
