// CPU read-bandwidth probe: how fast can N threads stream a large buffer, pinned (cudaHostAlloc, where Dara keeps host-resident layers) vs pageable?
// The ceiling for CPU decode of host-resident layers (src/model/cpu_decode.cpp).
#include <cuda_runtime.h>
#include <immintrin.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double read_gbs(const uint8_t* p, size_t bytes, int threads) {
    std::vector<std::thread> th;
    std::vector<long long> sink(threads * 16, 0);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&, t] {
            const size_t per = bytes / threads, a = per * t, b = a + per;
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = a; i + 32 <= b; i += 32) acc = _mm256_add_epi8(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + i)));
            sink[t * 16] = _mm256_extract_epi64(acc, 0);
        });
    for (auto& x : th) x.join();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return bytes / s / 1e9;
}

int main() {
    const size_t bytes = 512ull << 20;
    uint8_t* pinned = nullptr;
    if (cudaHostAlloc(reinterpret_cast<void**>(&pinned), bytes, cudaHostAllocDefault) != cudaSuccess) { std::puts("cudaHostAlloc failed"); return 1; }
    uint8_t* pageable = static_cast<uint8_t*>(std::malloc(bytes));
    if (!pageable) { std::puts("malloc failed"); return 1; }
    std::memset(pinned, 1, bytes);
    std::memset(pageable, 1, bytes);
    for (int threads : {1, 2, 4, 6, 8, 12}) {
        double bp = 0, bm = 0;
        for (int r = 0; r < 3; ++r) { bp = std::max(bp, read_gbs(pinned, bytes, threads)); bm = std::max(bm, read_gbs(pageable, bytes, threads)); }
        std::printf("threads %2d: pinned %5.1f GB/s | pageable %5.1f GB/s\n", threads, bp, bm);
    }
    cudaFreeHost(pinned);
    std::free(pageable);
    return 0;
}
