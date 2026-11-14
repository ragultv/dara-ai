// CPU decode kernel benchmark: GB/s of CpuDecoder::matvec per quantisation format on matrices far larger than the caches (the regime of host-resident
// layers), so kernel changes can be measured without the noise of whole-model runs. Data is random bytes (timing only; values are meaningless).
// Usage: cpu_kernel_bench [threads]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "model/cpu_decode.h"

using namespace dara::model;
using dara::kernels::QuantType;

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 0;
    CpuDecoder dec(threads);
    std::printf("threads %d\n", dec.threads());
    struct Case { QuantType t; const char* name; size_t block_bytes; };
    const Case cases[] = {{QuantType::Q4_K, "Q4_K", 144}, {QuantType::Q5_K, "Q5_K", 176}, {QuantType::Q6_K, "Q6_K", 210}};
    const size_t rows = 12288, cols = 4096;   // a 9B MLP matrix
    std::mt19937 rng(1);
    for (const Case& c : cases) {
        const size_t row_bytes = cols / 256 * c.block_bytes;
        const int copies = static_cast<int>((256ull << 20) / (rows * row_bytes)) + 1;   // > 256 MB in total: never cache-resident
        std::vector<std::vector<uint8_t>> mats(copies, std::vector<uint8_t>(rows * row_bytes));
        for (auto& m : mats) for (auto& b : m) b = static_cast<uint8_t>(rng());
        // keep the fp16 block scales finite so the arithmetic is representative
        for (auto& m : mats)
            for (size_t r = 0; r < rows; ++r)
                for (size_t b = 0; b < cols / 256; ++b) {
                    uint8_t* blk = m.data() + r * row_bytes + b * c.block_bytes;
                    uint8_t* dp = c.t == QuantType::Q6_K ? blk + 208 : blk;
                    dp[0] = 0x00; dp[1] = 0x1c;                       // 0.0039 as fp16
                    if (c.t != QuantType::Q6_K) { dp[2] = 0x00; dp[3] = 0x1c; }
                }
        std::vector<float> x(cols), y(rows);
        for (auto& v : x) v = std::normal_distribution<float>(0.f, 1.f)(rng);
        double best = 0;
        for (int rep = 0; rep < 5; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < copies; ++i) {
                HostMatrix hm{mats[i].data(), c.t, rows, cols, row_bytes};
                dec.matvec(hm, x.data(), y.data());
            }
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::max(best, copies * rows * row_bytes / s / 1e9);
        }
        std::printf("  %s %zux%zu x%d: %.1f GB/s\n", c.name, rows, cols, copies, best);
    }
    return 0;
}
