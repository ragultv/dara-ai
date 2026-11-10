// Wide-load decode GEMV vs the original kernels (gemv_q4k_cuda_ref / gemv_q5k_cuda_ref): accuracy and achieved memory bandwidth.
// Usage: gemv16_test
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/quant.h"

namespace dara::kernels {
int gemv_q4k_cuda_ref(const void*, size_t, const float*, float*, size_t, size_t, void*);
int gemv_q5k_cuda_ref(const void*, size_t, const float*, float*, size_t, size_t, void*);
}  // namespace dara::kernels

using namespace dara::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static int failures = 0;

static int run(QuantType qt, const char* name, size_t rows, size_t cols) {
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    const QuantFormat& f = quant_format(qt);
    const size_t blocks = cols / f.block_weights;
    std::vector<uint8_t> w(rows * blocks * f.block_bytes);
    {
        std::vector<float> row(cols);
        for (size_t r = 0; r < rows; ++r) {
            for (auto& v : row) v = nd(rng) * 0.05f;
            f.quantize(row.data(), cols, w.data() + r * blocks * f.block_bytes);
        }
    }
    std::vector<float> x(cols);
    for (auto& v : x) v = nd(rng);
    uint8_t* dW; float *dx, *dy1, *dy2;
    CK(cudaMalloc(&dW, w.size())); CK(cudaMalloc(&dx, cols * 4)); CK(cudaMalloc(&dy1, rows * 4)); CK(cudaMalloc(&dy2, rows * 4));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), cols * 4, cudaMemcpyHostToDevice));
    auto ref = [&] { return qt == QuantType::Q4_K ? gemv_q4k_cuda_ref(dW, blocks, dx, dy1, rows, cols, nullptr) : gemv_q5k_cuda_ref(dW, blocks, dx, dy1, rows, cols, nullptr); };
    auto neu = [&] { return qt == QuantType::Q4_K ? gemv_q4k_cuda(dW, blocks, dx, dy2, rows, cols, nullptr) : gemv_q5k_cuda(dW, blocks, dx, dy2, rows, cols, nullptr); };
    if (ref() || neu()) { std::puts("launch failed"); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<float> a(rows), b(rows);
    CK(cudaMemcpy(a.data(), dy1, rows * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dy2, rows * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < rows; ++i) { const double e = static_cast<double>(a[i]) - b[i]; num += e * e; den += static_cast<double>(a[i]) * a[i]; }
    const double rel = std::sqrt(num / den);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& fn) {
        float us = 0;
        for (int rep = 0; rep < 5; ++rep) {
            cudaEventRecord(e0);
            for (int it = 0; it < 100; ++it) fn();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); us = ms * 10;
        }
        return us;
    };
    const float us_ref = time_it(ref), us_new = time_it(neu);
    const double gbs_ref = w.size() / us_ref / 1e3, gbs_new = w.size() / us_new / 1e3;
    std::printf("  %s %5zux%-5zu %6.1f MB: rel diff %.1e | original %6.1f us (%5.1f GB/s) | wide-load %6.1f us (%5.1f GB/s) | %.2fx\n", name, rows, cols, w.size() / 1e6, rel,
                us_ref, gbs_ref, us_new, gbs_new, us_ref / us_new);
    if (rel > 1e-5) { std::printf("    FAIL: differs from the original by %.2e\n", rel); ++failures; }
    cudaFree(dW); cudaFree(dx); cudaFree(dy1); cudaFree(dy2);
    return 0;
}

int main() {
    std::puts("working set is a single matrix (13-40 MB, larger than L2): results are DRAM bandwidth, 182 GB/s is the probe's streaming ceiling");
    run(QuantType::Q4_K, "Q4_K", 9216, 2560);   // 4B ffn gate/up
    run(QuantType::Q4_K, "Q4_K", 2560, 9216);   // 4B ffn down (Q4_K layers)
    run(QuantType::Q4_K, "Q4_K", 12288, 1536);  // Nano ffn gate/up (KV-sharing layers)
    run(QuantType::Q4_K, "Q4_K", 262144, 1536); // Nano / tied output head
    run(QuantType::Q4_K, "Q4_K", 1024, 2560);   // small projection
    run(QuantType::Q5_K, "Q5_K", 8192, 2560);   // 4B DeltaNet attn_qkv
    run(QuantType::Q5_K, "Q5_K", 2560, 4096);   // 4B DeltaNet ssm_out
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
