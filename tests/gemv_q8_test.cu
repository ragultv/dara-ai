// int8-activation (dp4a) Q4_K / Q5_K / Q6_K GEMV vs the FP32 GEMV: accuracy (the loss from 8-bit activations) and achieved memory bandwidth.
// Usage: gemv_q8_test
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/q4k.h"
#include "kernels/quant.h"

namespace dara::kernels {
int gemv_q4k_cuda_ref(const void*, size_t, const float*, float*, size_t, size_t, void*);
int gemv_q5k_cuda_ref(const void*, size_t, const float*, float*, size_t, size_t, void*);
int gemv_q8_cuda(QuantType, const void*, size_t, const void*, size_t, int, int, float*, size_t, void*);
size_t q8act_bytes(size_t K, int T);
int quantize_activations_cuda(const float* dx, size_t ldx, size_t K, int T, void* scratch, void* stream);
int gemv_q4k_q8_cuda(const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int t, float* dy, size_t rows, void* stream);
}  // namespace dara::kernels

using namespace dara::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static int run(QuantType qt, const char* name, size_t rows, size_t cols) {
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    const QuantFormat& f = quant_format(qt);
    const size_t blocks = cols / 256;
    std::vector<uint8_t> w(rows * blocks * f.block_bytes);
    {
        std::vector<float> row(cols);
        for (size_t r = 0; r < rows; ++r) { for (auto& v : row) v = nd(rng) * 0.05f; f.quantize(row.data(), cols, w.data() + r * blocks * f.block_bytes); }
    }
    std::vector<float> x(cols);
    for (size_t i = 0; i < cols; ++i) x[i] = nd(rng) * (1.0f + 0.5f * std::sin(0.01f * i));   // mildly non-uniform magnitudes
    uint8_t* dW; float *dx, *dy1, *dy2; void* scr;
    CK(cudaMalloc(&dW, w.size())); CK(cudaMalloc(&dx, cols * 4)); CK(cudaMalloc(&dy1, rows * 4)); CK(cudaMalloc(&dy2, rows * 4)); CK(cudaMalloc(&scr, q8act_bytes(cols, 1)));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), cols * 4, cudaMemcpyHostToDevice));
    auto ref = [&] { return qt == QuantType::Q4_K ? gemv_q4k_cuda_ref(dW, blocks, dx, dy1, rows, cols, nullptr)
                           : qt == QuantType::Q5_K ? gemv_q5k_cuda_ref(dW, blocks, dx, dy1, rows, cols, nullptr)
                                                   : gemv_quant_cuda(qt, dW, blocks, dx, dy1, rows, cols, nullptr); };
    if (ref() || quantize_activations_cuda(dx, cols, cols, 1, scr, nullptr) || gemv_q8_cuda(qt, dW, blocks, scr, cols, 1, 0, dy2, rows, nullptr)) {
        std::puts("launch failed"); return 1;
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> a(rows), b(rows);
    CK(cudaMemcpy(a.data(), dy1, rows * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dy2, rows * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0, worst = 0;
    for (size_t i = 0; i < rows; ++i) { const double e = static_cast<double>(a[i]) - b[i]; num += e * e; den += static_cast<double>(a[i]) * a[i]; worst = std::max(worst, std::fabs(e)); }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& fn) {
        float us = 0;
        for (int rep = 0; rep < 5; ++rep) { cudaEventRecord(e0); for (int it = 0; it < 100; ++it) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); us = ms * 10; }
        return us;
    };
    const float us_ref = time_it([&] { ref(); });
    const float us_q8 = time_it([&] { gemv_q8_cuda(qt, dW, blocks, scr, cols, 1, 0, dy2, rows, nullptr); });
    const float us_qa = time_it([&] { quantize_activations_cuda(dx, cols, cols, 1, scr, nullptr); });
    std::printf("  %s %6zux%-5zu %6.1f MB: relative error vs FP32 GEMV %.2e | FP32 kernel %6.1f us (%5.1f GB/s) | int8 dp4a %6.1f us (%5.1f GB/s, %.2fx) | quantise x: %.1f us\n", name,
                rows, cols, w.size() / 1e6, std::sqrt(num / den), us_ref, w.size() / us_ref / 1e3, us_q8, w.size() / us_q8 / 1e3, us_ref / us_q8, us_qa);
    cudaFree(dW); cudaFree(dx); cudaFree(dy1); cudaFree(dy2); cudaFree(scr);
    return 0;
}

int main() {
    run(QuantType::Q4_K, "Q4_K", 9216, 2560);
    run(QuantType::Q4_K, "Q4_K", 2560, 9216);
    run(QuantType::Q4_K, "Q4_K", 12288, 1536);
    run(QuantType::Q4_K, "Q4_K", 262144, 1536);
    run(QuantType::Q4_K, "Q4_K", 1024, 2560);
    run(QuantType::Q5_K, "Q5_K", 8192, 2560);
    run(QuantType::Q5_K, "Q5_K", 2560, 4096);
    run(QuantType::Q6_K, "Q6_K", 2560, 9216);
    run(QuantType::Q6_K, "Q6_K", 1024, 2560);
    run(QuantType::Q6_K, "Q6_K", 1536, 6144);
    return 0;
}
