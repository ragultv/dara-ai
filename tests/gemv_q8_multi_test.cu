// Multi-token int8 GEMV (speculative verification): accuracy vs the single-vector int8 kernel, and time per call for T = 1..16 tokens, against the
// FP32 multi-vector GEMV and T separate single-token calls. The goal: checking T tokens should cost little more than one token (the weights are read once).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/quant.h"

using namespace dara::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static int run(QuantType qt, const char* name, size_t rows, size_t cols) {
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    const QuantFormat& f = quant_format(qt);
    const size_t blocks = cols / 256;
    std::vector<uint8_t> w(rows * blocks * f.block_bytes);
    { std::vector<float> row(cols); for (size_t r = 0; r < rows; ++r) { for (auto& v : row) v = nd(rng) * 0.05f; f.quantize(row.data(), cols, w.data() + r * blocks * f.block_bytes); } }
    constexpr int kMaxT = 16;
    std::vector<float> x(kMaxT * cols);
    for (auto& v : x) v = nd(rng);
    uint8_t* dW; float *dx, *dy, *dy1; void* scr;
    CK(cudaMalloc(&dW, w.size())); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, rows * kMaxT * 4)); CK(cudaMalloc(&dy1, rows * kMaxT * 4));
    CK(cudaMalloc(&scr, q8act_bytes(cols, kMaxT)));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& fn) {
        for (int i = 0; i < 5; ++i) fn();
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) { cudaEventRecord(e0); for (int it = 0; it < 20; ++it) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms / 20); }
        return best * 1000.0f;   // us
    };
    std::printf("  %s %zux%zu (%.1f MB):\n", name, rows, cols, w.size() / 1e6);
    float us1 = 0;
    for (int T : {1, 2, 3, 4, 6, 8, 12, 16}) {
        // accuracy: the multi kernel vs T single-vector int8 calls on the same quantised activations
        if (quantize_activations_cuda(dx, cols, cols, T, scr, nullptr) || gemv_q8_multi_cuda(qt, dW, blocks, scr, cols, T, dy, rows, rows, nullptr)) { std::puts("launch failed"); return 1; }
        for (int t = 0; t < T; ++t) if (gemv_q8_cuda(qt, dW, blocks, scr, cols, T, t, dy1 + t * rows, rows, nullptr)) { std::puts("launch failed"); return 1; }
        CK(cudaDeviceSynchronize());
        std::vector<float> a(rows * T), b(rows * T);
        CK(cudaMemcpy(a.data(), dy, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dy1, b.size() * 4, cudaMemcpyDeviceToHost));
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); ++i) { num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]); den += double(b[i]) * b[i]; }
        const float t_multi = time_it([&] { quantize_activations_cuda(dx, cols, cols, T, scr, nullptr); gemv_q8_multi_cuda(qt, dW, blocks, scr, cols, T, dy, rows, rows, nullptr); });
        const float t_fp32 = time_it([&] { gemv_quant_multi_cuda(qt, dW, blocks, dx, cols, dy, rows, rows, cols, T, nullptr); });
        if (T == 1) us1 = t_multi;
        std::printf("    T=%2d: int8 multi %7.1f us (%.2fx one token, %5.1f GB/s) | FP32 multi %7.1f us | rel. diff vs single-vector int8 %.1e\n", T, t_multi, t_multi / us1,
                    w.size() / t_multi / 1e3, t_fp32, std::sqrt(num / den));
    }
    cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dy1); cudaFree(scr);
    return 0;
}

int main() {
    run(QuantType::Q4_K, "Q4_K", 9216, 2560);   // 4B ffn gate/up
    run(QuantType::Q5_K, "Q5_K", 8192, 2560);   // 4B attn/dn qkv
    run(QuantType::Q6_K, "Q6_K", 2560, 9216);   // 4B ffn down (Q6_K half)
    return 0;
}
