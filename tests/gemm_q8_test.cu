// int8-activation (dp4a) prefill GEMM vs the FP32 GEMM: accuracy (loss from 8-bit activations) and throughput, for prompt-sized token counts.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/quant.h"

namespace dara::kernels {
bool gemm_q8_available(QuantType t);
int gemm_q8_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream);
}  // namespace dara::kernels
using namespace dara::kernels;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static int run(QuantType qt, const char* name, size_t rows, size_t cols, int T) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const QuantFormat& f = quant_format(qt);
    const size_t blocks = cols / 256;
    std::vector<uint8_t> w(rows * blocks * f.block_bytes);
    { std::vector<float> row(cols); for (size_t r = 0; r < rows; ++r) { for (auto& v : row) v = nd(rng) * 0.05f; f.quantize(row.data(), cols, w.data() + r * blocks * f.block_bytes); } }
    std::vector<float> x(static_cast<size_t>(T) * cols);
    for (size_t i = 0; i < x.size(); ++i) x[i] = nd(rng) * (1.0f + 0.5f * std::sin(0.01f * (i % cols)));
    uint8_t* dW; float *dx, *dy1, *dy2; void* scr;
    CK(cudaMalloc(&dW, w.size())); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy1, rows * T * 4)); CK(cudaMalloc(&dy2, rows * T * 4)); CK(cudaMalloc(&scr, q8act_bytes(cols, T)));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    auto ref = [&] { return gemm_available(qt) ? gemm_quant_cuda(qt, dW, blocks, dx, cols, dy1, rows, rows, cols, T, nullptr)
                                               : gemv_quant_multi_cuda(qt, dW, blocks, dx, cols, dy1, rows, rows, cols, T, nullptr); };  // Q6_K: today's FP32 path
    auto q8 = [&] { return quantize_activations_cuda(dx, cols, cols, T, scr, nullptr) | gemm_q8_cuda(qt, dW, blocks, scr, cols, T, dy2, rows, rows, nullptr); };
    if (ref() || q8()) { std::puts("launch failed"); return 1; }
    CK(cudaDeviceSynchronize());
    std::vector<float> a(rows * T), b(rows * T);
    CK(cudaMemcpy(a.data(), dy1, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dy2, b.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = static_cast<double>(a[i]) - b[i]; num += e * e; den += static_cast<double>(a[i]) * a[i]; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto time_it = [&](auto&& fn) {
        float best = 1e30f;
        for (int rep = 0; rep < 5; ++rep) { cudaEventRecord(e0); for (int it = 0; it < 5; ++it) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms / 5); }
        return best;
    };
    const float ms_ref = time_it(ref), ms_q8 = time_it(q8);
    const double flop = 2.0 * rows * cols * T;
    std::printf("  %s %6zux%-5zu T=%3d: rel err %.2e | FP32 GEMM %7.3f ms (%5.2f TFLOPS) | int8 GEMM incl. quantise %7.3f ms (%5.2f TOPS, %.2fx)\n", name, rows, cols, T,
                std::sqrt(num / den), ms_ref, flop / ms_ref / 1e9, ms_q8, flop / ms_q8 / 1e9, ms_ref / ms_q8);
    cudaFree(dW); cudaFree(dx); cudaFree(dy1); cudaFree(dy2); cudaFree(scr);
    return 0;
}

int main() {
    run(QuantType::Q4_K, "Q4_K", 9216, 2560, 512);    // 4B ffn gate/up
    run(QuantType::Q4_K, "Q4_K", 2560, 9216, 512);    // 4B ffn down (Q4_K half)
    run(QuantType::Q5_K, "Q5_K", 8192, 2560, 512);    // 4B attn qkv
    run(QuantType::Q4_K, "Q4_K", 6144, 1536, 512);    // Nano mlp
    run(QuantType::Q4_K, "Q4_K", 2048, 1536, 512);    // Nano q
    run(QuantType::Q4_K, "Q4_K", 9216, 2560, 100);    // ragged token count
    run(QuantType::Q5_K, "Q5_K", 1000, 2560, 77);     // ragged rows and tokens
    run(QuantType::Q6_K, "Q6_K", 2560, 9216, 512);    // 4B ffn down (Q6_K half)
    run(QuantType::Q6_K, "Q6_K", 1536, 6144, 512);    // Nano-sized down
    run(QuantType::Q6_K, "Q6_K", 1000, 2560, 77);     // ragged
    return 0;
}
