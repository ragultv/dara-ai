// Q4_K prefill GEMM vs the multi-vector GEMV (the correctness oracle), and speed in TFLOPS against the GPU's FP32 peak.
// Usage: gemm_test
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/quant.h"

using namespace dara::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static int failures = 0;
static void check(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what); if (!ok) ++failures; }

static int run_format(QuantType qt, const char* qname, const double peak);

int main() {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    // FP32 peak = SMs x 64 FMA lanes (Turing) x 2 flops x boost clock
    const double peak = prop.multiProcessorCount * 64.0 * 2.0 * 1.6e9 / 1e12;
    std::printf("GPU %s, %d SMs; nominal FP32 peak about %.2f TFLOPS at 1.6 GHz (the laptop part throttles below this)\n", prop.name, prop.multiProcessorCount, peak);
    int rc = run_format(QuantType::Q4_K, "Q4_K", peak);
    rc += run_format(QuantType::Q5_K, "Q5_K", peak);
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : rc;
}

static int run_format(QuantType qt, const char* qname, const double peak) {
    std::printf("=== %s ===\n", qname);
    struct Shape { const char* name; size_t rows, cols; };
    const Shape shapes[] = {{"ffn gate/up (4B)  9216 x 2560", 9216, 2560}, {"ffn down   (4B)  2560 x 9216", 2560, 9216}, {"attn/ssm proj     8192 x 2560", 8192, 2560},
                            {"odd rows         1000 x 1536 (tile edges)", 1000, 1536}};
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    const QuantFormat& f = quant_format(qt);
    for (const Shape& sh : shapes) {
        std::printf("%s\n", sh.name);
        const size_t blocks = sh.cols / f.block_weights;
        std::vector<uint8_t> w(sh.rows * blocks * f.block_bytes);
        {
            std::vector<float> row(sh.cols);
            for (size_t r = 0; r < sh.rows; ++r) {
                for (auto& v : row) v = nd(rng) * 0.05f;
                f.quantize(row.data(), sh.cols, w.data() + r * blocks * f.block_bytes);
            }
        }
        uint8_t* dW; CK(cudaMalloc(&dW, w.size())); CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice));
        for (int T : {13, 32, 64, 100, 128}) {
            std::vector<float> x(static_cast<size_t>(T) * sh.cols);
            for (auto& v : x) v = nd(rng);
            float *dx, *dy1, *dy2;
            CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy1, static_cast<size_t>(T) * sh.rows * 4)); CK(cudaMalloc(&dy2, static_cast<size_t>(T) * sh.rows * 4));
            CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            if (gemv_quant_multi_cuda(qt, dW, blocks, dx, sh.cols, dy1, sh.rows, sh.rows, sh.cols, T) || gemm_quant_cuda(qt, dW, blocks, dx, sh.cols, dy2, sh.rows, sh.rows, sh.cols, T)) {
                std::puts("launch failed"); return 1;
            }
            CK(cudaDeviceSynchronize());
            std::vector<float> a(static_cast<size_t>(T) * sh.rows), b(a.size());
            CK(cudaMemcpy(a.data(), dy1, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dy2, b.size() * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0;
            for (size_t i = 0; i < a.size(); ++i) { const double e = static_cast<double>(a[i]) - b[i]; num += e * e; den += static_cast<double>(a[i]) * a[i]; }
            const double rel = std::sqrt(num / den);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            auto time_it = [&](auto&& fn) {
                float us = 0;
                for (int rep = 0; rep < 4; ++rep) {   // first reps warm the clocks
                    cudaEventRecord(e0);
                    for (int it = 0; it < 20; ++it) fn();
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1); us = ms * 1000 / 20;
                }
                return us;
            };
            const float us_gemv = time_it([&] { gemv_quant_multi_cuda(qt, dW, blocks, dx, sh.cols, dy1, sh.rows, sh.rows, sh.cols, T); });
            const float us_gemm = time_it([&] { gemm_quant_cuda(qt, dW, blocks, dx, sh.cols, dy2, sh.rows, sh.rows, sh.cols, T); });
            const double flops = 2.0 * static_cast<double>(sh.rows) * sh.cols * T;
            std::printf("  T=%3d: rel diff %.1e | multi-GEMV %8.1f us (%.2f TFLOPS) | GEMM %8.1f us (%.2f TFLOPS, %.0f%% of peak) | speedup %.2fx\n", T, rel, us_gemv,
                        flops / us_gemv / 1e6, us_gemm, flops / us_gemm / 1e6, 100.0 * flops / us_gemm / 1e6 / peak, us_gemv / us_gemm);
            if (rel > 1e-5) { std::printf("    FAIL: GEMM differs from GEMV by %.2e\n", rel); ++failures; }
            cudaFree(dx); cudaFree(dy1); cudaFree(dy2);
        }
        cudaFree(dW);
    }
    return 0;
}
