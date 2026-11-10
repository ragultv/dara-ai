// Multi-vector quantized GEMV: (1) every output bitwise identical to the single-vector kernel, for all nt in 1..13 and both
// fused formats; (2) time for nt vectors vs nt single-vector launches, on real model shapes.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kernels/gemv.h"
#include "kernels/quant.h"

using namespace dara::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

static bool make_matrix(QuantType t, size_t rows, size_t cols, std::mt19937& rng, std::vector<uint8_t>& out) {
    const QuantFormat& f = quant_format(t);
    const size_t blocks = cols / f.block_weights;
    out.assign(rows * blocks * f.block_bytes, 0);
    std::normal_distribution<float> nd(0.f, 0.05f);
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        for (auto& v : row) v = nd(rng);
        f.quantize(row.data(), cols, out.data() + r * blocks * f.block_bytes);
    }
    return true;
}

static int run(QuantType t, const char* name, size_t rows, size_t cols, bool timing) {
    std::mt19937 rng(7);
    const QuantFormat& f = quant_format(t);
    std::vector<uint8_t> w;
    make_matrix(t, rows, cols, rng, w);
    const int NT = 13;
    std::vector<float> x(static_cast<size_t>(NT) * cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);
    uint8_t* dW; float *dx, *dy_s, *dy_m;
    CK(cudaMalloc(&dW, w.size()));
    CK(cudaMalloc(&dx, x.size() * 4));
    CK(cudaMalloc(&dy_s, static_cast<size_t>(NT) * rows * 4));
    CK(cudaMalloc(&dy_m, static_cast<size_t>(NT) * rows * 4));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    const size_t ld = cols / f.block_weights;
    for (int t1 = 0; t1 < NT; ++t1)
        if (gemv_quant_cuda(t, dW, ld, dx + static_cast<size_t>(t1) * cols, dy_s + static_cast<size_t>(t1) * rows, rows, cols)) return 1;
    CK(cudaDeviceSynchronize());
    std::vector<float> ys(static_cast<size_t>(NT) * rows), ym(static_cast<size_t>(NT) * rows);
    CK(cudaMemcpy(ys.data(), dy_s, ys.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        CK(cudaMemset(dy_m, 0xFF, static_cast<size_t>(NT) * rows * 4));
        if (gemv_quant_multi_cuda(t, dW, ld, dx, cols, dy_m, rows, rows, cols, nt)) return 1;
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ym.data(), dy_m, ym.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(ym.data(), ys.data(), static_cast<size_t>(nt) * rows * 4) != 0) { ++bad; std::printf("  %s %zux%zu nt=%d: MISMATCH\n", name, rows, cols, nt); }
    }
    std::printf("  %s %zux%zu: nt 1..%d bitwise identical to single-vector kernel: %s\n", name, rows, cols, NT, bad ? "NO" : "yes");
    if (timing) {
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        std::printf("    nt   single x nt (us)   multi (us)   multi/single-1   speedup\n");
        float t_one = 0;
        for (int nt : {1, 2, 4, 8}) {
            float ts = 0, tm = 0;
            for (int rep = 0; rep < 3; ++rep) {  // rep 0 warms clocks
                cudaEventRecord(a);
                for (int it = 0; it < 50; ++it) for (int t1 = 0; t1 < nt; ++t1) gemv_quant_cuda(t, dW, ld, dx + static_cast<size_t>(t1) * cols, dy_s + static_cast<size_t>(t1) * rows, rows, cols);
                cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ts = ms * 1000 / 50;
                cudaEventRecord(a);
                for (int it = 0; it < 50; ++it) gemv_quant_multi_cuda(t, dW, ld, dx, cols, dy_m, rows, rows, cols, nt);
                cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms, a, b); tm = ms * 1000 / 50;
            }
            if (nt == 1) t_one = tm;
            std::printf("    %2d   %12.1f   %10.1f   %12.2fx   %6.2fx\n", nt, ts, tm, tm / t_one, ts / tm);
        }
    }
    cudaFree(dW); cudaFree(dx); cudaFree(dy_s); cudaFree(dy_m);
    return bad;
}

namespace dara::kernels { void set_rows_per_warp_experiment(int r); }

// Q4_K single/few-vector cost vs rows-per-warp, on the model's projection shapes.
static int rows_per_warp_sweep(size_t rows, size_t cols) {
    std::mt19937 rng(3);
    std::vector<uint8_t> w; make_matrix(QuantType::Q4_K, rows, cols, rng, w);
    std::vector<float> x(8 * cols, 0.5f);
    uint8_t* dW; float *dx, *dy;
    CK(cudaMalloc(&dW, w.size())); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMalloc(&dy, 8 * rows * 4));
    CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const size_t ld = cols / 256;
    for (int it = 0; it < 3000; ++it) gemv_quant_multi_cuda(QuantType::Q4_K, dW, ld, dx, cols, dy, rows, rows, cols, 4);  // clocks up
    cudaDeviceSynchronize();
    std::printf("  Q4_K %zux%zu (%.1f MB) us per call:  nt:rows/warp ->", rows, cols, w.size() / 1e6);
    for (int nt : {1, 2, 4}) for (int r : {1, 2, 4}) {
        dara::kernels::set_rows_per_warp_experiment(r);
        float us = 0;
        for (int rep = 0; rep < 3; ++rep) {
            cudaEventRecord(a);
            for (int it = 0; it < 100; ++it) gemv_quant_multi_cuda(QuantType::Q4_K, dW, ld, dx, cols, dy, rows, rows, cols, nt);
            cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); us = ms * 10;
        }
        std::printf("  %d:%d=%.0f", nt, r, us);
    }
    dara::kernels::set_rows_per_warp_experiment(0);
    std::printf("\n");
    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    return 0;
}

static int run_bf16(size_t rows, size_t cols) {
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t ld = (cols + 7) / 8 * 8, ldx = (cols + 3) / 4 * 4;  // padded strides: the kernel needs 16-byte aligned rows
    std::vector<uint16_t> w(rows * ld, 0);
    for (auto& v : w) { float f = nd(rng) * 0.05f; uint32_t u; std::memcpy(&u, &f, 4); v = static_cast<uint16_t>(u >> 16); }
    const int NT = 13;
    std::vector<float> x(static_cast<size_t>(NT) * ldx, 0.0f);
    for (auto& v : x) v = nd(rng);
    uint16_t* dW; float *dx, *dy1, *dy2;
    cudaMalloc(&dW, w.size() * 2); cudaMalloc(&dx, x.size() * 4); cudaMalloc(&dy1, static_cast<size_t>(NT) * rows * 4); cudaMalloc(&dy2, static_cast<size_t>(NT) * rows * 4);
    cudaMemcpy(dW, w.data(), w.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
    for (int k = 0; k < NT; ++k) gemv_cuda(WeightType::BF16, dW, ld, dx + static_cast<size_t>(k) * ldx, dy1 + static_cast<size_t>(k) * rows, rows, cols);
    cudaDeviceSynchronize();
    std::vector<float> a(static_cast<size_t>(NT) * rows), b(a.size());
    cudaMemcpy(a.data(), dy1, a.size() * 4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int nt = 1; nt <= NT; ++nt) {
        cudaMemset(dy2, 0xFF, static_cast<size_t>(NT) * rows * 4);
        if (gemv_multi_cuda(WeightType::BF16, dW, ld, dx, ldx, dy2, rows, rows, cols, nt)) { std::printf("  BF16 %zux%zu nt=%d: launch failed\n", rows, cols, nt); return 1; }
        cudaDeviceSynchronize();
        cudaMemcpy(b.data(), dy2, b.size() * 4, cudaMemcpyDeviceToHost);
        if (std::memcmp(a.data(), b.data(), static_cast<size_t>(nt) * rows * 4) != 0) { ++bad; std::printf("  BF16 %zux%zu nt=%d: MISMATCH\n", rows, cols, nt); }
    }
    std::printf("  BF16 %zux%zu: nt 1..%d bitwise identical to single-vector kernel: %s\n", rows, cols, NT, bad ? "NO" : "yes");
    cudaFree(dW); cudaFree(dx); cudaFree(dy1); cudaFree(dy2);
    return bad;
}

int main() {
    int bad = 0;
    rows_per_warp_sweep(9216, 2560);
    rows_per_warp_sweep(2560, 9216);
    rows_per_warp_sweep(4096, 4096);
    rows_per_warp_sweep(1024, 2560);
    bad += run_bf16(256, 1536);   // Nano inp_gate
    bad += run_bf16(1536, 256);   // Nano per-layer projection
    bad += run_bf16(8960, 1536);  // Nano per_layer_model_proj
    bad += run_bf16(100, 515);    // odd width: exercises the scalar tail
    bad += run(QuantType::Q4_K, "Q4_K", 100, 512, false);
    bad += run(QuantType::Q6_K, "Q6_K", 100, 512, false);
    bad += run(QuantType::Q4_K, "Q4_K", 9216, 2560, true);   // 4B ffn gate/up
    bad += run(QuantType::Q4_K, "Q4_K", 2560, 9216, true);   // 4B ffn down shape
    bad += run(QuantType::Q6_K, "Q6_K", 2560, 9216, true);
    bad += run(QuantType::Q6_K, "Q6_K", 4096, 4096, true);
    bad += run(QuantType::Q5_K, "Q5_K", 100, 512, false);
    bad += run(QuantType::Q8_0, "Q8_0", 100, 512, false);
    bad += run(QuantType::Q5_K, "Q5_K", 8192, 2560, true);   // 4B DeltaNet attn_qkv (all 24 DeltaNet layers)
    bad += run(QuantType::Q5_K, "Q5_K", 2560, 4096, true);   // 4B DeltaNet ssm_out
    bad += run(QuantType::Q8_0, "Q8_0", 32, 2560, true);     // 4B ssm_alpha / ssm_beta
    std::printf(bad ? "FAILED\n" : "ok\n");
    return bad ? 1 : 0;
}
