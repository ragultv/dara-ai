// Prefill attention scores with int8 queries x int8 keys (dp4a) vs the FP32 GEMM path, on Q8 caches: accuracy of the attention output (relative L2 and max
// error against the FP32 path) and speed of one 512-token chunk at several context lengths, for the shapes the models use.
// Usage: attn_prefill_i8_test [--no-speed]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "model/attn_prefill.h"
#include "model/attn_splitkv.h"

using namespace dara::model;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char** argv) {
    const bool speed = !(argc > 1 && std::strcmp(argv[1], "--no-speed") == 0);
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    std::printf("compute capability major %d: the int8 path is %s on this GPU\n", major, major >= 8 ? "NOT used (tensor-core attention is)" : "available");
    struct Shape { const char* name; int heads, kv_heads, d, window, N; float scale; };
    const Shape shapes[] = {
        {"Nano global  (8 heads : 1 kv, d=512, causal)", 8, 1, 512, 0, 66000, 0.2f},
        {"Nano sliding (8 heads : 1 kv, d=256, window 512)", 8, 1, 256, 512, 20000, 0.2f},
        {"Qwen         (16 heads : 4 kv, d=256, causal)", 16, 4, 256, 0, 34000, 1.0f / 16.0f},
    };
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int bad = 0;
    for (const Shape& sh : shapes) {
        std::printf("%s\n", sh.name);
        const int N = sh.N, d = sh.d, KH = sh.kv_heads, H = sh.heads, n = 512;
        const size_t kvn = static_cast<size_t>(N) * KH * d;
        std::vector<float> hK(kvn), hV(kvn);
        for (auto& v : hK) v = nd(rng);
        for (auto& v : hV) v = nd(rng);
        float *dK = nullptr, *dV = nullptr, *dq = nullptr, *dout0 = nullptr, *dout1 = nullptr, *dscr = nullptr, *dKn = nullptr, *dVn = nullptr;
        int8_t *dKc = nullptr, *dVc = nullptr;
        CK(cudaMalloc(&dK, kvn * 4)); CK(cudaMalloc(&dV, kvn * 4));
        CK(cudaMemcpy(dK, hK.data(), kvn * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, hV.data(), kvn * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dKc, kvn)); CK(cudaMalloc(&dVc, kvn));
        const size_t scn = static_cast<size_t>(N) * KH * (d / 32);
        CK(cudaMalloc(&dKn, scn * 4)); CK(cudaMalloc(&dVn, scn * 4));
        {
            std::vector<int64_t> hp(N);
            for (int i = 0; i < N; ++i) hp[i] = i;
            int64_t* dp = nullptr;
            CK(cudaMalloc(&dp, static_cast<size_t>(N) * 8));
            CK(cudaMemcpy(dp, hp.data(), static_cast<size_t>(N) * 8, cudaMemcpyHostToDevice));
            for (int p0 = 0; p0 < N; p0 += 4096) {
                const int m = std::min(4096, N - p0);
                if (q8kv_append_cuda(dK + static_cast<size_t>(p0) * KH * d, dp + p0, m, KH, d, reinterpret_cast<uint8_t*>(dKc), dKn)) { std::puts("append failed"); return 1; }
                if (q8kv_append_cuda(dV + static_cast<size_t>(p0) * KH * d, dp + p0, m, KH, d, reinterpret_cast<uint8_t*>(dVc), dVn)) { std::puts("append failed"); return 1; }
            }
            CK(cudaDeviceSynchronize());
            CK(cudaFree(dp));
        }
        const size_t qn = static_cast<size_t>(n) * H * d;
        std::vector<float> hq(qn);
        for (auto& v : hq) v = nd(rng);
        CK(cudaMalloc(&dq, qn * 4)); CK(cudaMalloc(&dout0, qn * 4)); CK(cudaMalloc(&dout1, qn * 4));
        CK(cudaMemcpy(dq, hq.data(), qn * 4, cudaMemcpyHostToDevice));
        size_t scr = 0;
        for (int pos0 : {0, 600, 4096, 20000, 33000, 65000}) if (pos0 + n <= N) scr = std::max(scr, attn_prefill_scratch_floats(n, pos0, sh.window, H, KH));
        CK(cudaMalloc(&dscr, scr * 4));
        auto run = [&](int mode, int pos0, float* out) {
            attn_prefill_set_qk_int8(mode);
            return attn_prefill_cuda(AttnFmt::Q8, dq, nullptr, dKc, dKn, dVc, dVn, n, pos0, H, KH, d, sh.window, sh.scale, dscr, scr, out);
        };
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int pos0 : {0, 600, 4096, 20000, 33000, 65000}) {
            if (pos0 + n > N) continue;
            if (run(0, pos0, dout0) || run(1, pos0, dout1)) { std::puts("launch failed"); return 1; }
            CK(cudaDeviceSynchronize());
            std::vector<float> a(qn), b(qn);
            CK(cudaMemcpy(a.data(), dout0, qn * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dout1, qn * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0, mx = 0;
            for (size_t i = 0; i < qn; ++i) { const double e = static_cast<double>(b[i]) - a[i]; num += e * e; den += static_cast<double>(a[i]) * a[i]; mx = std::max(mx, std::fabs(e)); }
            const double rel = std::sqrt(num / std::max(den, 1e-30));
            float ms[2] = {0, 0};
            if (speed) {
                for (int mode = 0; mode < 2; ++mode) {
                    run(mode, pos0, dout0); cudaDeviceSynchronize();
                    cudaEventRecord(e0);
                    for (int r = 0; r < 3; ++r) run(mode, pos0, dout0);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float t = 0; cudaEventElapsedTime(&t, e0, e1); ms[mode] = t / 3;
                }
            }
            std::printf("    context %6d + 512 tokens: relative L2 error %.4f, max |err| %.4f%s", pos0, rel, mx, major >= 8 ? "  (same path: int8 not used)" : "");
            if (speed) std::printf(" | FP32 %.2f ms | int8-QK %.2f ms | x%.2f", ms[0], ms[1], ms[0] / ms[1]);
            std::printf("\n");
            if (major < 8 && rel > 0.05) ++bad;
            if (KH > 1) {   // grouped-query prefill vs one GEMM pair per query head (FP32 scores in both): must be bitwise identical
                attn_prefill_set_group(0);
                if (run(0, pos0, dout0)) { std::puts("launch failed"); return 1; }
                attn_prefill_set_group(1);
                if (run(0, pos0, dout1)) { std::puts("launch failed"); return 1; }
                CK(cudaDeviceSynchronize());
                std::vector<float> g0(qn), g1(qn);
                CK(cudaMemcpy(g0.data(), dout0, qn * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(g1.data(), dout1, qn * 4, cudaMemcpyDeviceToHost));
                const bool same = std::memcmp(g0.data(), g1.data(), qn * 4) == 0;
                float gms[2] = {0, 0};
                if (speed) {
                    for (int grp = 0; grp < 2; ++grp) {
                        attn_prefill_set_group(grp);
                        run(0, pos0, dout0); cudaDeviceSynchronize();
                        cudaEventRecord(e0);
                        for (int r = 0; r < 3; ++r) run(0, pos0, dout0);
                        cudaEventRecord(e1); cudaEventSynchronize(e1);
                        float t = 0; cudaEventElapsedTime(&t, e0, e1); gms[grp] = t / 3;
                    }
                }
                attn_prefill_set_group(-1);
                std::printf("      grouped heads vs per-head pairs: %s", same ? "bitwise identical" : "DIFFERENT  <-- FAIL");
                if (speed) std::printf(" | per-head %.2f ms | grouped %.2f ms | x%.2f", gms[0], gms[1], gms[0] / gms[1]);
                std::printf("\n");
                if (!same) ++bad;
            }
        }
        attn_prefill_set_qk_int8(-1);
        cudaFree(dK); cudaFree(dV); cudaFree(dKc); cudaFree(dVc); cudaFree(dKn); cudaFree(dVn); cudaFree(dq); cudaFree(dout0); cudaFree(dout1); cudaFree(dscr);
    }
    std::printf("\n%s\n", bad ? "FAILED: an error above 5% relative" : "done");
    return bad ? 1 : 0;
}
