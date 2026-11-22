// Grouped-query decode attention (partial_gqa_kernel) vs the per-head kernels it replaces: the output must be BITWISE identical for F32 and Q8 caches,
// across the shapes the models use (Nano global 8:1 d=512, Nano sliding 8:1 d=256 window 512, Qwen 4:1 d=256 with gate), short and long contexts, one token
// and several; then decode speed of both at long contexts (the reason the kernel exists: the per-head kernels re-read the K/V cache once per query head).
// Usage: attn_gqa_test [--no-speed]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "model/attn_splitkv.h"

using namespace dara::model;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static int failures = 0;
static void check(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what); if (!ok) ++failures; }

int main(int argc, char** argv) {
    const bool speed = !(argc > 1 && std::strcmp(argv[1], "--no-speed") == 0);
    if (attn_splitkv_init()) { std::puts("init failed"); return 1; }
    struct Shape { const char* name; int heads, kv_heads, d, window, N; float scale; bool gate; };
    const Shape shapes[] = {
        {"Nano global  (8 heads : 1 kv, d=512, causal)", 8, 1, 512, 0, 131072, 0.2f, false},
        {"Nano sliding (8 heads : 1 kv, d=256, window 512)", 8, 1, 256, 512, 32768, 0.2f, false},
        {"Qwen         (16 heads : 4 kv, d=256, causal, gate)", 16, 4, 256, 0, 32768, 1.0f / 16.0f, true},
        {"Qwen         (8 heads : 4 kv, d=256, causal) 2:1", 8, 4, 256, 0, 32768, 1.0f / 16.0f, false},
    };
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const Shape& sh : shapes) {
        std::printf("%s\n", sh.name);
        const int N = sh.N, d = sh.d, KH = sh.kv_heads, H = sh.heads, T = 3;
        const size_t kvn = static_cast<size_t>(N) * KH * d;
        std::vector<float> hK(kvn), hV(kvn);
        for (auto& v : hK) v = nd(rng);
        for (auto& v : hV) v = nd(rng);
        float *dK = nullptr, *dV = nullptr, *dq = nullptr, *dg = nullptr, *dout0 = nullptr, *dout1 = nullptr, *dscr = nullptr;
        int8_t *dKc = nullptr, *dVc = nullptr;
        float *dKn = nullptr, *dVn = nullptr;
        int64_t* dpos = nullptr;
        CK(cudaMalloc(&dK, kvn * 4)); CK(cudaMalloc(&dV, kvn * 4));
        CK(cudaMemcpy(dK, hK.data(), kvn * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, hV.data(), kvn * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dKc, kvn)); CK(cudaMalloc(&dVc, kvn));
        const size_t scn = static_cast<size_t>(N) * KH * (d / 32);
        CK(cudaMalloc(&dKn, scn * 4)); CK(cudaMalloc(&dVn, scn * 4));
        {   // quantise the same K/V into INT8 caches position by position range (chunks keep the temporary small)
            std::vector<int64_t> hp(N);
            for (int i = 0; i < N; ++i) hp[i] = i;
            int64_t* dp = nullptr;
            CK(cudaMalloc(&dp, static_cast<size_t>(N) * 8));
            CK(cudaMemcpy(dp, hp.data(), static_cast<size_t>(N) * 8, cudaMemcpyHostToDevice));
            const int chunk = 4096;
            for (int p0 = 0; p0 < N; p0 += chunk) {
                const int n = std::min(chunk, N - p0);
                if (q8kv_append_cuda(dK + static_cast<size_t>(p0) * KH * d, dp + p0, n, KH, d, reinterpret_cast<uint8_t*>(dKc), dKn)) { std::puts("append failed"); return 1; }
                if (q8kv_append_cuda(dV + static_cast<size_t>(p0) * KH * d, dp + p0, n, KH, d, reinterpret_cast<uint8_t*>(dVc), dVn)) { std::puts("append failed"); return 1; }
            }
            CK(cudaDeviceSynchronize());
            CK(cudaFree(dp));
        }
        const size_t qn = static_cast<size_t>(T) * H * d;
        std::vector<float> hq(qn), hg(qn);
        for (auto& v : hq) v = nd(rng);
        for (auto& v : hg) v = nd(rng);
        CK(cudaMalloc(&dq, qn * 4)); CK(cudaMalloc(&dg, qn * 4)); CK(cudaMalloc(&dout0, qn * 4)); CK(cudaMalloc(&dout1, qn * 4));
        CK(cudaMemcpy(dq, hq.data(), qn * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg, hg.data(), qn * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dpos, T * 8));
        const size_t per = attn_splitkv_scratch_floats_per_token(AttnFmt::F32, H, d, N);
        const size_t scr_floats = 8 * per;
        CK(cudaMalloc(&dscr, scr_floats * 4));
        auto run = [&](AttnFmt fmt, int gqa, int n, float* out) {
            attn_splitkv_set_gqa(gqa);
            const void *K = fmt == AttnFmt::F32 ? static_cast<const void*>(dK) : static_cast<const void*>(dKc), *V = fmt == AttnFmt::F32 ? static_cast<const void*>(dV) : static_cast<const void*>(dVc);
            return attn_splitkv_cuda(fmt, dq, sh.gate ? dg : nullptr, K, fmt == AttnFmt::F32 ? nullptr : dKn, V, fmt == AttnFmt::F32 ? nullptr : dVn, dpos, n, H, KH, d, sh.window,
                                     sh.scale, N, dscr, scr_floats, out);
        };
        for (AttnFmt fmt : {AttnFmt::F32, AttnFmt::Q8}) {
            bool all_same = true;
            double worst = 0;
            for (int ctx : {1, 37, 128, 129, 500, 4097, 20000, N - 8}) {
                if (ctx + 3 > N) continue;
                for (int n : {1, 3}) {
                    int64_t hp[3] = {ctx - 1, ctx, ctx + 1};   // the query tokens sit at the last positions of the context
                    CK(cudaMemcpy(dpos, hp, sizeof hp, cudaMemcpyHostToDevice));
                    if (run(fmt, 0, n, dout0) || run(fmt, 1, n, dout1)) { std::puts("launch failed"); return 1; }
                    CK(cudaDeviceSynchronize());
                    std::vector<float> a(static_cast<size_t>(n) * H * d), b(a.size());
                    CK(cudaMemcpy(a.data(), dout0, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dout1, b.size() * 4, cudaMemcpyDeviceToHost));
                    if (std::memcmp(a.data(), b.data(), a.size() * 4) != 0) {
                        all_same = false;
                        for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
                        std::printf("    differs at ctx %d, %d token(s), max |diff| %.3g\n", ctx, n, worst);
                    }
                }
            }
            check(all_same, fmt == AttnFmt::F32 ? "F32 cache: GQA output bitwise identical to the per-head kernels (ctx 1 .. N, 1 and 3 tokens)"
                                                : "Q8 cache : GQA output bitwise identical to the per-head kernels (ctx 1 .. N, 1 and 3 tokens)");
        }
        if (speed) {
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int ctx : {1024, 8192, 32768, N - 8}) {
                if (ctx + 1 > N) continue;
                const int64_t hp = ctx - 1;
                CK(cudaMemcpy(dpos, &hp, 8, cudaMemcpyHostToDevice));
                float ms[2] = {0, 0};
                for (int gqa = 0; gqa < 2; ++gqa) {
                    for (int w = 0; w < 5; ++w) run(AttnFmt::Q8, gqa, 1, dout0);
                    cudaDeviceSynchronize();
                    cudaEventRecord(e0);
                    for (int r = 0; r < 40; ++r) run(AttnFmt::Q8, gqa, 1, dout0);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float t = 0; cudaEventElapsedTime(&t, e0, e1); ms[gqa] = t / 40;
                }
                const double mb = static_cast<double>(std::min(ctx, sh.window > 0 ? sh.window : ctx)) * KH * (d + d / 8) * 2 / 1e6;   // K + V bytes read once (codes + scales)
                std::printf("    Q8 decode, ctx %6d: per-head %.3f ms | GQA %.3f ms | x%.2f | cache %.1f MB -> GQA reads at %.0f GB/s\n", ctx, ms[0], ms[1], ms[0] / ms[1], mb, mb / 1e3 / (ms[1] / 1e3));
            }
        }
        cudaFree(dK); cudaFree(dV); cudaFree(dKc); cudaFree(dVc); cudaFree(dKn); cudaFree(dVn); cudaFree(dq); cudaFree(dg); cudaFree(dout0); cudaFree(dout1); cudaFree(dscr); cudaFree(dpos);
    }
    attn_splitkv_set_gqa(-1);
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all GQA checks passed", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
