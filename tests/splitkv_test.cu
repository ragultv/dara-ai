// Split-KV attention vs (1) a double-precision CPU reference (F32 caches), (2) the validated single-block TurboQuant kernel (TQ4 caches),
// including multi-token batches, windows, and a grid sized for a larger maximum than the current context; then speed vs the old kernels.
// Usage: splitkv_test
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "model/attention.h"
#include "model/attention_cuda.h"
#include "model/attn_splitkv.h"
#include "model/gemma_ops_cuda.h"
#include "model/tq_kv.h"

using namespace dara::model;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
static int failures = 0;
static void check(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what); if (!ok) ++failures; }

static double rel_l2(const std::vector<float>& a, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = ref[i] - a[i]; num += e * e; den += ref[i] * ref[i]; }
    return std::sqrt(num / den);
}
static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double e = static_cast<double>(b[i]) - a[i]; num += e * e; den += static_cast<double>(b[i]) * b[i]; }
    return std::sqrt(num / den);
}

int main() {
    if (tq_init() || attn_splitkv_init()) { std::puts("init failed"); return 1; }
    struct Shape { const char* name; int heads, kv_heads, d, window; float scale; bool gate; };
    const Shape shapes[] = {
        {"Qwen   (16 heads, 4 kv, d=256, causal, gate, scale 1/16)", 16, 4, 256, 0, 1.0f / 16.0f, true},
        {"Gemma sliding (8 heads, 1 kv, d=256, window 512)", 8, 1, 256, 512, 0.2f, false},
        {"Gemma global  (8 heads, 1 kv, d=512, causal)", 8, 1, 512, 0, 0.2f, false},
    };
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (const Shape& sh : shapes) {
        std::printf("%s\n", sh.name);
        const int N = 8192, d = sh.d, KH = sh.kv_heads, H = sh.heads, T = 8;
        std::vector<float> chan(d);
        for (auto& v : chan) v = std::exp(0.5f * nd(rng));
        std::vector<float> K(static_cast<size_t>(N) * KH * d), V(K.size()), q(static_cast<size_t>(T) * H * d), gate(q.size());
        for (size_t i = 0; i < K.size(); ++i) { K[i] = nd(rng) * chan[i % d]; V[i] = nd(rng); }
        for (auto& v : q) v = nd(rng);
        for (auto& v : gate) v = nd(rng);
        // T consecutive tokens ending at position p0+T-1 (a verification-sized batch)
        const int p0 = 5000;
        std::vector<int64_t> pos(T);
        for (int t = 0; t < T; ++t) pos[t] = p0 + t;
        float *dK, *dV, *dq, *dg, *dout, *dout2, *dKn, *dVn, *dscr;
        uint8_t *dKc, *dVc;
        int64_t* dpos;
        const size_t scr_floats = 4 * attn_splitkv_scratch_floats_per_token(AttnFmt::TQ4, H, d, N) + 1024;  // room for 4 tokens per launch group: forces batching
        CK(cudaMalloc(&dK, K.size() * 4)); CK(cudaMalloc(&dV, V.size() * 4)); CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dg, gate.size() * 4));
        CK(cudaMalloc(&dout, q.size() * 4)); CK(cudaMalloc(&dout2, q.size() * 4)); CK(cudaMalloc(&dscr, scr_floats * 4));
        CK(cudaMalloc(&dKc, static_cast<size_t>(N) * KH * d / 2)); CK(cudaMalloc(&dVc, static_cast<size_t>(N) * KH * d / 2));
        CK(cudaMalloc(&dKn, static_cast<size_t>(N) * KH * 4)); CK(cudaMalloc(&dVn, static_cast<size_t>(N) * KH * 4)); CK(cudaMalloc(&dpos, T * 8));
        CK(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), T * 8, cudaMemcpyHostToDevice));
        std::vector<int64_t> allpos(N);
        for (int i = 0; i < N; ++i) allpos[i] = i;
        int64_t* dall;
        CK(cudaMalloc(&dall, N * 8)); CK(cudaMemcpy(dall, allpos.data(), N * 8, cudaMemcpyHostToDevice));
        if (tq_append_cuda(dK, dall, N, KH, d, dKc, dKn) || tq_append_cuda(dV, dall, N, KH, d, dVc, dVn)) { std::puts("append failed"); return 1; }

        // ---- F32: vs CPU double reference (token 0 and token T-1)
        if (attn_splitkv_cuda(AttnFmt::F32, dq, sh.gate ? dg : nullptr, dK, nullptr, dV, nullptr, dpos, T, H, KH, d, sh.window, sh.scale, N, dscr, scr_floats, dout)) { std::puts("F32 launch failed"); return 1; }
        CK(cudaDeviceSynchronize());
        std::vector<float> got(q.size());
        CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
        auto ref_token = [&](int t, std::vector<double>& outv) {
            const int p = p0 + t, start = sh.window ? std::max(0, p - sh.window + 1) : 0;
            outv.assign(static_cast<size_t>(H) * d, 0.0);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / KH);
                std::vector<double> sc(p - start + 1);
                double mx = -1e300;
                for (int j = start; j <= p; ++j) {
                    double s = 0;
                    for (int i = 0; i < d; ++i) s += static_cast<double>(q[(static_cast<size_t>(t) * H + h) * d + i]) * K[(static_cast<size_t>(j) * KH + kvh) * d + i];
                    sc[j - start] = s * sh.scale; mx = std::max(mx, sc[j - start]);
                }
                double sum = 0;
                for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
                for (int i = 0; i < d; ++i) {
                    double o = 0;
                    for (int j = start; j <= p; ++j) o += sc[j - start] * V[(static_cast<size_t>(j) * KH + kvh) * d + i];
                    o /= sum;
                    if (sh.gate) o *= 1.0 / (1.0 + std::exp(-static_cast<double>(gate[(static_cast<size_t>(t) * H + 0) * d * 0 + (static_cast<size_t>(t) * H + h) * d + i])));
                    outv[static_cast<size_t>(h) * d + i] = o;
                }
            }
        };
        double worst = 0;
        for (int t : {0, T - 1}) {
            std::vector<double> ref;
            ref_token(t, ref);
            std::vector<float> g1(got.begin() + static_cast<long>(t) * H * d, got.begin() + static_cast<long>(t + 1) * H * d);
            worst = std::max(worst, rel_l2(g1, ref));
        }
        std::printf("  F32 split-KV vs CPU double reference: relative L2 %.2e (8-token batch, grid sized for %d positions, scratch forces 4-token groups)\n", worst, N);
        check(worst < 1e-5, "F32 split-KV matches the double-precision reference");

        // ---- TQ4: vs the validated single-block TurboQuant kernel
        if (attn_splitkv_cuda(AttnFmt::TQ4, dq, sh.gate ? dg : nullptr, dKc, dKn, dVc, dVn, dpos, T, H, KH, d, sh.window, sh.scale, N, dscr, scr_floats, dout)) { std::puts("TQ launch failed"); return 1; }
        if (tq_attention_cuda(dq, sh.gate ? dg : nullptr, dKc, dKn, dVc, dVn, dpos, T, H, KH, d, sh.window, sh.scale, std::min(N, sh.window ? sh.window : p0 + T), dout2)) { std::puts("old TQ launch failed"); return 1; }
        CK(cudaDeviceSynchronize());
        std::vector<float> a(q.size()), b(q.size());
        CK(cudaMemcpy(a.data(), dout, a.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(b.data(), dout2, b.size() * 4, cudaMemcpyDeviceToHost));
        const double r_tq = rel_l2(a, b);
        std::printf("  TQ4 split-KV vs single-block TurboQuant kernel: relative L2 %.2e\n", r_tq);
        check(r_tq < 1e-5, "TQ4 split-KV agrees with the validated single-block kernel");

        // ---- speed (decode: 1 token), old kernels vs split-KV
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::printf("  decode attention, us per layer (1 token):  depth | old FP32 | split FP32 | old TQ4 | split TQ4\n");
        for (int depth : {512, 2048, 8192}) {
            if (sh.window && depth > 1024 && depth != 2048) continue;
            const int64_t pdepth = depth - 1;
            CK(cudaMemcpy(dpos, &pdepth, 8, cudaMemcpyHostToDevice));
            auto time_it = [&](auto&& fn) {
                float us = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    cudaEventRecord(e0);
                    for (int it = 0; it < 100; ++it) fn();
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1); us = ms * 10;
                }
                return us;
            };
            dara::model::AttentionDims ad; ad.hidden = 2560; ad.heads = H; ad.kv_heads = KH; ad.head_dim = d;
            const int win_pos = sh.window ? std::min(depth, sh.window) : depth;
            const float old_f32 = sh.heads == 16
                ? time_it([&] { attention_core_cuda(ad, dq, dg, dK, dV, depth - 1, 1, dout2); })
                : time_it([&] { gemma_attention_cuda(dq, dK, dV, dpos, 1, H, KH, d, sh.window, win_pos, dout2); });
            const float new_f32 = time_it([&] { attn_splitkv_cuda(AttnFmt::F32, dq, sh.gate ? dg : nullptr, dK, nullptr, dV, nullptr, dpos, 1, H, KH, d, sh.window, sh.scale, N, dscr, scr_floats, dout); });
            const float old_tq = time_it([&] { tq_attention_cuda(dq, sh.gate ? dg : nullptr, dKc, dKn, dVc, dVn, dpos, 1, H, KH, d, sh.window, sh.scale, win_pos, dout2); });
            const float new_tq = time_it([&] { attn_splitkv_cuda(AttnFmt::TQ4, dq, sh.gate ? dg : nullptr, dKc, dKn, dVc, dVn, dpos, 1, H, KH, d, sh.window, sh.scale, N, dscr, scr_floats, dout); });
            std::printf("                                          %5d | %8.1f | %10.1f | %7.1f | %9.1f\n", depth, old_f32, new_f32, old_tq, new_tq);
        }
        cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dg); cudaFree(dout); cudaFree(dout2); cudaFree(dscr); cudaFree(dKc); cudaFree(dVc);
        cudaFree(dKn); cudaFree(dVn); cudaFree(dpos); cudaFree(dall);
    }
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
