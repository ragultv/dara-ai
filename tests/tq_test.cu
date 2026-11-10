// TurboQuant KV kernels: (1) the codebook, (2) the quantiser round trip, (3) the attention kernel against a CPU implementation of the SAME
// quantiser (isolates kernel correctness from quantisation error), (4) accuracy vs the exact cache for information, (5) speed vs the FP32 kernel.
// Usage: tq_test
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "model/attention.h"
#include "model/attention_cuda.h"
#include "model/tq_kv.h"

using namespace dara::model;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("CUDA error %s at line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static int failures = 0;
static void check(bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what); if (!ok) ++failures; }

// ---- CPU implementation of the quantiser (same hash, same Hadamard, same boundaries as the device code) ----
static float sign_of(int i) {
    uint32_t h = static_cast<uint32_t>(i) * 0x9E3779B1u + 0x85EBCA6Bu;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    return (h & 1u) ? -1.0f : 1.0f;
}
static void fwht(std::vector<double>& s) {
    const size_t d = s.size();
    for (size_t h = 1; h < d; h <<= 1)
        for (size_t i = 0; i < d; i += 2 * h)
            for (size_t j = i; j < i + h; ++j) { const double a = s[j], b = s[j + h]; s[j] = a + b; s[j + h] = a - b; }
}
// quantise then dequantise one head vector: returns x_hat (the vector the cache represents)
static std::vector<double> tq_roundtrip(const float* x, int d) {
    const float* c = tq_centroids();
    const float* bnd = tq_boundaries();
    double n2 = 0;
    for (int i = 0; i < d; ++i) n2 += static_cast<double>(x[i]) * x[i];
    const double n = std::sqrt(n2);
    std::vector<double> z(d), out(d, 0.0);
    for (int i = 0; i < d; ++i) z[i] = n > 1e-30 ? x[i] / n * sign_of(i) : 0.0;
    fwht(z);
    std::vector<double> cv(d);
    for (int i = 0; i < d; ++i) { int k = 0; for (int b = 0; b < 15; ++b) k += z[i] > bnd[b]; cv[i] = c[k]; }
    fwht(cv);  // H c
    for (int i = 0; i < d; ++i) out[i] = n / d * sign_of(i) * cv[i];  // x_hat = n/d * D H c
    return out;
}

int main() {
    if (tq_init() != 0) { std::puts("tq_init failed"); return 1; }
    const float* c = tq_centroids();
    std::printf("Lloyd-Max 16-level codebook for N(0,1):");
    for (int i = 0; i < 16; ++i) std::printf(" %.4f", c[i]);
    std::printf("\n  quantiser MSE %.5f (theory for 4 bits: 0.00950)\n", tq_codebook_mse());
    check(std::fabs(tq_codebook_mse() - 0.0095) < 0.0004, "codebook MSE matches the known 4-bit Gaussian Lloyd-Max value");
    check(std::fabs(c[0] + c[15]) < 1e-4 && std::fabs(c[7] + c[8]) < 1e-4, "codebook is symmetric");

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    {   // quantiser round trip on vectors with per-channel scale variation (like real keys): expected relative error ~ sqrt(MSE) = 0.097
        double num = 0, den = 0;
        for (int d : {256, 512}) {
            std::vector<float> chan(d);
            for (auto& v : chan) v = std::exp(0.6f * nd(rng));
            for (int rep = 0; rep < 200; ++rep) {
                std::vector<float> x(d);
                for (int i = 0; i < d; ++i) x[i] = nd(rng) * chan[i];
                const auto xh = tq_roundtrip(x.data(), d);
                for (int i = 0; i < d; ++i) { const double e = xh[i] - x[i]; num += e * e; den += static_cast<double>(x[i]) * x[i]; }
            }
        }
        const double rel = std::sqrt(num / den);
        std::printf("  quantiser round trip relative L2 error %.4f (theory sqrt(MSE) = %.4f)\n", rel, std::sqrt(tq_codebook_mse()));
        check(rel > 0.08 && rel < 0.12, "round-trip error equals the codebook's theoretical value (rotation is orthogonal, no bias)");
    }

    struct Shape { const char* name; int heads, kv_heads, d, window; float scale; bool gate; };
    const Shape shapes[] = {
        {"Qwen 4B/9B   (16 heads, 4 kv, d=256, causal, output gate)", 16, 4, 256, 0, 1.0f / 16.0f, true},
        {"Gemma sliding (8 heads, 1 kv, d=256, window 512, scale 0.2)", 8, 1, 256, 512, 0.2f, false},
        {"Gemma global  (8 heads, 1 kv, d=512, causal, scale 0.2)", 8, 1, 512, 0, 0.2f, false},
    };
    for (const Shape& sh : shapes) {
        std::printf("%s\n", sh.name);
        const int N = 8192, d = sh.d, KH = sh.kv_heads, H = sh.heads;
        std::vector<float> chan(d);
        for (auto& v : chan) v = std::exp(0.6f * nd(rng));
        std::vector<float> K(static_cast<size_t>(N) * KH * d), V(K.size()), q(static_cast<size_t>(H) * d), gate(q.size());
        for (size_t i = 0; i < K.size(); ++i) { K[i] = nd(rng) * chan[i % d]; V[i] = nd(rng); }
        for (auto& v : q) v = nd(rng);
        for (auto& v : gate) v = nd(rng);
        std::vector<int64_t> pos(N);
        for (int i = 0; i < N; ++i) pos[i] = i;
        const int64_t last = N - 1;
        float *dK, *dV, *dq, *dg, *dout, *dKn, *dVn;
        uint8_t *dKc, *dVc;
        int64_t *dpos, *dlast;
        CK(cudaMalloc(&dK, K.size() * 4)); CK(cudaMalloc(&dV, V.size() * 4)); CK(cudaMalloc(&dq, q.size() * 4)); CK(cudaMalloc(&dg, gate.size() * 4));
        CK(cudaMalloc(&dout, q.size() * 4));
        CK(cudaMalloc(&dKc, static_cast<size_t>(N) * KH * d / 2)); CK(cudaMalloc(&dVc, static_cast<size_t>(N) * KH * d / 2));
        CK(cudaMalloc(&dKn, static_cast<size_t>(N) * KH * 4)); CK(cudaMalloc(&dVn, static_cast<size_t>(N) * KH * 4));
        CK(cudaMalloc(&dpos, N * 8)); CK(cudaMalloc(&dlast, 8));
        CK(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dpos, pos.data(), N * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dlast, &last, 8, cudaMemcpyHostToDevice));
        if (tq_append_cuda(dK, dpos, N, KH, d, dKc, dKn) || tq_append_cuda(dV, dpos, N, KH, d, dVc, dVn)) { std::puts("append failed"); return 1; }
        if (tq_attention_cuda(dq, sh.gate ? dg : nullptr, dKc, dKn, dVc, dVn, dlast, 1, H, KH, d, sh.window, sh.scale, std::min(N, sh.window ? sh.window : N), dout)) {
            std::puts("attention failed"); return 1;
        }
        CK(cudaDeviceSynchronize());
        std::vector<float> got(q.size());
        CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));

        // reference attention in double over (a) the dequantised cache the kernel should be using, (b) the exact cache
        const int p = N - 1, start = sh.window ? std::max(0, p - sh.window + 1) : 0;
        std::vector<double> Kd(static_cast<size_t>(p - start + 1) * KH * d), Vd(Kd.size());
        for (int j = start; j <= p; ++j)
            for (int kvh = 0; kvh < KH; ++kvh) {
                const size_t off = (static_cast<size_t>(j) * KH + kvh) * d, o2 = (static_cast<size_t>(j - start) * KH + kvh) * d;
                const auto kh = tq_roundtrip(&K[off], d), vh = tq_roundtrip(&V[off], d);
                for (int i = 0; i < d; ++i) { Kd[o2 + i] = kh[i]; Vd[o2 + i] = vh[i]; }
            }
        auto attend = [&](bool quantised, std::vector<double>& outv) {
            outv.assign(static_cast<size_t>(H) * d, 0.0);
            for (int h = 0; h < H; ++h) {
                const int kvh = h / (H / KH);
                std::vector<double> sc(p - start + 1);
                double mx = -1e300;
                for (int j = start; j <= p; ++j) {
                    double s = 0;
                    for (int i = 0; i < d; ++i)
                        s += static_cast<double>(q[h * d + i]) * (quantised ? Kd[(static_cast<size_t>(j - start) * KH + kvh) * d + i] : K[(static_cast<size_t>(j) * KH + kvh) * d + i]);
                    sc[j - start] = s * sh.scale;
                    mx = std::max(mx, sc[j - start]);
                }
                double sum = 0;
                for (auto& s : sc) { s = std::exp(s - mx); sum += s; }
                for (int i = 0; i < d; ++i) {
                    double o = 0;
                    for (int j = start; j <= p; ++j) o += sc[j - start] * (quantised ? Vd[(static_cast<size_t>(j - start) * KH + kvh) * d + i] : V[(static_cast<size_t>(j) * KH + kvh) * d + i]);
                    o /= sum;
                    if (sh.gate) o *= 1.0 / (1.0 + std::exp(-static_cast<double>(gate[h * d + i])));
                    outv[static_cast<size_t>(h) * d + i] = o;
                }
            }
        };
        std::vector<double> ref_q, ref_x;
        attend(true, ref_q);
        attend(false, ref_x);
        auto rel = [&](const std::vector<double>& ref) {
            double num = 0, den = 0;
            for (size_t i = 0; i < got.size(); ++i) { const double e = ref[i] - got[i]; num += e * e; den += ref[i] * ref[i]; }
            return std::sqrt(num / den);
        };
        const double r_impl = rel(ref_q), r_quant = rel(ref_x);
        std::printf("  kernel vs CPU TurboQuant reference (same quantiser): relative L2 %.2e\n", r_impl);
        std::printf("  kernel vs exact FP32 cache (information: includes the 4-bit quantisation error on random data): relative L2 %.4f\n", r_quant);
        check(r_impl < 1e-3, "attention kernel implements the intended quantised attention (rotated-domain math, softmax, gate)");

        if (sh.heads == 16) {   // the existing FP32 kernel is built for head_dim 256 with an output gate: the Qwen shape
            dara::model::AttentionDims ad;
            ad.hidden = 2560; ad.heads = H; ad.kv_heads = KH; ad.head_dim = d; ad.rot_dim = 64; ad.eps = 1e-6f; ad.rope_theta = 1e7;
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
            for (int depth : {512, 2048, 8192}) {
                const int64_t dm1 = depth - 1;
                CK(cudaMemcpy(dlast, &dm1, 8, cudaMemcpyHostToDevice));  // token at position depth-1: both kernels attend to `depth` positions
                float us_tq = 0, us_f32 = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    CK(cudaEventRecord(a));
                    for (int it = 0; it < 100; ++it) tq_attention_cuda(dq, dg, dKc, dKn, dVc, dVn, dlast, 1, H, KH, d, 0, sh.scale, depth, dout);
                    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; cudaEventElapsedTime(&ms, a, b); us_tq = ms * 1000 / 100;
                    CK(cudaEventRecord(a));
                    for (int it = 0; it < 100; ++it) attention_core_cuda(ad, dq, dg, dK, dV, depth - 1, 1, dout);
                    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b); us_f32 = ms * 1000 / 100;
                }
                std::printf("  depth %5d: FP32-KV kernel %7.1f us/layer, TurboQuant-4bit kernel %7.1f us/layer\n", depth, us_f32, us_tq);
            }
        }
        cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dg); cudaFree(dout); cudaFree(dKc); cudaFree(dVc); cudaFree(dKn); cudaFree(dVn); cudaFree(dpos); cudaFree(dlast);
    }
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
