// Full-attention layer on the GPU vs the transformers reference (golden) and the validated CPU path.
// Projections use the existing BF16 GEMV kernel (the checkpoint's native weights); everything else is the new
// attention path: q|gate split, per-head RMSNorm, RoPE, KV append, causal GQA attention with the fused gate.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/gpu_telemetry.h"
#include "kernels/gemv.h"
#include "kernels/half.h"
#include "model/attention.h"
#include "model/attention_cuda.h"
#include "model/ops.h"
#include "model/ops_cuda.h"
#include "support/golden.h"

using namespace dara::model;
using namespace dara::kernels;
using dara::test::GoldenTensor;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <class T> struct Dev {
    T* p = nullptr;
    size_t n;
    explicit Dev(size_t n_) : n(n_) { CUDA(cudaMalloc(&p, n * sizeof(T))); CUDA(cudaMemset(p, 0, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : n(v.size()) {
        CUDA(cudaMalloc(&p, n * sizeof(T)));
        CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void put(const T* src, size_t count, size_t off = 0) { CUDA(cudaMemcpy(p + off, src, count * sizeof(T), cudaMemcpyHostToDevice)); }
    std::vector<T> get(size_t count, size_t off = 0) const { std::vector<T> v(count); CUDA(cudaMemcpy(v.data(), p + off, count * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0, s = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - b[i])); s = std::max(s, std::fabs(double(b[i]))); }
    return m / s;
}

static std::vector<uint16_t> to_bf16(const std::vector<float>& f) {
    std::vector<uint16_t> u(f.size());
    for (size_t i = 0; i < f.size(); ++i) u[i] = float_to_bf16(f[i]);  // exact: values came from BF16
    return u;
}

constexpr int kCap = 512;

struct GpuAttention {
    AttentionDims d;
    Dev<uint16_t> Wq, Wk, Wv, Wo;
    Dev<float> qn, kn, inv, Kc, Vc;
    size_t len = 0;  // cache position counter (the whole cache "state")
    GpuAttention(AttentionWeights& w, std::map<std::string, GoldenTensor>& g)
        : d(w.dims), Wq(to_bf16(g["w.q_proj"].f)), Wk(to_bf16(g["w.k_proj"].f)), Wv(to_bf16(g["w.v_proj"].f)), Wo(to_bf16(g["w.o_proj"].f)),
          qn(w.q_norm), kn(w.k_norm), inv(w.inv_freq), Kc(static_cast<size_t>(kCap) * w.dims.kv_width()), Vc(static_cast<size_t>(kCap) * w.dims.kv_width()) {}

    // Processes T tokens at positions len .. len+T-1; returns out [T][hidden]; optionally the gated vector.
    std::vector<float> run(const float* x, int T, std::vector<float>* gated_out = nullptr) {
        const int H = d.heads, KH = d.kv_heads, hd = d.head_dim;
        Dev<float> dx(static_cast<size_t>(T) * d.hidden), qg(static_cast<size_t>(T) * d.q_rows()), k(static_cast<size_t>(T) * d.kv_width()),
            v(static_cast<size_t>(T) * d.kv_width()), q(static_cast<size_t>(T) * H * hd), gate(static_cast<size_t>(T) * d.out_width()),
            att(static_cast<size_t>(T) * d.out_width()), out(static_cast<size_t>(T) * d.hidden);
        dx.put(x, dx.n);
        for (int t = 0; t < T; ++t) {
            const float* xt = dx.p + static_cast<size_t>(t) * d.hidden;
            CHECK(gemv_cuda(WeightType::BF16, Wq.p, d.hidden, xt, qg.p + static_cast<size_t>(t) * d.q_rows(), d.q_rows(), d.hidden) == 0);
            CHECK(gemv_cuda(WeightType::BF16, Wk.p, d.hidden, xt, k.p + static_cast<size_t>(t) * d.kv_width(), d.kv_width(), d.hidden) == 0);
            CHECK(gemv_cuda(WeightType::BF16, Wv.p, d.hidden, xt, v.p + static_cast<size_t>(t) * d.kv_width(), d.kv_width(), d.hidden) == 0);
        }
        CHECK(attn_split_qgate_cuda(d, qg.p, T, q.p, gate.p) == 0);
        CHECK(rmsnorm_cuda(q.p, qn.p, T * H, hd, d.eps, q.p) == 0);
        CHECK(rmsnorm_cuda(k.p, kn.p, T * KH, hd, d.eps, k.p) == 0);
        std::vector<int64_t> pos(T);
        for (int t = 0; t < T; ++t) pos[t] = static_cast<int64_t>(len) + t;
        Dev<int64_t> dpos(pos);
        CHECK(rope_cuda(q.p, T, H, hd, d.rot_dim, dpos.p, inv.p) == 0);
        CHECK(rope_cuda(k.p, T, KH, hd, d.rot_dim, dpos.p, inv.p) == 0);
        CHECK(attn_kv_append_cuda(d, Kc.p, Vc.p, k.p, v.p, static_cast<int>(len), T) == 0);
        CHECK(attention_core_cuda(d, q.p, gate.p, Kc.p, Vc.p, static_cast<int>(len), T, att.p) == 0);
        for (int t = 0; t < T; ++t)
            CHECK(gemv_cuda(WeightType::BF16, Wo.p, d.out_width(), att.p + static_cast<size_t>(t) * d.out_width(), out.p + static_cast<size_t>(t) * d.hidden, d.hidden, d.out_width()) == 0);
        CUDA(cudaDeviceSynchronize());
        len += T;
        if (gated_out) *gated_out = att.get(att.n);
        return out.get(out.n);
    }
};

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/attention_layer3.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (!n) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));

    AttentionWeights w;
    w.q_proj = g["w.q_proj"].f; w.k_proj = g["w.k_proj"].f; w.v_proj = g["w.v_proj"].f; w.o_proj = g["w.o_proj"].f;
    w.q_norm = fold_plus_one(g["w.q_norm"].f);
    w.k_norm = fold_plus_one(g["w.k_norm"].f);
    w.finalize();
    const AttentionDims d = w.dims;
    const int T = static_cast<int>(g["x"].dims[0]);
    const std::vector<float>& x = g["x"].f;
    std::printf("device: %s sm_%d%d | attention layer 3, T=%d\n", prop.name, prop.major, prop.minor, T);
    const double tol = 2e-4;

    // ---- (b) one-shot prefill vs the reference -----------------------------------------------------
    GpuAttention a(w, g);
    std::vector<float> gated;
    const auto out_b = a.run(x.data(), T, &gated);
    const double e_out = rel_err(out_b, g["out"].f), e_gated = rel_err(gated, g["gated"].f);
    std::printf("  one-shot prefill (%d queries): gated attention %.3e | layer output %.3e (vs reference)\n", T, e_gated, e_out);
    CHECK(e_out < tol && e_gated < tol);

    // ---- (a) token-by-token decode, (c) chunked prefill: must equal one-shot BITWISE ------------------
    {
        GpuAttention s(w, g);
        std::vector<float> out_a;
        for (int t = 0; t < T; ++t) { auto o = s.run(x.data() + static_cast<size_t>(t) * d.hidden, 1); out_a.insert(out_a.end(), o.begin(), o.end()); }
        CHECK(out_a == out_b);
        std::printf("  %d single-token decode steps: bitwise identical to one-shot prefill\n", T);

        GpuAttention c(w, g);
        std::vector<float> out_c;
        int pos = 0;
        for (int nn : {1, 17, 52}) { auto o = c.run(x.data() + static_cast<size_t>(pos) * d.hidden, nn); out_c.insert(out_c.end(), o.begin(), o.end()); pos += nn; }
        CHECK(pos == T && out_c == out_b);
        std::printf("  chunked prefill 1+17+52: bitwise identical to one-shot prefill\n");

        // prefill 40 then decode the rest
        GpuAttention p(w, g);
        auto o1 = p.run(x.data(), 40);
        std::vector<float> out_p = o1;
        for (int t = 40; t < T; ++t) { auto o = p.run(x.data() + static_cast<size_t>(t) * d.hidden, 1); out_p.insert(out_p.end(), o.begin(), o.end()); }
        CHECK(out_p == out_b);
        std::printf("  prefill 40 + 30 decode steps: bitwise identical, and within %.1e of the reference\n", rel_err(out_p, g["out"].f));
    }

    // ---- (d) KV checkpoint / restore = move the position counter back ------------------------------
    {
        const size_t mark = 40;
        a.len = mark;  // restore: later cache entries are overwritten by the continuation
        const auto again = a.run(x.data() + mark * d.hidden, T - static_cast<int>(mark));
        std::vector<float> tail(out_b.begin() + static_cast<long>(mark) * d.hidden, out_b.end());
        CHECK(again == tail);
        std::printf("  KV restore to position %zu + re-run tokens %zu..%d: bitwise identical\n", mark, mark, T - 1);
    }

    // ---- limits and validation -------------------------------------------------------------------
    {
        Dev<float> q(static_cast<size_t>(d.heads) * d.head_dim), gt(static_cast<size_t>(d.out_width())), o(static_cast<size_t>(d.out_width()));
        Dev<float> K(static_cast<size_t>(16) * d.kv_width()), V(static_cast<size_t>(16) * d.kv_width());
        CHECK(attention_core_cuda(d, q.p, gt.p, K.p, V.p, kAttentionMaxKvLen, 1, o.p) != 0);  // beyond the smem limit: refused
        CHECK(attention_core_cuda(d, nullptr, gt.p, K.p, V.p, 0, 1, o.p) != 0);
        AttentionDims bad = d; bad.head_dim = 128;
        CHECK(attention_core_cuda(bad, q.p, gt.p, K.p, V.p, 0, 1, o.p) != 0);
        CUDA(cudaDeviceSynchronize());
    }

    // ---- cost of one decode step's attention core at several context lengths -------------------------
    {
        const int maxlen = kAttentionMaxKvLen - 1;
        std::vector<float> hostK(static_cast<size_t>(maxlen) * d.kv_width()), hostV(hostK.size());
        for (size_t i = 0; i < hostK.size(); ++i) { hostK[i] = std::sin(0.001f * i); hostV[i] = std::cos(0.002f * i); }
        Dev<float> K(hostK), V(hostV), q(static_cast<size_t>(d.heads) * d.head_dim), gt(static_cast<size_t>(d.out_width())), o(static_cast<size_t>(d.out_width()));
        std::vector<float> hq(q.n, 0.1f);
        q.put(hq.data(), hq.size());
        // warm up the GPU for ~500 ms
        cudaEvent_t e0, e1;
        CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
        float el = 0;
        CUDA(cudaEventRecord(e0));
        while (el < 500.0f) {
            for (int i = 0; i < 50; ++i) attention_core_cuda(d, q.p, gt.p, K.p, V.p, 4095, 1, o.p);
            CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1)); CUDA(cudaEventElapsedTime(&el, e0, e1));
        }
        std::printf("  decode attention core (16 heads, 1 query), one layer, FP32 KV:\n");
        for (int ctx : {256, 1024, 4096, 8192, maxlen}) {
            const int iters = 100;
            CUDA(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) attention_core_cuda(d, q.p, gt.p, K.p, V.p, ctx - 1, 1, o.p);
            CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1));
            float ms = 0; CUDA(cudaEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0 / iters;
            const double kv_bytes = 2.0 * ctx * d.kv_width() * 4;  // K and V read once per kv head group... (lower bound)
            std::printf("    context %5d: %8.1f us  (cache %.1f MiB -> %.0f GB/s lower-bound traffic)\n", ctx, us, kv_bytes / 1048576.0, kv_bytes / (us * 1e-6) / 1e9);
        }
        if (auto t = dara::core::query_gpu_telemetry())
            std::printf("    conditions: SM %u/%u MHz, %u C, %.0f W, throttle: %s\n", t->sm_mhz, t->max_sm_mhz, t->temp_c, t->power_w, t->throttle_text().c_str());
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
