// DeltaNet decode step on the GPU vs the transformers reference (golden data) and the validated CPU path.
// Requires build/golden/deltanet_layer0.bin. The projections (GEMVs) are computed by the CPU path; this tests
// everything the GPU kernels own: conv + state, gates, l2norm, recurrence, gated RMSNorm, V-head order, and
// device-side checkpoint/restore.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/gpu_telemetry.h"
#include "model/deltanet.h"
#include "model/deltanet_cuda.h"
#include "support/golden.h"

using namespace dara::model;
using dara::test::GoldenTensor;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <class T> struct Dev {
    T* p = nullptr;
    size_t n;
    explicit Dev(size_t n_) : n(n_) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : n(v.size()) {
        CUDA(cudaMalloc(&p, n * sizeof(T)));
        CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const { std::vector<T> v(n); CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0, s = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - b[i])); s = std::max(s, std::fabs(double(b[i]))); }
    return m / s;
}

static DeltaNetWeights weights_from_golden(std::map<std::string, GoldenTensor>& g) {
    DeltaNetWeights w;
    w.order = VHeadOrder::Checkpoint;
    w.in_qkv = g["w.in_qkv"].f; w.in_z = g["w.in_z"].f; w.in_b = g["w.in_b"].f; w.in_a = g["w.in_a"].f;
    w.conv_w = g["w.conv"].f; w.dt_bias = g["w.dt_bias"].f; w.norm_w = g["w.norm"].f; w.out = g["w.out"].f;
    const auto& A = g["w.A_log"].f;
    w.ssm_a.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) w.ssm_a[i] = -std::exp(A[i]);
    return w;
}

struct Projections {  // per token, computed once on the CPU for a given weight order
    std::vector<float> qkv, z, b, a;
};

static Projections project_all(const DeltaNetWeights& w, const std::vector<float>& x, int T) {
    const DeltaNetDims& d = w.dims;
    Projections p;
    p.qkv.resize(static_cast<size_t>(T) * d.conv_dim()); p.z.resize(static_cast<size_t>(T) * d.value_dim());
    p.b.resize(static_cast<size_t>(T) * d.v_heads); p.a.resize(p.b.size());
    for (int t = 0; t < T; ++t)
        deltanet_project(w, x.data() + static_cast<size_t>(t) * d.hidden, &p.qkv[static_cast<size_t>(t) * d.conv_dim()],
                         &p.z[static_cast<size_t>(t) * d.value_dim()], &p.b[static_cast<size_t>(t) * d.v_heads],
                         &p.a[static_cast<size_t>(t) * d.v_heads]);
    return p;
}

// Device-resident layer: small params + state + per-token buffers.
struct GpuLayer {
    DeltaNetDims d;
    VHeadOrder order;
    Dev<float> conv_w, ssm_a, dt_bias, norm_w, conv_state, S, scratch, qkv, z, b, a, normed;
    explicit GpuLayer(const DeltaNetWeights& w)
        : d(w.dims), order(w.order), conv_w(w.conv_w), ssm_a(w.ssm_a), dt_bias(w.dt_bias), norm_w(w.norm_w),
          conv_state(deltanet_conv_state_floats(w.dims)), S(deltanet_S_floats(w.dims)), scratch(w.dims.conv_dim()),
          qkv(w.dims.conv_dim()), z(w.dims.value_dim()), b(w.dims.v_heads), a(w.dims.v_heads), normed(w.dims.value_dim()) {
        reset();
    }
    void reset() {
        CUDA(cudaMemset(conv_state.p, 0, conv_state.n * 4));
        CUDA(cudaMemset(S.p, 0, S.n * 4));
    }
    // One token from precomputed projections; returns the normed vector.
    std::vector<float> step(const Projections& pr, int t) {
        CUDA(cudaMemcpy(qkv.p, &pr.qkv[static_cast<size_t>(t) * d.conv_dim()], qkv.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(z.p, &pr.z[static_cast<size_t>(t) * d.value_dim()], z.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(b.p, &pr.b[static_cast<size_t>(t) * d.v_heads], b.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(a.p, &pr.a[static_cast<size_t>(t) * d.v_heads], a.n * 4, cudaMemcpyHostToDevice));
        DeltaNetDeviceParams p{conv_w.p, ssm_a.p, dt_bias.p, norm_w.p};
        const int rc = deltanet_step_cuda(d, order, p, conv_state.p, S.p, qkv.p, z.p, b.p, a.p, scratch.p, normed.p);
        CHECK(rc == 0);
        CUDA(cudaDeviceSynchronize());
        return normed.get();
    }
};

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/deltanet_layer0.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (!n) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));

    const int T = static_cast<int>(g["x"].dims[0]);
    const std::vector<float> x = g["x"].f;
    DeltaNetWeights w = weights_from_golden(g);
    const DeltaNetDims d = w.dims;
    std::printf("device: %s sm_%d%d | DeltaNet layer 0, T=%d\n", prop.name, prop.major, prop.minor, T);

    // ---- checkpoint order: stage outputs vs the transformers reference ------------------------------
    const Projections pr = project_all(w, x, T);
    GpuLayer gpu(w);
    std::vector<float> normed(static_cast<size_t>(T) * d.value_dim());
    for (int t = 0; t < T; ++t) {
        auto nt = gpu.step(pr, t);
        std::copy(nt.begin(), nt.end(), normed.begin() + static_cast<long>(t) * d.value_dim());
    }
    const double e_norm = rel_err(normed, g["normed"].f);
    const double e_S = rel_err(gpu.S.get(), g["final_state"].f);
    const double e_conv = rel_err(gpu.conv_state.get(), g["conv_state"].f);
    std::vector<float> out(static_cast<size_t>(T) * d.hidden);
    for (int t = 0; t < T; ++t)
        deltanet_out_project(w, &normed[static_cast<size_t>(t) * d.value_dim()], &out[static_cast<size_t>(t) * d.hidden]);
    const double e_out = rel_err(out, g["out"].f);
    std::printf("  GPU vs reference: gated-norm output %.3e | final state %.3e | conv state %.3e | layer output %.3e\n",
                e_norm, e_S, e_conv, e_out);
    CHECK(e_norm < 2e-4 && e_S < 2e-4 && e_conv < 2e-4 && e_out < 2e-4);
    const std::vector<float> S_ckpt_order = gpu.S.get();

    // ---- device-side checkpoint / restore ----------------------------------------------------------
    {
        const int split = 40;
        gpu.reset();
        for (int t = 0; t < split; ++t) gpu.step(pr, t);
        Dev<float> ck_conv(gpu.conv_state.n), ck_S(gpu.S.n);   // the layer checkpoint = two buffer copies
        CUDA(cudaMemcpy(ck_conv.p, gpu.conv_state.p, ck_conv.n * 4, cudaMemcpyDeviceToDevice));
        CUDA(cudaMemcpy(ck_S.p, gpu.S.p, ck_S.n * 4, cudaMemcpyDeviceToDevice));
        std::vector<float> first;
        for (int t = split; t < T; ++t) { auto v = gpu.step(pr, t); first.insert(first.end(), v.begin(), v.end()); }
        const auto S_first = gpu.S.get();
        CUDA(cudaMemcpy(gpu.conv_state.p, ck_conv.p, ck_conv.n * 4, cudaMemcpyDeviceToDevice));  // restore
        CUDA(cudaMemcpy(gpu.S.p, ck_S.p, ck_S.n * 4, cudaMemcpyDeviceToDevice));
        std::vector<float> again;
        for (int t = split; t < T; ++t) { auto v = gpu.step(pr, t); again.insert(again.end(), v.begin(), v.end()); }
        CHECK(first == again);                 // deterministic kernels: bitwise identical after restore
        CHECK(S_first == gpu.S.get());
        std::printf("  checkpoint at token %d, restore, continue: bitwise identical (%zu floats + state)\n", split, first.size());
    }

    // ---- V-head order on the GPU ------------------------------------------------------------------
    {
        DeltaNetWeights wg = reorder_v_heads(w, VHeadOrder::Gguf);
        const Projections pg = project_all(wg, x, T);  // projections from the GGUF-ordered weights
        GpuLayer gg(wg);
        std::vector<float> og(normed.size());
        for (int t = 0; t < T; ++t) {
            auto nt = gg.step(pg, t);
            std::copy(nt.begin(), nt.end(), og.begin() + static_cast<long>(t) * d.value_dim());
        }
        // GGUF head j holds checkpoint head checkpoint_head_of_gguf(j): compare per head against checkpoint order.
        double e = 0, s = 1e-30;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < d.v_heads; ++j) {
                const int h = checkpoint_head_of_gguf(j, d);
                for (int i = 0; i < d.v_dim; ++i) {
                    const double a = og[(static_cast<size_t>(t) * d.v_heads + j) * d.v_dim + i];
                    const double b = normed[(static_cast<size_t>(t) * d.v_heads + h) * d.v_dim + i];
                    e = std::max(e, std::fabs(a - b)); s = std::max(s, std::fabs(b));
                }
            }
        std::printf("  GGUF-order weights + GGUF mapping on GPU vs checkpoint order (per head): %.3e\n", e / s);
        CHECK(e / s < 1e-5);

        GpuLayer wrong(wg);
        wrong.order = VHeadOrder::Checkpoint;  // GGUF-ordered weights, wrong mapping
        double ew = 0;
        for (int t = 0; t < 8; ++t) {
            auto nt = wrong.step(pg, t);
            for (int j = 0; j < d.v_heads; ++j)
                for (int i = 0; i < d.v_dim; ++i) {
                    const double a = nt[static_cast<size_t>(j) * d.v_dim + i];
                    const double b = normed[(static_cast<size_t>(t) * d.v_heads + checkpoint_head_of_gguf(j, d)) * d.v_dim + i];
                    ew = std::max(ew, std::fabs(a - b));
                }
        }
        std::printf("  GGUF-order weights + WRONG mapping: max abs diff %.3e (must be large)\n", ew);
        CHECK(ew > 1e-2);
    }

    // ---- cost of one step ---------------------------------------------------------------------------
    {
        gpu.reset();
        const int iters = 400;
        // stage inputs once; time only the kernels
        CUDA(cudaMemcpy(gpu.qkv.p, pr.qkv.data(), gpu.qkv.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(gpu.z.p, pr.z.data(), gpu.z.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(gpu.b.p, pr.b.data(), gpu.b.n * 4, cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(gpu.a.p, pr.a.data(), gpu.a.n * 4, cudaMemcpyHostToDevice));
        DeltaNetDeviceParams p{gpu.conv_w.p, gpu.ssm_a.p, gpu.dt_bias.p, gpu.norm_w.p};
        // Warm up for ~600 ms so the GPU leaves its idle clocks before anything is timed.
        {
            cudaEvent_t w0, w1;
            CUDA(cudaEventCreate(&w0)); CUDA(cudaEventCreate(&w1));
            float elapsed = 0;
            CUDA(cudaEventRecord(w0));
            while (elapsed < 600.0f) {
                for (int i = 0; i < 100; ++i) deltanet_step_cuda(d, w.order, p, gpu.conv_state.p, gpu.S.p, gpu.qkv.p, gpu.z.p, gpu.b.p, gpu.a.p, gpu.scratch.p, gpu.normed.p);
                CUDA(cudaEventRecord(w1)); CUDA(cudaEventSynchronize(w1));
                CUDA(cudaEventElapsedTime(&elapsed, w0, w1));
            }
        }
        cudaEvent_t e0, e1;
        CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
        CUDA(cudaEventRecord(e0));
        for (int i = 0; i < iters; ++i) deltanet_step_cuda(d, w.order, p, gpu.conv_state.p, gpu.S.p, gpu.qkv.p, gpu.z.p, gpu.b.p, gpu.a.p, gpu.scratch.p, gpu.normed.p);
        CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1));
        float ms = 0; CUDA(cudaEventElapsedTime(&ms, e0, e1));
        const double us = ms * 1000.0 / iters;
        const double bytes = 2.0 * deltanet_S_floats(d) * 4 + 2.0 * deltanet_conv_state_floats(d) * 4;  // state read + write
        std::printf("  one layer, one token (conv + recurrence + gated norm): %.1f us  (state traffic %.1f MiB -> %.0f GB/s)\n",
                    us, bytes / 1048576.0, bytes / (us * 1e-6) / 1e9);
        if (auto t = dara::core::query_gpu_telemetry())
            std::printf("  conditions: SM %u/%u MHz, %u C, %.0f W, throttle: %s\n", t->sm_mhz, t->max_sm_mhz, t->temp_c, t->power_w, t->throttle_text().c_str());
        std::printf("  x 24 DeltaNet layers = %.2f ms per token (state update only; GPU under whatever clocks it has now)\n", us * 24 / 1000.0);
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
