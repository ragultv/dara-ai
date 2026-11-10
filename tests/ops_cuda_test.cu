// CUDA RMSNorm and RoPE vs the transformers reference (golden data) and the validated CPU implementation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "model/ops.h"
#include "model/ops_cuda.h"
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
    explicit Dev(const std::vector<T>& v) : n(v.size()) {
        CUDA(cudaMalloc(&p, n * sizeof(T)));
        CUDA(cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    std::vector<T> get() const { std::vector<T> v(n); CUDA(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v; }
};

static void test_rmsnorm(std::map<std::string, GoldenTensor>& g, const char* tag) {
    const std::string t = tag;
    const auto scale = fold_plus_one(g["norm_" + t + ".w"].f);
    const auto& X = g["norm_" + t + ".x"];
    const auto& Y = g["norm_" + t + ".y"].f;
    const int cases = static_cast<int>(X.dims[0]), n = static_cast<int>(X.dims[1]);
    Dev<float> dx(X.f), ds(scale), dy(std::vector<float>(X.f.size()));
    CHECK(rmsnorm_cuda(dx.p, ds.p, cases, n, 1e-6f, dy.p) == 0);
    CUDA(cudaDeviceSynchronize());
    const auto y = dy.get();
    double vs_ref = 0, vs_cpu = 0;
    std::vector<float> cpu(n);
    for (int c = 0; c < cases; ++c) {
        rmsnorm(&X.f[static_cast<size_t>(c) * n], scale.data(), n, 1e-6f, cpu.data());
        for (int i = 0; i < n; ++i) {
            const double ref = Y[static_cast<size_t>(c) * n + i], got = y[static_cast<size_t>(c) * n + i];
            CHECK(std::isfinite(got));
            if (ref == 0.0) { CHECK(got == 0.0); continue; }
            vs_ref = std::max(vs_ref, std::fabs(got - ref) / std::fabs(ref));
            vs_cpu = std::max(vs_cpu, std::fabs(got - cpu[i]) / std::max(std::fabs(double(cpu[i])), 1e-30));
        }
    }
    std::printf("  RMSNorm %-6s n=%d x %d cases on GPU: worst elementwise rel err vs reference %.2e, vs CPU %.2e\n", tag, n, cases, vs_ref, vs_cpu);
    CHECK(vs_ref < 2e-6 && vs_cpu < 2e-6);
    // in-place (x and y alias)
    Dev<float> dz(X.f);
    CHECK(rmsnorm_cuda(dz.p, ds.p, cases, n, 1e-6f, dz.p) == 0);
    CUDA(cudaDeviceSynchronize());
    CHECK(dz.get() == y);
}

static void test_rope(std::map<std::string, GoldenTensor>& g) {
    const auto& posf = g["rope.positions"].f;
    const int T = static_cast<int>(posf.size()), rot = 64, hd = 256, QH = 16, KH = 4;
    std::vector<int64_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = static_cast<int64_t>(posf[t]);
    const std::vector<float> inv = rope_inv_freq(rot, 1e7);
    Dev<int64_t> dp(pos);
    Dev<float> di(inv), dq(g["rope.q"].f), dk(g["rope.k"].f);
    CHECK(rope_cuda(dq.p, T, QH, hd, rot, dp.p, di.p) == 0);
    CHECK(rope_cuda(dk.p, T, KH, hd, rot, dp.p, di.p) == 0);
    CUDA(cudaDeviceSynchronize());
    const auto q = dq.get(), k = dk.get();
    double eq = 0, ek = 0, cq = 0;
    for (size_t i = 0; i < q.size(); ++i) eq = std::max(eq, std::fabs(double(q[i]) - g["rope.q_rot"].f[i]));
    for (size_t i = 0; i < k.size(); ++i) ek = std::max(ek, std::fabs(double(k[i]) - g["rope.k_rot"].f[i]));
    // vs the CPU implementation
    std::vector<float> cs(rot / 2), sn(rot / 2);
    for (int t = 0; t < T; ++t) {
        std::vector<float> qq(g["rope.q"].f.begin() + static_cast<long>(t) * QH * hd, g["rope.q"].f.begin() + static_cast<long>(t + 1) * QH * hd);
        rope_cos_sin(pos[t], inv, cs.data(), sn.data());
        rope_apply(qq.data(), QH, hd, rot, cs.data(), sn.data());
        for (size_t i = 0; i < qq.size(); ++i) cq = std::max(cq, std::fabs(double(q[static_cast<size_t>(t) * QH * hd + i]) - qq[i]));
    }
    std::printf("  RoPE on GPU, %d positions up to %lld: max abs err vs reference q %.2e, k %.2e; vs CPU %.2e\n", T,
                static_cast<long long>(pos.back()), eq, ek, cq);
    CHECK(eq < 5e-6 && ek < 5e-6 && cq < 5e-6);
    // partial dims untouched, position 0 identity
    bool untouched = true;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < QH; ++h)
            for (int d = rot; d < hd; ++d) untouched &= q[(static_cast<size_t>(t) * QH + h) * hd + d] == g["rope.q"].f[(static_cast<size_t>(t) * QH + h) * hd + d];
    CHECK(untouched);
    bool ident = true;
    for (int i = 0; i < QH * hd; ++i) ident &= q[i] == g["rope.q"].f[i];  // token 0 is position 0
    CHECK(ident);
    // API validation
    CHECK(rope_cuda(dq.p, T, QH, hd, 63, dp.p, di.p) != 0);       // odd rotary dim
    CHECK(rope_cuda(dq.p, T, QH, hd, 512, dp.p, di.p) != 0);      // rot > head_dim
    CHECK(rmsnorm_cuda(nullptr, nullptr, 1, 8, 1e-6f, nullptr) != 0);
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/ops.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (!n) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    test_rmsnorm(g, "hidden");
    test_rmsnorm(g, "head");
    test_rope(g);
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
