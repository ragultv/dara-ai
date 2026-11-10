// Decode GEMV: correctness against CPU references, then a latency/bandwidth baseline.
// Shapes come from the Qwen3.5-4B manifest (real tensor headers), plus awkward shapes.
//
// Error bound: the GPU sums each row as 32 strided lane partials (<= cols/32 terms each) and a
// 5-level tree, so |err| <= (cols/32 + 8) * u * sum|w*x| with u = 2^-24. The naive sequential CPU
// float sum is bounded by cols * u * sum|w*x|. Both are checked against a double-precision truth.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/manifest.h"
#include "kernels/gemv.h"

using namespace dara::kernels;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static constexpr double kU = 5.960464477539063e-8;  // 2^-24

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    float uni() {  // [-1, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<float>(static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0);
    }
};

enum class Fill { Random, ZeroX, ZeroW, Cancel, WideRange };
static const char* name(Fill f) {
    switch (f) {
        case Fill::Random: return "random"; case Fill::ZeroX: return "zero-x";
        case Fill::ZeroW: return "zero-w"; case Fill::Cancel: return "cancel";
        case Fill::WideRange: return "wide-range";
    }
    return "?";
}

template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Result { double max_gpu_ratio = 0, max_cpu_ratio = 0; bool ok = true; };

// ratio = |error| / bound; <= 1 passes.
static Result run_case(WeightType t, size_t rows, size_t cols, size_t ld, size_t base_off, Fill fill, uint64_t seed) {
    Rng rng(seed);
    std::vector<uint16_t> hW(base_off + rows * ld, 0);
    std::vector<float> hx(cols);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) {
            float v = 0.05f * rng.uni();
            if (fill == Fill::ZeroW) v = 0.0f;
            if (fill == Fill::Cancel) v = (c & 1) ? -0.125f : 0.125f;
            hW[base_off + r * ld + c] = from_float(v, t);
        }
    for (size_t c = 0; c < cols; ++c) {
        float v = rng.uni();
        if (fill == Fill::ZeroX) v = 0.0f;
        if (fill == Fill::Cancel) v = 1.0f;
        if (fill == Fill::WideRange) v *= std::ldexp(1.0f, static_cast<int>(c % 21) - 10);
        hx[c] = v;
    }

    std::vector<double> truth(rows), sum_abs(rows);
    std::vector<float> cpu(rows);
    gemv_cpu_f64(t, hW.data() + base_off, ld, hx.data(), truth.data(), sum_abs.data(), rows, cols);
    gemv_cpu_f32(t, hW.data() + base_off, ld, hx.data(), cpu.data(), rows, cols);

    constexpr size_t kGuard = 64;
    const float sentinel = 1234.5f;
    Dev<uint16_t> dW(hW.size());
    Dev<float> dx(cols), dy(rows + 2 * kGuard);
    CUDA(cudaMemcpy(dW.p, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));
    std::vector<float> init(rows + 2 * kGuard, sentinel);
    CUDA(cudaMemcpy(dy.p, init.data(), init.size() * 4, cudaMemcpyHostToDevice));

    int rc = gemv_cuda(t, dW.p + base_off, ld, dx.p, dy.p + kGuard, rows, cols);
    CUDA(cudaDeviceSynchronize());
    Result res;
    if (rc != 0) { std::printf("  gemv_cuda returned %d\n", rc); res.ok = false; return res; }

    std::vector<float> gpu(rows + 2 * kGuard);
    CUDA(cudaMemcpy(gpu.data(), dy.p, gpu.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kGuard; ++i)  // no out-of-bounds writes on either side of y
        if (gpu[i] != sentinel || gpu[kGuard + rows + i] != sentinel) { res.ok = false; break; }

    for (size_t r = 0; r < rows; ++r) {
        double g = gpu[kGuard + r];
        double bound_gpu = (static_cast<double>(cols) / 32.0 + 8.0) * kU * sum_abs[r];
        double bound_cpu = static_cast<double>(cols) * kU * sum_abs[r];
        double eg = std::fabs(g - truth[r]), ec = std::fabs(static_cast<double>(cpu[r]) - truth[r]);
        if (!std::isfinite(g)) { res.ok = false; continue; }
        if (bound_gpu == 0.0) { if (eg != 0.0) res.ok = false; }
        else res.max_gpu_ratio = std::max(res.max_gpu_ratio, eg / bound_gpu);
        if (bound_cpu != 0.0) res.max_cpu_ratio = std::max(res.max_cpu_ratio, ec / bound_cpu);
    }
    if (res.max_gpu_ratio > 1.0 || res.max_cpu_ratio > 1.0) res.ok = false;
    return res;
}

static const char* tname(WeightType t) { return t == WeightType::BF16 ? "bf16" : "f16"; }

static void test_conversions() {
    CHECK(bf16_to_float(float_to_bf16(1.0f)) == 1.0f && bf16_to_float(float_to_bf16(-0.15625f)) == -0.15625f);
    CHECK(half_to_float(float_to_half(1.0f)) == 1.0f && half_to_float(float_to_half(-0.15625f)) == -0.15625f);
    CHECK(half_to_float(float_to_half(65504.0f)) == 65504.0f);
    CHECK(std::isinf(half_to_float(float_to_half(1e6f))));
    CHECK(half_to_float(float_to_half(5.960464477539063e-8f)) == 5.960464477539063e-8f);  // smallest subnormal
    // Round-trip over a range: error within half an ulp.
    for (float f = -4.0f; f < 4.0f; f += 0.0137f) {
        CHECK(std::fabs(half_to_float(float_to_half(f)) - f) <= std::fabs(f) * 4.9e-4f + 6e-8f);
        CHECK(std::fabs(bf16_to_float(float_to_bf16(f)) - f) <= std::fabs(f) * 3.95e-3f);
    }
}

static void test_odd_shapes() {
    struct S { size_t rows, cols; };
    const S shapes[] = {{1, 1}, {1, 7}, {3, 5}, {8, 8}, {9, 16}, {17, 33}, {100, 1000}, {257, 4097}, {1031, 2563}};
    int n = 0;
    double worst = 0;
    for (auto t : {WeightType::BF16, WeightType::F16})
        for (auto s : shapes) {
            struct V { size_t ld, off; };
            const V variants[] = {{s.cols, 0}, {(s.cols + 7) / 8 * 8, 0}, {s.cols + 3, 0}, {s.cols, 1}, {(s.cols + 7) / 8 * 8 + 8, 1}};
            for (auto v : variants)
                for (auto f : {Fill::Random, Fill::ZeroX, Fill::ZeroW, Fill::Cancel, Fill::WideRange}) {
                    auto r = run_case(t, s.rows, s.cols, v.ld, v.off, f, 1000 + n);
                    ++n;
                    worst = std::max(worst, r.max_gpu_ratio);
                    if (!r.ok) {
                        std::printf("  FAIL %s %zux%zu ld=%zu off=%zu %s (gpu ratio %.3f, cpu ratio %.3f)\n", tname(t),
                                    s.rows, s.cols, v.ld, v.off, name(f), r.max_gpu_ratio, r.max_cpu_ratio);
                        ++failures;
                    }
                }
        }
    std::printf("  odd shapes: %d cases, worst GPU error / bound = %.4f\n", n, worst);
}

static void test_manifest_shapes(const std::string& manifest) {
    auto shapes = dara::core::load_matrix_shapes(manifest);
    CHECK(shapes.size() >= 10);
    double worst = 0;
    for (auto& m : shapes) {
        CHECK(m.dtype == "BF16");  // the checkpoint stores BF16; kernels must handle it natively
        bool big = m.rows * m.cols > 100'000'000ull;  // tied output head: BF16 only, one fill
        for (auto t : {WeightType::BF16, WeightType::F16}) {
            if (big && t == WeightType::F16) continue;
            auto r = run_case(t, m.rows, m.cols, m.cols, 0, Fill::Random, 7);
            worst = std::max(worst, r.max_gpu_ratio);
            if (!r.ok) {
                std::printf("  FAIL %s %s %llux%llu (gpu ratio %.3f, cpu ratio %.3f)\n", tname(t), m.name.c_str(),
                            (unsigned long long)m.rows, (unsigned long long)m.cols, r.max_gpu_ratio, r.max_cpu_ratio);
                ++failures;
            }
        }
    }
    std::printf("  %zu real Qwen3.5-4B shapes: worst GPU error / bound = %.4f\n", shapes.size(), worst);
}

// ---- benchmark ------------------------------------------------------------------------------

static void test_api_validation() {
    float* p = nullptr;
    CUDA(cudaMalloc(&p, 1024));
    CHECK(gemv_cuda(WeightType::BF16, p, 4, p, p, 2, 8) != 0);   // ld < cols
    CHECK(gemv_cuda(WeightType::BF16, nullptr, 8, p, p, 2, 8) != 0);
    CHECK(gemv_cuda(WeightType::BF16, p, 8, p, p, 0, 8) == 0);   // zero rows: nothing to do
    cudaFree(p);
}

static double percentile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * (v.size() - 1))];
}

__global__ void read_probe(const float4* p, size_t n4, float* out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 123456.0f) *out = acc;  // keep the loads alive without contending on a global atomic
}

static void benchmark(const std::string& manifest) {
    int dev = 0, clk_khz = 0, bus_bits = 0;
    CUDA(cudaGetDevice(&dev));
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
    double peak = 2.0 * clk_khz * 1e3 * (bus_bits / 8.0) / 1e9;  // GB/s, double data rate

    // Achievable read bandwidth: a plain streaming-read kernel (same access pattern as GEMV).
    const size_t cb = 256ull << 20;
    Dev<float4> a(cb / sizeof(float4));
    Dev<float> sink(1);
    CUDA(cudaMemset(a.p, 1, cb));  // touch the buffer so allocation cost is not timed
    CUDA(cudaMemset(sink.p, 0, 4));
    cudaEvent_t e0, e1;
    CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
    const size_t n4 = cb / sizeof(float4);
    for (int i = 0; i < 3; ++i) read_probe<<<1024, 256>>>(a.p, n4, sink.p);
    CUDA(cudaDeviceSynchronize());
    std::vector<float> probe_ms;
    for (int i = 0; i < 15; ++i) {
        CUDA(cudaEventRecord(e0));
        read_probe<<<1024, 256>>>(a.p, n4, sink.p);
        CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1));
        float t = 0; CUDA(cudaEventElapsedTime(&t, e0, e1));
        probe_ms.push_back(t);
    }
    double achievable = static_cast<double>(cb) / (percentile(probe_ms, 0.5) / 1e3) / 1e9;
    std::printf("  memory: spec peak %.0f GB/s (clock %.0f MHz, %d-bit bus), measured streaming read %.0f GB/s\n", peak,
                clk_khz / 1e3, bus_bits, achievable);
    std::printf("  %-42s %9s %9s %8s %8s\n", "shape (BF16 weights)", "median us", "p95 us", "GB/s", "% read");

    auto shapes = dara::core::load_matrix_shapes(manifest);
    for (auto& m : shapes) {
        if (m.rows < 1000) continue;  // tiny gate projections are launch-bound; not bandwidth shapes
        const size_t rows = m.rows, cols = m.cols;
        std::vector<uint16_t> hW(rows * cols);
        Rng rng(3);
        for (auto& w : hW) w = from_float(0.05f * rng.uni(), WeightType::BF16);
        std::vector<float> hx(cols);
        for (auto& v : hx) v = rng.uni();

        Dev<uint16_t> dW(hW.size());
        Dev<float> dx(cols), dy(rows);
        cudaEvent_t h0, h1;
        CUDA(cudaEventCreate(&h0)); CUDA(cudaEventCreate(&h1));
        CUDA(cudaEventRecord(h0));
        CUDA(cudaMemcpy(dW.p, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice));  // transfer: timed separately
        CUDA(cudaEventRecord(h1)); CUDA(cudaEventSynchronize(h1));
        float h2d_ms = 0; CUDA(cudaEventElapsedTime(&h2d_ms, h0, h1));
        CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));

        for (int i = 0; i < 20; ++i) gemv_cuda(WeightType::BF16, dW.p, cols, dx.p, dy.p, rows, cols);  // warm-up
        CUDA(cudaDeviceSynchronize());
        const int iters = m.rows * m.cols > 100'000'000ull ? 30 : 200;
        std::vector<cudaEvent_t> ev(2 * iters);
        for (auto& e : ev) CUDA(cudaEventCreate(&e));
        for (int i = 0; i < iters; ++i) {
            CUDA(cudaEventRecord(ev[2 * i]));
            gemv_cuda(WeightType::BF16, dW.p, cols, dx.p, dy.p, rows, cols);
            CUDA(cudaEventRecord(ev[2 * i + 1]));
        }
        CUDA(cudaDeviceSynchronize());
        std::vector<float> t_us(iters);
        for (int i = 0; i < iters; ++i) { float t = 0; CUDA(cudaEventElapsedTime(&t, ev[2 * i], ev[2 * i + 1])); t_us[i] = t * 1000.0f; }
        for (auto& e : ev) cudaEventDestroy(e);
        cudaEventDestroy(h0); cudaEventDestroy(h1);

        double med = percentile(t_us, 0.5), p95 = percentile(t_us, 0.95);
        double bytes = static_cast<double>(rows) * cols * 2 + cols * 4.0 + rows * 4.0;
        double gbs = bytes / (med * 1e-6) / 1e9;
        char label[96];
        std::snprintf(label, sizeof label, "%s %zux%zu", m.name.c_str(), rows, cols);
        std::printf("  %-42.42s %9.1f %9.1f %8.1f %7.0f%%   (H2D of W: %.0f ms)\n", label, med, p95, gbs,
                    100.0 * gbs / achievable, h2d_ms);

        if (m.name == "attention.mlp.gate_proj.weight") {  // one CPU data point for scale
            std::vector<float> y(rows);
            auto c0 = std::chrono::steady_clock::now();
            gemv_cpu_f32(WeightType::BF16, hW.data(), cols, hx.data(), y.data(), rows, cols);
            double cpu_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - c0).count();
            std::printf("      CPU reference, same shape, 1 thread: %.0f us (GPU %.0fx faster)\n", cpu_us, cpu_us / med);
        }
    }
}

int main(int argc, char** argv) {
    std::string manifest = argc > 1 ? argv[1] : "manifests/qwen3.5-4b.json";
    bool bench = !(argc > 2 && std::string(argv[2]) == "--no-bench");
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (n == 0) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp p{};
    CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    test_conversions();
    test_api_validation();
    test_odd_shapes();
    test_manifest_shapes(manifest);
    if (bench && !failures) benchmark(manifest);
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
