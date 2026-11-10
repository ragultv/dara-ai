// Q4_K fused dequantize + GEMV: layout fidelity, correctness, then a benchmark on real tensor shapes.
//
//  1. CPU decode must equal an independent implementation (gguf-py) bit for bit, on real bytes from
//     the pinned Q4_K_M GGUF and on random-but-valid blocks (tests/fixtures/*.bin).
//  2. GPU decode must equal the same expected values: x = e_c makes the GEMV return column c exactly.
//  3. GEMV vs a double-precision truth over adversarial shapes, strides and value ranges.
//     Bound: |err| <= (cols/32 + 9) * 2^-24 * sum|w*x| (32 strided lane sums + 5-level tree + weight rounding).
//  4. Benchmark on the distinct Q4_K shapes found in the real GGUF header.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "kernels/gemv.h"
#include "kernels/q4k.h"

using namespace dara::kernels;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

static constexpr double kU = 5.960464477539063e-8;

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return static_cast<float>(static_cast<double>(next() >> 11) / 4503599627370496.0 - 1.0); }
};

template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

static double percentile(std::vector<float> v, double q) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(q * (v.size() - 1))];
}

// ---- fixtures -------------------------------------------------------------------------------

struct Fixture {
    std::vector<BlockQ4K> blocks;
    std::vector<float> expected;
};

static bool load_fixture(const std::string& path, Fixture& f) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), 4);
    f.blocks.resize(n);
    f.expected.resize(static_cast<size_t>(n) * 256);
    in.read(reinterpret_cast<char*>(f.blocks.data()), static_cast<std::streamsize>(n) * sizeof(BlockQ4K));
    in.read(reinterpret_cast<char*>(f.expected.data()), static_cast<std::streamsize>(f.expected.size() * 4));
    return static_cast<bool>(in);
}

static void test_fixture(const std::string& dir, const char* file, size_t rows, size_t cols) {
    Fixture f;
    if (!load_fixture(dir + "/" + file, f)) { std::printf("FAIL cannot load fixture %s\n", file); ++failures; return; }
    CHECK(f.blocks.size() == rows * cols / 256);

    // 1. CPU decode == independent reference, bit for bit.
    std::vector<float> dec(f.expected.size());
    q4k_dequant(f.blocks.data(), f.blocks.size(), dec.data());
    size_t cpu_bad = 0;
    for (size_t i = 0; i < dec.size(); ++i) cpu_bad += dec[i] != f.expected[i];
    CHECK(cpu_bad == 0);

    // 2. GPU decode: GEMV with x = e_c returns column c of W exactly.
    Dev<uint8_t> dW(f.blocks.size() * sizeof(BlockQ4K));
    Dev<float> dx(cols), dy(rows);
    CUDA(cudaMemcpy(dW.p, f.blocks.data(), f.blocks.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice));
    CUDA(cudaMemset(dx.p, 0, cols * 4));
    std::vector<float> y(rows);
    const float one = 1.0f, zero = 0.0f;
    size_t gpu_bad = 0;
    for (size_t c = 0; c < cols; ++c) {
        CUDA(cudaMemcpy(dx.p + c, &one, 4, cudaMemcpyHostToDevice));
        CHECK(gemv_q4k_cuda(dW.p, cols / 256, dx.p, dy.p, rows, cols) == 0);
        CUDA(cudaMemcpy(y.data(), dy.p, rows * 4, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < rows; ++r) gpu_bad += y[r] != f.expected[r * cols + c];
        CUDA(cudaMemcpy(dx.p + c, &zero, 4, cudaMemcpyHostToDevice));
    }
    CHECK(gpu_bad == 0);
    std::printf("  fixture %-15s %zu rows x %zu cols: CPU mismatches %zu, GPU mismatches %zu of %zu weights\n", file,
                rows, cols, cpu_bad, gpu_bad, f.expected.size());
}

// ---- GEMV cases -----------------------------------------------------------------------------

enum class Mode { Quantized, RawSane, RawExtreme, ZeroX, ZeroW };
static const char* name(Mode m) {
    switch (m) {
        case Mode::Quantized: return "quantized"; case Mode::RawSane: return "raw-sane";
        case Mode::RawExtreme: return "raw-extreme"; case Mode::ZeroX: return "zero-x"; case Mode::ZeroW: return "zero-w";
    }
    return "?";
}

static void fill_blocks(std::vector<BlockQ4K>& w, size_t rows, size_t cols, size_t ld, Mode mode, Rng& rng) {
    std::memset(w.data(), 0xFF, w.size() * sizeof(BlockQ4K));  // padding blocks decode to NaN if ever read
    const size_t nb = cols / 256;
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        BlockQ4K* b = w.data() + r * ld;
        if (mode == Mode::Quantized) {
            for (auto& v : row) v = 0.03f * (rng.uni() + rng.uni() + rng.uni());
            q4k_quantize(row.data(), cols, b);
            continue;
        }
        for (size_t i = 0; i < nb; ++i) {
            uint8_t raw[sizeof(BlockQ4K)];
            for (auto& v : raw) v = static_cast<uint8_t>(rng.next() >> 33);
            std::memcpy(&b[i], raw, sizeof raw);
            if (mode == Mode::ZeroW) { std::memset(&b[i], 0, sizeof(BlockQ4K)); continue; }
            if (mode == Mode::RawSane) {
                b[i].d = float_to_half(0.0005f + 0.01f * std::fabs(rng.uni()));
                b[i].dmin = float_to_half(0.0005f + 0.01f * std::fabs(rng.uni()));
            } else {  // finite fp16 across ~30 binades, max just under 2^15 (fp16 overflows at 65504)
                b[i].d = float_to_half(std::ldexp(0.5f + 0.5f * std::fabs(rng.uni()), static_cast<int>(rng.next() % 30) - 14));
                b[i].dmin = float_to_half(std::ldexp(0.5f + 0.5f * std::fabs(rng.uni()), static_cast<int>(rng.next() % 30) - 14));
            }
        }
    }
}

struct Res { double gpu = 0, cpu = 0, quant_rel_rmse = -1; bool ok = true; };

static Res run_case(size_t rows, size_t cols, size_t ld, Mode mode, uint64_t seed) {
    Rng rng(seed);
    std::vector<BlockQ4K> hW(rows * ld);
    fill_blocks(hW, rows, cols, ld, mode, rng);
    std::vector<float> hx(cols);
    for (size_t c = 0; c < cols; ++c) hx[c] = mode == Mode::ZeroX ? 0.0f : rng.uni();

    std::vector<double> truth(rows), sum_abs(rows);
    std::vector<float> cpu(rows);
    gemv_q4k_cpu_f64(hW.data(), ld, hx.data(), truth.data(), sum_abs.data(), rows, cols);
    gemv_q4k_cpu_f32(hW.data(), ld, hx.data(), cpu.data(), rows, cols);

    constexpr size_t kGuard = 64;
    const float sentinel = 1234.5f;
    Dev<uint8_t> dW(hW.size() * sizeof(BlockQ4K));
    Dev<float> dx(cols), dy(rows + 2 * kGuard);
    CUDA(cudaMemcpy(dW.p, hW.data(), hW.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));
    std::vector<float> init(rows + 2 * kGuard, sentinel);
    CUDA(cudaMemcpy(dy.p, init.data(), init.size() * 4, cudaMemcpyHostToDevice));

    Res res;
    if (gemv_q4k_cuda(dW.p, ld, dx.p, dy.p + kGuard, rows, cols) != 0) { res.ok = false; return res; }
    CUDA(cudaDeviceSynchronize());
    std::vector<float> gpu(rows + 2 * kGuard);
    CUDA(cudaMemcpy(gpu.data(), dy.p, gpu.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kGuard; ++i)
        if (gpu[i] != sentinel || gpu[kGuard + rows + i] != sentinel) { res.ok = false; break; }

    for (size_t r = 0; r < rows; ++r) {
        const double g = gpu[kGuard + r];
        if (!std::isfinite(g)) { res.ok = false; continue; }
        const double bg = (cols / 32.0 + 9.0) * kU * sum_abs[r], bc = cols * kU * sum_abs[r];
        const double eg = std::fabs(g - truth[r]), ec = std::fabs(static_cast<double>(cpu[r]) - truth[r]);
        if (bg == 0.0) { if (eg != 0.0) res.ok = false; }
        else res.gpu = std::max(res.gpu, eg / bg);
        if (bc != 0.0) res.cpu = std::max(res.cpu, ec / bc);
    }
    if (res.gpu > 1.0 || res.cpu > 1.0) res.ok = false;
    return res;
}

static void test_cases() {
    const size_t rows_list[] = {1, 2, 3, 7, 9, 100, 257, 1031};
    const size_t cols_list[] = {256, 512, 768, 2560, 4096};
    int n = 0;
    double worst = 0;
    for (size_t rows : rows_list)
        for (size_t cols : cols_list)
            for (size_t pad : {size_t(0), size_t(1), size_t(3)})  // padded row stride in blocks
                for (Mode m : {Mode::Quantized, Mode::RawSane, Mode::RawExtreme, Mode::ZeroX, Mode::ZeroW}) {
                    if (rows * cols > 3'000'000 && m != Mode::Quantized) continue;  // keep the run short
                    auto r = run_case(rows, cols, cols / 256 + pad, m, 500 + n++);
                    worst = std::max(worst, r.gpu);
                    if (!r.ok) {
                        std::printf("  FAIL %zux%zu pad=%zu %s (gpu ratio %.3f, cpu ratio %.3f)\n", rows, cols, pad,
                                    name(m), r.gpu, r.cpu);
                        ++failures;
                    }
                }
    std::printf("  adversarial cases: %d, worst GPU error / bound = %.4f\n", n, worst);
}

static void test_quantizer_quality() {
    Rng rng(99);
    const size_t cols = 4096, rows = 64;
    std::vector<float> w(rows * cols);
    for (auto& v : w) v = 0.03f * (rng.uni() + rng.uni() + rng.uni());
    std::vector<BlockQ4K> q(rows * cols / 256);
    std::vector<float> dq(w.size());
    for (size_t r = 0; r < rows; ++r) q4k_quantize(w.data() + r * cols, cols, q.data() + r * cols / 256);
    q4k_dequant(q.data(), q.size(), dq.data());
    double num = 0, den = 0;
    for (size_t i = 0; i < w.size(); ++i) { num += (double(dq[i]) - w[i]) * (double(dq[i]) - w[i]); den += double(w[i]) * w[i]; }
    const double rel = std::sqrt(num / den);
    std::printf("  test quantizer on Gaussian-like weights: relative RMSE %.3f\n", rel);
    CHECK(rel > 0.0 && rel < 0.15);  // 4-bit with 8 sub-blocks: ~5-8% is expected
}

static void test_api_validation() {
    Dev<uint8_t> w(4096);
    Dev<float> x(1024), y(16);
    CHECK(gemv_q4k_cuda(w.p, 1, x.p, y.p, 1, 255) != 0);      // cols not a whole number of blocks
    CHECK(gemv_q4k_cuda(w.p, 1, x.p, y.p, 1, 0) != 0);
    CHECK(gemv_q4k_cuda(w.p, 1, x.p, y.p, 1, 512) != 0);      // stride smaller than a row
    CHECK(gemv_q4k_cuda(w.p + 4, 2, x.p, y.p, 1, 512) != 0);  // misaligned weights
    CHECK(gemv_q4k_cuda(w.p, 2, x.p + 1, y.p, 1, 512) != 0);  // misaligned x
    CHECK(gemv_q4k_cuda(nullptr, 2, x.p, y.p, 1, 512) != 0);
    CHECK(gemv_q4k_cuda(w.p, 2, x.p, y.p, 0, 512) == 0);      // zero rows
}

// ---- benchmark ------------------------------------------------------------------------------

__global__ void read_probe(const float4* p, size_t n4, float* out) {
    float acc = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        float4 v = __ldg(p + i);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 123456.0f) *out = acc;
}

static double read_bandwidth(Dev<float4>& a, Dev<float>& sink, size_t bytes) {
    cudaEvent_t e0, e1;
    CUDA(cudaEventCreate(&e0)); CUDA(cudaEventCreate(&e1));
    const size_t n4 = bytes / sizeof(float4);
    for (int i = 0; i < 2; ++i) read_probe<<<1024, 256>>>(a.p, n4, sink.p);
    std::vector<float> ms;
    for (int i = 0; i < 10; ++i) {
        CUDA(cudaEventRecord(e0));
        read_probe<<<1024, 256>>>(a.p, n4, sink.p);
        CUDA(cudaEventRecord(e1)); CUDA(cudaEventSynchronize(e1));
        float t = 0; CUDA(cudaEventElapsedTime(&t, e0, e1)); ms.push_back(t);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return static_cast<double>(bytes) / (percentile(ms, 0.5) / 1e3) / 1e9;
}

template <class F> static std::vector<float> time_us(F&& launch, int iters) {
    for (int i = 0; i < 20; ++i) launch();
    CUDA(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> ev(2 * iters);
    for (auto& e : ev) CUDA(cudaEventCreate(&e));
    for (int i = 0; i < iters; ++i) { CUDA(cudaEventRecord(ev[2 * i])); launch(); CUDA(cudaEventRecord(ev[2 * i + 1])); }
    CUDA(cudaDeviceSynchronize());
    std::vector<float> t(iters);
    for (int i = 0; i < iters; ++i) { float ms = 0; CUDA(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1])); t[i] = ms * 1000.0f; }
    for (auto& e : ev) cudaEventDestroy(e);
    return t;
}

static void benchmark(const std::string& tensors_json) {
    std::ifstream in(tensors_json);
    nlohmann::json j;
    in >> j;
    std::set<std::pair<uint64_t, uint64_t>> shapes;  // (rows, cols)
    for (auto& t : j.at("tensors"))
        if (t.at("type") == "Q4_K" && t.at("dims").size() == 2)
            shapes.insert({t["dims"][1].get<uint64_t>(), t["dims"][0].get<uint64_t>()});
    CHECK(!shapes.empty());

    Dev<float4> probe_buf((128ull << 20) / sizeof(float4));
    Dev<float> sink(1);
    CUDA(cudaMemset(probe_buf.p, 1, 128ull << 20));
    CUDA(cudaMemset(sink.p, 0, 4));

    for (int i = 0; i < 40; ++i) read_bandwidth(probe_buf, sink, 128ull << 20);  // let the GPU reach steady clocks

    std::printf("  %-14s %9s %9s %8s %9s | %10s %8s\n", "shape (Q4_K)", "median us", "p95 us", "GB/s", "% read", "BF16 us", "speedup");
    for (auto [rows, cols] : shapes) {
        const size_t nb = cols / 256;
        std::vector<BlockQ4K> hW(rows * nb);
        Rng rng(5);
        for (auto& b : hW) {
            for (auto& v : reinterpret_cast<uint8_t(&)[sizeof(BlockQ4K)]>(b)) v = static_cast<uint8_t>(rng.next() >> 33);
            b.d = float_to_half(0.002f); b.dmin = float_to_half(0.002f);
        }
        std::vector<float> hx(cols);
        for (auto& v : hx) v = rng.uni();
        Dev<uint8_t> dW(hW.size() * sizeof(BlockQ4K));
        Dev<float> dx(cols), dy(rows);
        CUDA(cudaMemcpy(dW.p, hW.data(), hW.size() * sizeof(BlockQ4K), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));

        const double ceiling = read_bandwidth(probe_buf, sink, 128ull << 20);  // re-measured per shape
        auto tq = time_us([&] { gemv_q4k_cuda(dW.p, nb, dx.p, dy.p, rows, cols); }, 200);
        const double med = percentile(tq, 0.5), p95 = percentile(tq, 0.95);
        const double bytes = double(rows) * nb * 144 + cols * 4.0 + rows * 4.0;
        const double gbs = bytes / (med * 1e-6) / 1e9;

        // Same shape, BF16 weights, for the format-to-format ratio.
        std::vector<uint16_t> hb(rows * cols, 0x3c00);
        Dev<uint16_t> dB(hb.size());
        CUDA(cudaMemcpy(dB.p, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
        auto tb = time_us([&] { gemv_cuda(WeightType::BF16, dB.p, cols, dx.p, dy.p, rows, cols); }, 100);
        const double bmed = percentile(tb, 0.5);
        std::printf("  %5zux%-8zu %9.1f %9.1f %8.1f %8.0f%% | %10.1f %7.2fx\n", rows, cols, med, p95, gbs,
                    100.0 * gbs / ceiling, bmed, bmed / med);
    }
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    bool bench = !(argc > 2 && std::string(argv[2]) == "--no-bench");
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (n == 0) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp p{};
    CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    test_api_validation();
    test_fixture(root + "/tests/fixtures", "q4k_real.bin", 4, 2560);
    test_fixture(root + "/tests/fixtures", "q4k_random.bin", 8, 2048);
    test_quantizer_quality();
    test_cases();
    if (bench && !failures) benchmark(root + "/manifests/qwen3.5-4b-q4_k_m.tensors.json");
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
