// Q6_K fused dequantize + GEMV: layout fidelity and correctness (benchmarks live in workload_bench.cu).
//
//  1. CPU decode must equal an independent implementation (gguf-py) bit for bit, on real bytes from the
//     pinned Q4_K_M GGUF (output head and an ffn_down tensor) and on random-but-valid blocks.
//  2. GPU decode must equal the same values: x = e_c makes the GEMV return column c exactly.
//  3. GEMV vs a double-precision truth over adversarial shapes (including odd block counts, which change
//     row alignment because a block is 210 bytes), padded strides and extreme values.
//     Bound: |err| <= (cols/32 + 9) * 2^-24 * sum|w*x|.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/half.h"
#include "kernels/q6k.h"

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

struct Fixture {
    std::vector<BlockQ6K> blocks;
    std::vector<float> expected;
};

static bool load_fixture(const std::string& path, Fixture& f) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), 4);
    f.blocks.resize(n);
    f.expected.resize(static_cast<size_t>(n) * 256);
    in.read(reinterpret_cast<char*>(f.blocks.data()), static_cast<std::streamsize>(n) * sizeof(BlockQ6K));
    in.read(reinterpret_cast<char*>(f.expected.data()), static_cast<std::streamsize>(f.expected.size() * 4));
    return static_cast<bool>(in);
}

static void test_fixture(const std::string& dir, const char* file, size_t rows, size_t cols) {
    Fixture f;
    if (!load_fixture(dir + "/" + file, f)) { std::printf("FAIL cannot load fixture %s\n", file); ++failures; return; }
    CHECK(f.blocks.size() == rows * cols / 256);

    std::vector<float> dec(f.expected.size());
    q6k_dequant(f.blocks.data(), f.blocks.size(), dec.data());
    size_t cpu_bad = 0;
    for (size_t i = 0; i < dec.size(); ++i) cpu_bad += dec[i] != f.expected[i];
    CHECK(cpu_bad == 0);

    const size_t bytes = f.blocks.size() * sizeof(BlockQ6K);
    Dev<uint8_t> dW(bytes);
    Dev<float> dx(cols), dy(rows);
    CUDA(cudaMemcpy(dW.p, f.blocks.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(dx.p, 0, cols * 4));
    std::vector<float> y(rows);
    const float one = 1.0f, zero = 0.0f;
    size_t gpu_bad = 0;
    for (size_t c = 0; c < cols; ++c) {
        CUDA(cudaMemcpy(dx.p + c, &one, 4, cudaMemcpyHostToDevice));
        CHECK(gemv_q6k_cuda(dW.p, cols / 256, dx.p, dy.p, rows, cols) == 0);
        CUDA(cudaMemcpy(y.data(), dy.p, rows * 4, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < rows; ++r) gpu_bad += y[r] != f.expected[r * cols + c];
        CUDA(cudaMemcpy(dx.p + c, &zero, 4, cudaMemcpyHostToDevice));
    }
    CHECK(gpu_bad == 0);

    // Same data at a base address that is 2 mod 4 (all weights shifted by 2 bytes): the kernel must not
    // depend on 4-byte alignment. The first 128 columns are enough to cover every lane and block parity.
    Dev<uint8_t> dW2(bytes + 8);
    CUDA(cudaMemcpy(dW2.p + 2, f.blocks.data(), bytes, cudaMemcpyHostToDevice));
    size_t shifted_bad = 0;
    for (size_t c = 0; c < std::min<size_t>(cols, 128); ++c) {
        CUDA(cudaMemcpy(dx.p + c, &one, 4, cudaMemcpyHostToDevice));
        CHECK(gemv_q6k_cuda(dW2.p + 2, cols / 256, dx.p, dy.p, rows, cols) == 0);
        CUDA(cudaMemcpy(y.data(), dy.p, rows * 4, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < rows; ++r) shifted_bad += y[r] != f.expected[r * cols + c];
        CUDA(cudaMemcpy(dx.p + c, &zero, 4, cudaMemcpyHostToDevice));
    }
    CHECK(shifted_bad == 0);
    std::printf("  fixture %-14s %zu rows x %zu cols: CPU mismatches %zu, GPU mismatches %zu of %zu weights\n", file,
                rows, cols, cpu_bad, gpu_bad, f.expected.size());
}

enum Mode { Quantized, RawSane, RawExtreme, ZeroX, ZeroW };
static const char* name(Mode m) {
    static const char* n[] = {"quantized", "raw-sane", "raw-extreme", "zero-x", "zero-w"};
    return n[m];
}

static void fill_blocks(std::vector<BlockQ6K>& w, size_t rows, size_t cols, size_t ld, Mode mode, Rng& rng) {
    std::memset(w.data(), 0xFF, w.size() * sizeof(BlockQ6K));  // padding decodes to NaN (d = 0xFFFF) if ever read
    const size_t nb = cols / 256;
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        BlockQ6K* b = w.data() + r * ld;
        if (mode == Quantized) {
            for (auto& v : row) v = 0.03f * (rng.uni() + rng.uni() + rng.uni());
            q6k_quantize(row.data(), cols, b);
            continue;
        }
        for (size_t i = 0; i < nb; ++i) {
            uint8_t raw[sizeof(BlockQ6K)];
            for (auto& v : raw) v = static_cast<uint8_t>(rng.next() >> 33);
            std::memcpy(&b[i], raw, sizeof raw);
            if (mode == ZeroW) { std::memset(&b[i], 0, sizeof(BlockQ6K)); continue; }
            if (mode == RawSane) b[i].d = float_to_half(0.0002f + 0.004f * std::fabs(rng.uni()));
            else b[i].d = float_to_half(std::ldexp(0.5f + 0.5f * std::fabs(rng.uni()), static_cast<int>(rng.next() % 30) - 14));
        }
    }
}

struct Res { double gpu = 0, cpu = 0; bool ok = true; };

static Res run_case(size_t rows, size_t cols, size_t ld, Mode mode, uint64_t seed) {
    Rng rng(seed);
    std::vector<BlockQ6K> hW(rows * ld);
    fill_blocks(hW, rows, cols, ld, mode, rng);
    std::vector<float> hx(cols);
    for (size_t c = 0; c < cols; ++c) hx[c] = mode == ZeroX ? 0.0f : rng.uni();

    std::vector<double> truth(rows), sum_abs(rows);
    std::vector<float> cpu(rows);
    gemv_q6k_cpu_f64(hW.data(), ld, hx.data(), truth.data(), sum_abs.data(), rows, cols);
    gemv_q6k_cpu_f32(hW.data(), ld, hx.data(), cpu.data(), rows, cols);

    constexpr size_t kGuard = 64;
    const float sentinel = 1234.5f;
    Dev<uint8_t> dW(hW.size() * sizeof(BlockQ6K));
    Dev<float> dx(cols), dy(rows + 2 * kGuard);
    CUDA(cudaMemcpy(dW.p, hW.data(), hW.size() * sizeof(BlockQ6K), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));
    std::vector<float> init(rows + 2 * kGuard, sentinel);
    CUDA(cudaMemcpy(dy.p, init.data(), init.size() * 4, cudaMemcpyHostToDevice));

    Res res;
    if (gemv_q6k_cuda(dW.p, ld, dx.p, dy.p + kGuard, rows, cols) != 0) { res.ok = false; return res; }
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
    const size_t cols_list[] = {256, 512, 768, 1280, 2560, 4096};  // 1, 2, 3, 5, 10, 16 blocks: odd and even
    int n = 0;
    double worst = 0;
    for (size_t rows : rows_list)
        for (size_t cols : cols_list)
            for (size_t pad : {size_t(0), size_t(1), size_t(3)})
                for (int m = Quantized; m <= ZeroW; ++m) {
                    if (rows * cols > 3'000'000 && m != Quantized) continue;
                    auto r = run_case(rows, cols, cols / 256 + pad, static_cast<Mode>(m), 900 + n++);
                    worst = std::max(worst, r.gpu);
                    if (!r.ok) {
                        std::printf("  FAIL %zux%zu pad=%zu %s (gpu ratio %.3f, cpu ratio %.3f)\n", rows, cols, pad,
                                    name(static_cast<Mode>(m)), r.gpu, r.cpu);
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
    std::vector<BlockQ6K> q(rows * cols / 256);
    std::vector<float> dq(w.size());
    for (size_t r = 0; r < rows; ++r) q6k_quantize(w.data() + r * cols, cols, q.data() + r * cols / 256);
    q6k_dequant(q.data(), q.size(), dq.data());
    double num = 0, den = 0;
    for (size_t i = 0; i < w.size(); ++i) { num += (double(dq[i]) - w[i]) * (double(dq[i]) - w[i]); den += double(w[i]) * w[i]; }
    const double rel = std::sqrt(num / den);
    std::printf("  test quantizer on Gaussian-like weights: relative RMSE %.4f\n", rel);
    CHECK(rel > 0.0 && rel < 0.04);  // 6 bits: roughly 4x finer than 4-bit
}

static void test_api_validation() {
    Dev<uint8_t> w(8192);
    Dev<float> x(1024), y(16);
    CHECK(gemv_q6k_cuda(w.p, 1, x.p, y.p, 1, 255) != 0);       // cols not a whole number of blocks
    CHECK(gemv_q6k_cuda(w.p, 1, x.p, y.p, 1, 0) != 0);
    CHECK(gemv_q6k_cuda(w.p, 1, x.p, y.p, 1, 512) != 0);       // stride smaller than a row
    CHECK(gemv_q6k_cuda(w.p + 1, 2, x.p, y.p, 1, 512) != 0);   // odd weight address
    CHECK(gemv_q6k_cuda(w.p + 2, 2, x.p, y.p, 1, 512) == 0);   // 2-byte aligned weights are fine
    CHECK(gemv_q6k_cuda(w.p, 2, x.p + 1, y.p, 1, 512) != 0);   // misaligned x
    CHECK(gemv_q6k_cuda(nullptr, 2, x.p, y.p, 1, 512) != 0);
    CHECK(gemv_q6k_cuda(w.p, 2, x.p, y.p, 0, 512) == 0);       // zero rows
    CUDA(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    std::string root = argc > 1 ? argv[1] : ".";
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (n == 0) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp p{};
    CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    test_api_validation();
    test_fixture(root + "/tests/fixtures", "q6k_embd.bin", 4, 2560);
    test_fixture(root + "/tests/fixtures", "q6k_down.bin", 2, 9216);
    test_fixture(root + "/tests/fixtures", "q6k_random.bin", 16, 1024);
    test_quantizer_quality();
    test_cases();
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
