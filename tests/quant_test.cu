// Format-generic GEMV tests, run through the registry and dispatcher for Q4_K, Q5_K, Q6_K and Q8_0.
//
//  1. CPU decode == independent implementation (gguf-py), bit for bit, on real GGUF bytes and random blocks.
//  2. GPU decode == the same values (GEMV with x = e_c returns column c exactly), including at a shifted base.
//  3. GEMV vs double-precision truth over adversarial shapes, strides and value ranges, with guard memory.
//     Bound: |err| <= (cols/32 + 9) * 2^-24 * sum|w*x|.
//  4. Dispatcher/registry behavior and input validation.
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
#include "kernels/quant.h"

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

static const QuantType kAll[] = {QuantType::Q4_K, QuantType::Q5_K, QuantType::Q6_K, QuantType::Q8_0};

// ---- fixtures -------------------------------------------------------------------------------

struct FixtureSpec { QuantType t; const char* file; size_t rows, cols; };
static const FixtureSpec kFixtures[] = {
    {QuantType::Q4_K, "q4k_real.bin", 4, 2560},     {QuantType::Q4_K, "q4k_random.bin", 8, 2048},
    {QuantType::Q5_K, "q5k_qkv.bin", 4, 2560},      {QuantType::Q5_K, "q5k_ssmout.bin", 2, 4096},
    {QuantType::Q5_K, "q5k_random.bin", 8, 2048},   {QuantType::Q6_K, "q6k_embd.bin", 4, 2560},
    {QuantType::Q6_K, "q6k_down.bin", 2, 9216},     {QuantType::Q6_K, "q6k_random.bin", 16, 1024},
    {QuantType::Q8_0, "q80_alpha.bin", 8, 2560},    {QuantType::Q8_0, "q80_random.bin", 8, 1024},
};

static void test_fixture(const std::string& dir, const FixtureSpec& fs) {
    const QuantFormat& f = quant_format(fs.t);
    std::ifstream in(dir + "/" + fs.file, std::ios::binary);
    uint32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), 4);
    std::vector<uint8_t> blocks(static_cast<size_t>(n) * f.block_bytes);
    std::vector<float> expected(static_cast<size_t>(n) * f.block_weights);
    in.read(reinterpret_cast<char*>(blocks.data()), static_cast<std::streamsize>(blocks.size()));
    in.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(expected.size() * 4));
    if (!in || n != fs.rows * fs.cols / f.block_weights) { std::printf("FAIL fixture %s unreadable/mismatched\n", fs.file); ++failures; return; }

    std::vector<float> dec(expected.size());
    f.dequant_f32(blocks.data(), n, dec.data());
    size_t cpu_bad = 0;
    for (size_t i = 0; i < dec.size(); ++i) cpu_bad += dec[i] != expected[i];
    CHECK(cpu_bad == 0);

    // GPU: column extraction at the natural base and at a base shifted by the format's alignment
    // (2 mod 4 for the 2-byte-aligned formats), so nothing relies on a 256-byte-aligned allocation.
    const size_t shift = f.weight_align;
    Dev<uint8_t> dW(blocks.size() + 64);
    Dev<float> dx(fs.cols), dy(fs.rows);
    CUDA(cudaMemcpy(dW.p, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
    Dev<uint8_t> dW2(blocks.size() + 2 * shift);
    CUDA(cudaMemcpy(dW2.p + shift, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemset(dx.p, 0, fs.cols * 4));
    std::vector<float> y(fs.rows);
    const float one = 1.0f, zero = 0.0f;
    size_t bad = 0, bad_shift = 0;
    const size_t ld = fs.cols / f.block_weights;
    for (size_t c = 0; c < fs.cols; ++c) {
        CUDA(cudaMemcpy(dx.p + c, &one, 4, cudaMemcpyHostToDevice));
        CHECK(gemv_quant_cuda(fs.t, dW.p, ld, dx.p, dy.p, fs.rows, fs.cols) == 0);
        CUDA(cudaMemcpy(y.data(), dy.p, fs.rows * 4, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < fs.rows; ++r) bad += y[r] != expected[r * fs.cols + c];
        if (c < 128) {
            CHECK(gemv_quant_cuda(fs.t, dW2.p + shift, ld, dx.p, dy.p, fs.rows, fs.cols) == 0);
            CUDA(cudaMemcpy(y.data(), dy.p, fs.rows * 4, cudaMemcpyDeviceToHost));
            for (size_t r = 0; r < fs.rows; ++r) bad_shift += y[r] != expected[r * fs.cols + c];
        }
        CUDA(cudaMemcpy(dx.p + c, &zero, 4, cudaMemcpyHostToDevice));
    }
    CHECK(bad == 0 && bad_shift == 0);
    std::printf("  %-5s %-16s %2zu x %-5zu CPU mismatches %zu, GPU %zu (shifted base %zu) of %zu weights\n", f.name,
                fs.file, fs.rows, fs.cols, cpu_bad, bad, bad_shift, expected.size());
}

// ---- adversarial GEMV -----------------------------------------------------------------------

enum Mode { Quantized, RawSane, RawExtreme, ZeroX, ZeroW };
static const char* mode_name(int m) {
    static const char* n[] = {"quantized", "raw-sane", "raw-extreme", "zero-x", "zero-w"};
    return n[m];
}

static void fill(const QuantFormat& f, std::vector<uint8_t>& w, size_t rows, size_t cols, size_t ld, Mode mode, Rng& rng) {
    std::memset(w.data(), 0xFF, w.size());  // padding decodes to NaN if the kernel ever reads it
    const size_t nb = cols / f.block_weights;
    std::vector<float> row(cols);
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* b = w.data() + r * ld * f.block_bytes;
        if (mode == Quantized) {
            for (auto& v : row) v = 0.03f * (rng.uni() + rng.uni() + rng.uni());
            f.quantize(row.data(), cols, b);
            continue;
        }
        for (size_t i = 0; i < nb; ++i) {
            uint8_t* blk = b + i * f.block_bytes;
            for (int k = 0; k < f.block_bytes; ++k) blk[k] = static_cast<uint8_t>(rng.next() >> 33);
            if (mode == ZeroW) { std::memset(blk, 0, f.block_bytes); continue; }
            auto scale = [&] {
                return mode == RawSane ? 0.0005f + 0.01f * std::fabs(rng.uni())
                                       : std::ldexp(0.5f + 0.5f * std::fabs(rng.uni()), static_cast<int>(rng.next() % 30) - 14);
            };
            f.set_scales(blk, scale(), scale());  // finite fp16 across ~30 binades in the extreme mode
        }
    }
}

struct Res { double gpu = 0, cpu = 0; bool ok = true; };

static Res run_case(QuantType t, size_t rows, size_t cols, size_t ld, Mode mode, uint64_t seed) {
    const QuantFormat& f = quant_format(t);
    Rng rng(seed);
    std::vector<uint8_t> hW(rows * ld * f.block_bytes);
    fill(f, hW, rows, cols, ld, mode, rng);
    std::vector<float> hx(cols);
    for (size_t c = 0; c < cols; ++c) hx[c] = mode == ZeroX ? 0.0f : rng.uni();

    std::vector<double> truth(rows), sum_abs(rows);
    std::vector<float> cpu(rows);
    gemv_quant_cpu_f64(t, hW.data(), ld, hx.data(), truth.data(), sum_abs.data(), rows, cols);
    gemv_quant_cpu_f32(t, hW.data(), ld, hx.data(), cpu.data(), rows, cols);

    constexpr size_t kGuard = 64;
    const float sentinel = 1234.5f;
    Dev<uint8_t> dW(hW.size());
    Dev<float> dx(cols), dy(rows + 2 * kGuard);
    CUDA(cudaMemcpy(dW.p, hW.data(), hW.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dx.p, hx.data(), cols * 4, cudaMemcpyHostToDevice));
    std::vector<float> init(rows + 2 * kGuard, sentinel);
    CUDA(cudaMemcpy(dy.p, init.data(), init.size() * 4, cudaMemcpyHostToDevice));

    Res res;
    if (gemv_quant_cuda(t, dW.p, ld, dx.p, dy.p + kGuard, rows, cols) != 0) { res.ok = false; return res; }
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

static void test_cases(QuantType t) {
    const QuantFormat& f = quant_format(t);
    const size_t rows_list[] = {1, 2, 3, 7, 9, 100, 257, 1031};
    const size_t nb_list[] = {1, 2, 3, 5, f.block_weights == 256 ? 10u : 80u, f.block_weights == 256 ? 16u : 129u};
    int n = 0;
    double worst = 0;
    for (size_t rows : rows_list)
        for (size_t nb : nb_list) {
            const size_t cols = nb * f.block_weights;
            for (size_t pad : {size_t(0), size_t(1), size_t(3)})
                for (int m = Quantized; m <= ZeroW; ++m) {
                    if (rows * cols > 3'000'000 && m != Quantized) continue;
                    auto r = run_case(t, rows, cols, nb + pad, static_cast<Mode>(m), 300 + n++);
                    worst = std::max(worst, r.gpu);
                    if (!r.ok) {
                        std::printf("  FAIL %s %zux%zu pad=%zu %s (gpu ratio %.3f, cpu ratio %.3f)\n", f.name, rows, cols,
                                    pad, mode_name(m), r.gpu, r.cpu);
                        ++failures;
                    }
                }
        }
    std::printf("  %-5s adversarial cases: %4d, worst GPU error / bound = %.4f\n", f.name, n, worst);
}

static void test_quantizer_quality() {
    const double limit[] = {0.15, 0.08, 0.04, 0.01};
    int i = 0;
    for (QuantType t : kAll) {
        const QuantFormat& f = quant_format(t);
        Rng rng(99);
        const size_t cols = 4096, rows = 32;
        std::vector<float> w(rows * cols), dq(w.size());
        std::vector<uint8_t> q(rows * cols / f.block_weights * f.block_bytes);
        for (auto& v : w) v = 0.03f * (rng.uni() + rng.uni() + rng.uni());
        for (size_t r = 0; r < rows; ++r) f.quantize(w.data() + r * cols, cols, q.data() + r * (cols / f.block_weights) * f.block_bytes);
        f.dequant_f32(q.data(), q.size() / f.block_bytes, dq.data());
        double num = 0, den = 0;
        for (size_t k = 0; k < w.size(); ++k) { num += (double(dq[k]) - w[k]) * (double(dq[k]) - w[k]); den += double(w[k]) * w[k]; }
        const double rel = std::sqrt(num / den);
        std::printf("  %-5s test quantizer relative RMSE %.4f\n", f.name, rel);
        CHECK(rel > 0.0 && rel < limit[i++]);
    }
}

static void test_registry_and_validation() {
    QuantType t;
    CHECK(quant_type_from_name("Q5_K", t) && t == QuantType::Q5_K);
    CHECK(quant_type_from_name("Q8_0", t) && t == QuantType::Q8_0);
    CHECK(!quant_type_from_name("Q2_K", t) && !quant_type_from_name("F32", t));
    Dev<uint8_t> w(16384);
    Dev<float> x(2048), y(16);
    for (QuantType q : kAll) {
        const QuantFormat& f = quant_format(q);
        const size_t cols = 2 * f.block_weights;
        CHECK(gemv_quant_cuda(q, w.p, 2, x.p, y.p, 1, cols - 1) != 0);          // not a whole number of blocks
        CHECK(gemv_quant_cuda(q, w.p, 1, x.p, y.p, 1, cols) != 0);              // stride smaller than a row
        CHECK(gemv_quant_cuda(q, w.p + 1, 2, x.p, y.p, 1, cols) != 0);          // odd weight address
        CHECK(gemv_quant_cuda(q, w.p, 2, x.p + 1, y.p, 1, cols) != 0);          // misaligned x
        CHECK(gemv_quant_cuda(q, nullptr, 2, x.p, y.p, 1, cols) != 0);
        CHECK(gemv_quant_cuda(q, w.p, 2, x.p, y.p, 0, cols) == 0);              // zero rows
        CHECK(gemv_quant_cuda(q, w.p + f.weight_align, 2, x.p, y.p, 1, cols) == 0);  // exactly the required alignment
        if (f.weight_align > 2) CHECK(gemv_quant_cuda(q, w.p + 2, 2, x.p, y.p, 1, cols) != 0);  // 16-byte formats refuse 2-aligned
    }
    CUDA(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    int n = 0;
    CUDA(cudaGetDeviceCount(&n));
    if (n == 0) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp p{};
    CUDA(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s sm_%d%d\n", p.name, p.major, p.minor);

    test_registry_and_validation();
    for (auto& fs : kFixtures) test_fixture(root + "/tests/fixtures", fs);
    test_quantizer_quality();
    for (QuantType t : kAll) test_cases(t);
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
