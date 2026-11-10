// Per-token matrix workload for the real Qwen3.5-4B Q4_K_M, from the GGUF header's tensor inventory.
//
// Every 2-D weight tensor is read once per decoded token, so the matrix time per token is the sum over
// tensors of the GEMV time for that tensor's shape. Distinct (type, rows, cols) shapes are timed and
// weighted by how many tensors share them. Tensors whose type has no kernel yet (Q5_K, Q8_0) are
// counted as uncovered bytes, not guessed at. (All four formats in this model now have kernels.)
//
// Each shape is timed in 3 passes; GPU clocks, temperature, power and throttle reasons are sampled right
// after each timed loop so every number can be judged against the conditions it ran under.
// Usage: workload_bench.exe <repo root>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "core/gpu_telemetry.h"
#include "kernels/quant.h"

using namespace dara::kernels;

#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA error %s line %d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { CUDA(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

struct Rng {
    uint64_t s = 88172645463325252ull;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return static_cast<float>(static_cast<double>(next() >> 11) / 4503599627370496.0 - 1.0); }
};

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Telemetry {
    unsigned min_sm = ~0u, max_temp = 0;
    double max_power = 0;
    uint64_t throttle = 0;
    bool any = false;
    void add(const dara::core::GpuTelemetry& t) {
        any = true;
        min_sm = std::min(min_sm, t.sm_mhz);
        max_temp = std::max(max_temp, t.temp_c);
        max_power = std::max(max_power, t.power_w);
        throttle |= t.throttle_mask;
    }
};

// Valid random blocks, tiled from a pool so building a 500 MB head matrix is fast. Bandwidth does not
// depend on the values; set_scales keeps the fp16 scales finite.
static std::vector<uint8_t> build_matrix(const QuantFormat& f, size_t nblocks) {
    std::vector<uint8_t> pool(4096 * static_cast<size_t>(f.block_bytes));
    Rng rng;
    for (size_t i = 0; i < 4096; ++i) {
        uint8_t* b = pool.data() + i * f.block_bytes;
        for (int k = 0; k < f.block_bytes; ++k) b[k] = static_cast<uint8_t>(rng.next() >> 33);
        f.set_scales(b, 0.002f, 0.002f);
    }
    std::vector<uint8_t> out(nblocks * f.block_bytes);
    for (size_t i = 0; i < nblocks; ++i)
        std::memcpy(out.data() + i * f.block_bytes, pool.data() + (i % 4096) * f.block_bytes, f.block_bytes);
    return out;
}

// Median per-launch time in microseconds over `iters` events-timed launches after warm-up.
template <class Launch>
static double time_us(Launch&& launch, int iters) {
    for (int i = 0; i < 20; ++i) launch();
    CUDA(cudaDeviceSynchronize());
    std::vector<cudaEvent_t> ev(2 * iters);
    for (auto& e : ev) CUDA(cudaEventCreate(&e));
    for (int i = 0; i < iters; ++i) {
        CUDA(cudaEventRecord(ev[2 * i]));
        launch();
        CUDA(cudaEventRecord(ev[2 * i + 1]));
    }
    CUDA(cudaEventSynchronize(ev.back()));
    std::vector<double> t(iters);
    for (int i = 0; i < iters; ++i) {
        float ms = 0;
        CUDA(cudaEventElapsedTime(&ms, ev[2 * i], ev[2 * i + 1]));
        t[i] = ms * 1000.0;
    }
    for (auto& e : ev) cudaEventDestroy(e);
    return median(t);
}

struct Shape {
    std::string type;
    uint64_t rows = 0, cols = 0;
    int count = 0;
    double tensor_bytes = 0;
    std::vector<double> pass_us;
    Telemetry tel;
};

static double type_bytes(const std::string& t, uint64_t elems) {
    QuantType q;
    if (quant_type_from_name(t, q)) { const auto& f = quant_format(q); return double(elems) / f.block_weights * f.block_bytes; }
    if (t == "F16" || t == "BF16") return elems * 2.0;
    return elems * 4.0;  // F32
}

static void print_telemetry_line(const char* label) {
    if (auto t = dara::core::query_gpu_telemetry())
        std::printf("  %s: SM %u/%u MHz, mem %u MHz, %u C, %.1f W, throttle: %s\n", label, t->sm_mhz, t->max_sm_mhz,
                    t->mem_mhz, t->temp_c, t->power_w, t->throttle_text().c_str());
    else
        std::printf("  %s: telemetry unavailable\n", label);
}

int main(int argc, char** argv) {
    const std::string root = argc > 1 ? argv[1] : ".";
    std::ifstream in(root + "/manifests/qwen3.5-4b-q4_k_m.tensors.json");
    if (!in) { std::puts("cannot open tensor inventory"); return 1; }
    nlohmann::json j;
    in >> j;

    std::map<std::tuple<std::string, uint64_t, uint64_t>, Shape> shapes;
    std::map<std::string, double> total_bytes, covered_bytes;
    double all_bytes = 0;
    for (auto& t : j.at("tensors")) {
        const std::string type = t.at("type");
        const auto& dims = t.at("dims");
        uint64_t elems = 1;
        for (auto& d : dims) elems *= d.get<uint64_t>();
        const double bytes = type_bytes(type, elems);
        total_bytes[type] += bytes;
        all_bytes += bytes;
        if (dims.size() != 2) continue;  // norms and biases: negligible traffic
        const uint64_t cols = dims[0], rows = dims[1];
        auto& s = shapes[{type, rows, cols}];
        s.type = type; s.rows = rows; s.cols = cols; s.count++; s.tensor_bytes = bytes;
        QuantType qt;
        if (quant_type_from_name(type, qt)) covered_bytes[type] += bytes;
    }

    int dev = 0;
    CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s sm_%d%d\n", prop.name, prop.major, prop.minor);
    print_telemetry_line("start ");

    // Warm the GPU to steady clocks before measuring anything.
    {
        Dev<float4> buf((64ull << 20) / sizeof(float4));
        CUDA(cudaMemset(buf.p, 1, 64ull << 20));
        for (int i = 0; i < 200; ++i) CUDA(cudaMemsetAsync(buf.p, i & 0xff, 64ull << 20));
        CUDA(cudaDeviceSynchronize());
    }

    constexpr int kPasses = 3;
    std::vector<Shape*> order;
    for (auto& [k, s] : shapes) {
        QuantType qt;
        if (quant_type_from_name(s.type, qt)) order.push_back(&s);
    }

    for (int pass = 0; pass < kPasses; ++pass) {
        for (Shape* s : order) {
            QuantType qt;
            quant_type_from_name(s->type, qt);
            const QuantFormat& f = quant_format(qt);
            const size_t nb = s->cols / f.block_weights;
            std::vector<uint8_t> hW = build_matrix(f, s->rows * nb);
            std::vector<float> hx(s->cols);
            Rng rng;
            for (auto& v : hx) v = rng.uni();
            Dev<uint8_t> dW(hW.size());
            Dev<float> dx(s->cols), dy(s->rows);
            CUDA(cudaMemcpy(dW.p, hW.data(), hW.size(), cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(dx.p, hx.data(), s->cols * 4, cudaMemcpyHostToDevice));
            const int iters = s->rows * s->cols > 100'000'000ull ? 30 : 150;
            double us = time_us([&] { gemv_quant_cuda(qt, dW.p, nb, dx.p, dy.p, s->rows, s->cols); }, iters);
            s->pass_us.push_back(us);
            if (auto t = dara::core::query_gpu_telemetry()) s->tel.add(*t);
        }
    }
    print_telemetry_line("end   ");

    std::printf("\n  %-5s %-12s %5s | %9s %8s %7s | %6s %6s %6s %8s  %s\n", "type", "rows x cols", "count", "median us",
                "spread", "GB/s", "minSM", "temp", "power", "", "throttle");
    double total_us = 0;
    std::map<std::string, double> us_by_type;
    for (Shape* s : order) {
        const double med = median(s->pass_us);
        const double lo = *std::min_element(s->pass_us.begin(), s->pass_us.end());
        const double hi = *std::max_element(s->pass_us.begin(), s->pass_us.end());
        const double gbs = (s->tensor_bytes + s->cols * 4.0 + s->rows * 4.0) / (med * 1e-6) / 1e9;
        const double w = med * s->count;
        total_us += w;
        us_by_type[s->type] += w;
        char shp[32];
        std::snprintf(shp, sizeof shp, "%llux%llu", (unsigned long long)s->rows, (unsigned long long)s->cols);
        dara::core::GpuTelemetry tmp;
        tmp.throttle_mask = s->tel.throttle;
        std::printf("  %-5s %-12s %5d | %9.1f %7.0f%% %7.1f | %6u %5uC %5.0fW %8s  %s\n", s->type.c_str(), shp, s->count, med,
                    100.0 * (hi - lo) / med, gbs, s->tel.min_sm, s->tel.max_temp, s->tel.max_power, "",
                    tmp.throttle_text().c_str());
    }

    double covered = 0;
    for (auto& [t, b] : covered_bytes) covered += b;
    std::printf("\n  weight bytes by type (whole model, %.0f MiB):", all_bytes / 1048576.0);
    for (auto& [t, b] : total_bytes) {
        QuantType q;
        std::printf("  %s %.0f MiB%s", t.c_str(), b / 1048576.0, quant_type_from_name(t, q) ? "*" : "");
    }
    std::printf("\n  (* = has a GEMV kernel) covered: %.2f%% of all weight bytes; the rest are 1-D F32 vectors (norms, SSM\n"
                "  parameters) that need elementwise kernels, not GEMV\n", 100.0 * covered / all_bytes);
    std::printf("  estimated matrix time per token, all GEMV tensors: %.2f ms  (", total_us / 1000.0);
    for (auto& [t, u] : us_by_type) std::printf("%s %.2f ms  ", t.c_str(), u / 1000.0);
    std::printf(")\n  = %.1f tokens/s ceiling from the GEMVs alone (excludes attention, DeltaNet recurrence, norms,\n"
                "    sampling and launch gaps; not a model speed)\n", 1e6 / total_us);
    return 0;
}
