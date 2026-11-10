// Integrated decoder layer on the GPU vs the real transformers Qwen3_5DecoderLayer, any tier / either layer kind.
// Usage: layer_cuda_test.exe <golden file from tools/make_layer_golden.py>
// The CPU reference is validated against the same golden file by dara_layer_tests; this checks the GPU layer
// stage by stage and its determinism, chunking and state checkpoint/restore.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "model/layer_cuda.h"
#include "support/golden.h"
#include "support/layer_from_golden.h"

using namespace dara::model;
using dara::test::GoldenTensor;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0, s = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - b[i])); s = std::max(s, std::fabs(double(b[i]))); }
    return m / s;
}
static void expect(const char* what, const std::vector<float>& got, const std::vector<float>& want, double tol) {
    const double e = rel_err(got, want);
    std::printf("  %-26s max|err|/max|ref| = %.3e %s\n", what, e, e <= tol ? "" : "  <-- FAIL");
    CHECK(e <= tol);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::puts("usage: layer_cuda_test <golden.bin>"); return 2; }
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(argv[1]); }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || !n) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);

    const int T = static_cast<int>(g["x"].dims[0]);
    const std::vector<float> x = g["x"].f;
    std::map<std::string, std::vector<float>> ref;
    for (const char* k : {"normed1", "mixer", "resid1", "normed2", "mlp", "out"}) ref[k] = g[k].f;
    const bool has_state = g.count("final_state") > 0;
    const std::vector<float> ref_S = has_state ? g["final_state"].f : std::vector<float>{};
    const std::vector<float> ref_conv = has_state ? g["conv_state"].f : std::vector<float>{};
    DecoderLayerWeights w = dara::test::layer_from_golden(g);
    g.clear();  // release the host copies of the big weights before the device copies are made
    const int H = w.hidden;
    std::printf("device: %s sm_%d%d | %s layer, hidden %d, intermediate %d, T=%d\n", prop.name, prop.major, prop.minor,
                w.kind == LayerKind::DeltaNet ? "DeltaNet" : "attention", H, w.mlp.inter, T);

    try {
        GpuDecoderLayer layer(w, 256);

        // ---- 1. stages vs the reference --------------------------------------------------------
        LayerTrace tr;
        const std::vector<float> one_shot = layer.forward(x.data(), T, &tr);
        const double tol = 2e-4;
        expect("input RMSNorm", tr.normed1, ref["normed1"], tol);
        expect("mixer output", tr.mixer, ref["mixer"], tol);
        expect("first residual", tr.resid1, ref["resid1"], tol);
        expect("post-mixer RMSNorm", tr.normed2, ref["normed2"], tol);
        expect("MLP output", tr.mlp, ref["mlp"], tol);
        expect("layer output", one_shot, ref["out"], tol);

        // ---- 2. prefill + decode and chunking: bitwise identical to one shot -------------------------
        const int split = T * 2 / 3;
        {
            GpuDecoderLayer l(w, 256);
            std::vector<float> o = l.forward(x.data(), split);
            for (int t = split; t < T; ++t) { auto v = l.forward(x.data() + static_cast<size_t>(t) * H, 1); o.insert(o.end(), v.begin(), v.end()); }
            CHECK(o == one_shot);
            GpuDecoderLayer l2(w, 256);
            std::vector<float> o2;
            int pos = 0;
            for (int nn : {1, 5, T - 6}) { auto v = l2.forward(x.data() + static_cast<size_t>(pos) * H, nn); o2.insert(o2.end(), v.begin(), v.end()); pos += nn; }
            CHECK(pos == T && o2 == one_shot);
            std::printf("  prefill %d + %d decode steps, and chunks 1+5+%d: bitwise identical to one shot\n", split, T - split, T - 6);
        }

        // ---- 3. state checkpoint / restore ---------------------------------------------------------
        {
            GpuDecoderLayer l(w, 256);
            l.forward(x.data(), split);
            auto ck = l.checkpoint();
            CHECK(ck && l.position() == static_cast<size_t>(split));
            auto first = l.forward(x.data() + static_cast<size_t>(split) * H, T - split);
            l.restore(*ck);
            CHECK(l.position() == static_cast<size_t>(split));
            auto again = l.forward(x.data() + static_cast<size_t>(split) * H, T - split);
            CHECK(first == again);
            std::vector<float> tail(one_shot.begin() + static_cast<long>(split) * H, one_shot.end());
            CHECK(again == tail);
            std::printf("  checkpoint after %d tokens (%s), restore, continue: bitwise identical\n", split,
                        w.kind == LayerKind::DeltaNet ? "state copied" : "KV length");
            if (w.kind == LayerKind::DeltaNet && has_state) {
                // final state after the full sequence vs the reference
                auto full = l.checkpoint();
                expect("final recurrent state", full->S, ref_S, tol);
                expect("final conv state", full->conv, ref_conv, tol);
            }
        }
    } catch (const std::exception& e) {
        std::printf("EXCEPTION: %s\n", e.what());
        ++failures;
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
