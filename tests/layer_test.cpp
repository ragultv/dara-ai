// Integrated decoder layer (CPU FP32) vs the real transformers Qwen3_5DecoderLayer, any tier / either layer kind.
// Usage: dara_layer_tests <golden file from tools/make_layer_golden.py>
//
//  1. Every stage vs the reference: input norm, mixer output, first residual, post-mixer norm, MLP, layer output
//     (and, for DeltaNet layers, the final recurrent and conv state).
//  2. Prefill + decode and chunked prefill are identical to one shot.
//  3. State checkpoint/restore: DeltaNet state is cloned, the KV cache is truncated; continuing reproduces the outputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "model/layer.h"
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
    if (argc < 2) { std::puts("usage: dara_layer_tests <golden.bin>"); return 2; }
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(argv[1]); }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    const int T = static_cast<int>(g["x"].dims[0]);
    const std::vector<float> x = g["x"].f;
    const GoldenTensor st_S = g.count("final_state") ? g["final_state"] : GoldenTensor{};
    const GoldenTensor st_C = g.count("conv_state") ? g["conv_state"] : GoldenTensor{};
    std::map<std::string, std::vector<float>> ref;
    for (const char* k : {"normed1", "mixer", "resid1", "normed2", "mlp", "out"}) ref[k] = g[k].f;

    DecoderLayerWeights w = dara::test::layer_from_golden(g);
    const int H = w.hidden;
    if (w.kind == LayerKind::DeltaNet)
        std::printf("DeltaNet layer: hidden %d, intermediate %d, K heads %d, V heads %d (ratio %d), T=%d\n", H, w.mlp.inter,
                    w.dn.dims.k_heads, w.dn.dims.v_heads, w.dn.dims.ratio(), T);
    else
        std::printf("Attention layer: hidden %d, intermediate %d, heads %d / KV heads %d (groups %d), T=%d\n", H, w.mlp.inter,
                    w.at.dims.heads, w.at.dims.kv_heads, w.at.dims.groups(), T);

    // ---- 1. stages ---------------------------------------------------------------------------------
    LayerState st = make_layer_state(w, 256);
    std::vector<float> out(static_cast<size_t>(T) * H);
    LayerTrace tr;
    decoder_layer_forward(w, st, x.data(), T, out.data(), &tr);
    const double tol = 1e-4;
    expect("input RMSNorm", tr.normed1, ref["normed1"], tol);
    expect("mixer output", tr.mixer, ref["mixer"], tol);
    expect("first residual", tr.resid1, ref["resid1"], tol);
    expect("post-mixer RMSNorm", tr.normed2, ref["normed2"], tol);
    expect("MLP output", tr.mlp, ref["mlp"], tol);
    expect("layer output", tr.out, ref["out"], tol);
    if (w.kind == LayerKind::DeltaNet && st_S.numel()) {
        expect("final recurrent state", st.dn->S, st_S.f, tol);
        expect("final conv state", st.dn->conv, st_C.f, tol);
    }
    const std::vector<float> one_shot = out;

    // ---- 2. prefill + decode, chunked ---------------------------------------------------------------
    const int split = T * 2 / 3;
    {
        LayerState s = make_layer_state(w, 256);
        std::vector<float> o(out.size());
        decoder_layer_forward(w, s, x.data(), split, o.data());
        for (int t = split; t < T; ++t) decoder_layer_forward(w, s, x.data() + static_cast<size_t>(t) * H, 1, o.data() + static_cast<size_t>(t) * H);
        CHECK(o == one_shot);
        LayerState s2 = make_layer_state(w, 256);
        std::vector<float> o2(out.size());
        int pos = 0;
        for (int n : {1, 5, T - 6}) { decoder_layer_forward(w, s2, x.data() + static_cast<size_t>(pos) * H, n, o2.data() + static_cast<size_t>(pos) * H); pos += n; }
        CHECK(pos == T && o2 == one_shot);
        std::printf("  prefill %d + %d decode steps, and chunks 1+5+%d: identical to one shot\n", split, T - split, T - 6);
    }

    // ---- 3. checkpoint / restore --------------------------------------------------------------------
    {
        LayerState s = make_layer_state(w, 256);
        std::vector<float> o(out.size());
        decoder_layer_forward(w, s, x.data(), split, o.data());
        // checkpoint: DeltaNet state is copied; the KV cache only needs its length
        std::unique_ptr<DeltaNetState> ck_dn;
        size_t ck_len = 0;
        if (w.kind == LayerKind::DeltaNet) ck_dn = std::make_unique<DeltaNetState>(*s.dn);
        else ck_len = s.kv->size();
        decoder_layer_forward(w, s, x.data() + static_cast<size_t>(split) * H, T - split, o.data() + static_cast<size_t>(split) * H);
        if (w.kind == LayerKind::DeltaNet) *s.dn = *ck_dn;
        else s.kv->truncate(ck_len);
        std::vector<float> o2(out.size());
        decoder_layer_forward(w, s, x.data() + static_cast<size_t>(split) * H, T - split, o2.data() + static_cast<size_t>(split) * H);
        CHECK(std::equal(o2.begin() + static_cast<long>(split) * H, o2.end(), one_shot.begin() + static_cast<long>(split) * H));
        std::printf("  checkpoint after %d tokens (%s), restore, continue: identical\n", split, w.kind == LayerKind::DeltaNet ? "state copied" : "KV length");
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
