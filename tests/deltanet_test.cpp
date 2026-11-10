// DeltaNet layer (CPU FP32 reference) against the real transformers module with layer-0 weights.
// Requires build/golden/deltanet_layer0.bin (python -I tools/make_deltanet_golden.py).
//
//  1. Every stage (conv'd q/k/v, gates, recurrence output, gated norm, layer output, final states) vs the reference.
//  2. V-head order: GGUF-ordered weights + GGUF mapping give the same layer output as checkpoint order, and the
//     GGUF-ordered weights run with the wrong mapping do NOT (the mapping is load-bearing).
//  3. Sequence continuation: chunked / token-by-token execution is identical to one shot.
//  4. State is first-class: clone at a position, continue, restore, continue again -> identical.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "model/deltanet.h"
#include "support/golden.h"

using namespace dara::model;
using dara::test::GoldenTensor;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

// max |a-b| scaled by max |b|
static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double m = 0, s = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - b[i])); s = std::max(s, std::fabs(double(b[i]))); }
    return m / s;
}

static void expect(const char* what, const std::vector<float>& got, const GoldenTensor& want, double tol) {
    const double e = rel_err(got, want.f);
    std::printf("  %-22s max|err|/max|ref| = %.3e %s\n", what, e, e <= tol ? "" : "  <-- FAIL");
    CHECK(e <= tol);
}

static DeltaNetWeights weights_from_golden(std::map<std::string, GoldenTensor>& g) {
    DeltaNetWeights w;
    w.order = VHeadOrder::Checkpoint;  // the checkpoint's own order
    w.in_qkv = std::move(g["w.in_qkv"].f);
    w.in_z = std::move(g["w.in_z"].f);
    w.in_b = std::move(g["w.in_b"].f);
    w.in_a = std::move(g["w.in_a"].f);
    w.conv_w = std::move(g["w.conv"].f);
    w.dt_bias = std::move(g["w.dt_bias"].f);
    w.norm_w = std::move(g["w.norm"].f);
    w.out = std::move(g["w.out"].f);
    const auto& A = g["w.A_log"].f;
    w.ssm_a.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) w.ssm_a[i] = -std::exp(A[i]);
    return w;
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/deltanet_layer0.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data (%s): %s\n", path.c_str(), e.what()); return 2; }

    const int T = static_cast<int>(g["x"].dims[0]);
    const GoldenTensor x = g["x"], out_ref = g["out"];
    DeltaNetWeights w = weights_from_golden(g);
    const DeltaNetDims d = w.dims;
    std::printf("DeltaNet layer 0: T=%d tokens, %d V heads x %d, %d key heads\n", T, d.v_heads, d.v_dim, d.k_heads);

    // ---- 1. stage-by-stage vs the reference --------------------------------------------------------
    DeltaNetState st(d);
    std::vector<float> out(static_cast<size_t>(T) * d.hidden);
    DeltaNetTrace tr;
    deltanet_forward(w, st, x.f.data(), T, out.data(), &tr);
    const double tol = 2e-4;
    expect("q after conv+silu", tr.q_conv, g["q_conv"], tol);
    expect("k after conv+silu", tr.k_conv, g["k_conv"], tol);
    expect("v after conv+silu", tr.v_conv, g["v_conv"], tol);
    expect("beta", tr.beta, g["beta"], tol);
    expect("g (decay exponent)", tr.g, g["g"], tol);
    expect("recurrence output", tr.core, g["core"], tol);
    expect("gated RMSNorm", tr.normed, g["normed"], tol);
    expect("layer output", out, out_ref, tol);
    expect("final recurrent state", st.S, g["final_state"], tol);
    expect("final conv state", st.conv, g["conv_state"], tol);
    const std::vector<float> ref_out = out;
    const DeltaNetState final_ckpt_order = st;

    // ---- 2. V-head order ---------------------------------------------------------------------------
    {
        DeltaNetWeights wg = reorder_v_heads(w, VHeadOrder::Gguf);
        CHECK(wg.order == VHeadOrder::Gguf);
        // reordering twice returns the original weights exactly
        DeltaNetWeights back = reorder_v_heads(wg, VHeadOrder::Checkpoint);
        CHECK(back.in_qkv == w.in_qkv && back.out == w.out && back.ssm_a == w.ssm_a && back.conv_w == w.conv_w);

        DeltaNetState sg(d);
        std::vector<float> og(out.size());
        deltanet_forward(wg, sg, x.f.data(), T, og.data());
        const double e = rel_err(og, ref_out);
        std::printf("  GGUF order, GGUF mapping  : layer output vs checkpoint order  %.3e\n", e);
        CHECK(e <= 1e-5);
        // state heads follow the same permutation
        double se = 0;
        const size_t per = static_cast<size_t>(d.k_dim) * d.v_dim;
        for (int j = 0; j < d.v_heads; ++j) {
            const int h = checkpoint_head_of_gguf(j, d);
            for (size_t i = 0; i < per; ++i)
                se = std::max(se, std::fabs(double(sg.S[j * per + i]) - final_ckpt_order.S[static_cast<size_t>(h) * per + i]));
        }
        CHECK(se < 1e-4);

        // The mapping is load-bearing: GGUF-ordered weights with the checkpoint mapping must differ a lot.
        DeltaNetWeights wrong = wg;
        wrong.order = VHeadOrder::Checkpoint;
        DeltaNetState sw(d);
        std::vector<float> ow(out.size());
        deltanet_forward(wrong, sw, x.f.data(), 8, ow.data());
        std::vector<float> a(ow.begin(), ow.begin() + 8 * d.hidden), b(ref_out.begin(), ref_out.begin() + 8 * d.hidden);
        const double ew = rel_err(a, b);
        std::printf("  GGUF order, WRONG mapping : layer output vs checkpoint order  %.3e (must be large)\n", ew);
        CHECK(ew > 1e-2);
    }

    // ---- 3. continuation ---------------------------------------------------------------------------
    const int split = 40;
    {
        DeltaNetState a(d);
        std::vector<float> oa(out.size());
        deltanet_forward(w, a, x.f.data(), split, oa.data());
        const DeltaNetState at_split = a;
        for (int t = split; t < T; ++t)  // token-by-token decode after the prefill
            deltanet_forward(w, a, x.f.data() + static_cast<size_t>(t) * d.hidden, 1, oa.data() + static_cast<size_t>(t) * d.hidden);
        CHECK(oa == ref_out);          // identical arithmetic order: bitwise equal to the one-shot run
        CHECK(a == final_ckpt_order);
        std::printf("  prefill %d + %d single-token steps: output and state identical to one shot\n", split, T - split);

        // ---- 4. checkpoint / restore as first-class session state ----------------------------------
        auto ckpt = at_split.clone();                      // what IModelBackend::checkpoint would hold
        auto* cs = dynamic_cast<DeltaNetState*>(ckpt.get());
        CHECK(cs && cs->owner() == "deltanet-cpu");
        DeltaNetState r(*cs);                              // restore
        std::vector<float> orr(out.size());
        deltanet_forward(w, r, x.f.data() + static_cast<size_t>(split) * d.hidden, T - split,
                         orr.data() + static_cast<size_t>(split) * d.hidden);
        CHECK(r == final_ckpt_order);
        CHECK(std::equal(orr.begin() + static_cast<long>(split) * d.hidden, orr.end(), ref_out.begin() + static_cast<long>(split) * d.hidden));
        // a different continuation from the same checkpoint diverges (the restore really reset the state)
        DeltaNetState r2(*cs);
        std::vector<float> o2(static_cast<size_t>(T - split) * d.hidden);
        deltanet_forward(w, r2, x.f.data(), T - split, o2.data());   // different tokens
        CHECK(!(r2 == final_ckpt_order));
        std::printf("  checkpoint at token %d, restore, continue: identical; different continuation diverges\n", split);

        // chunked prefill with arbitrary split points
        DeltaNetState c(d);
        std::vector<float> oc(out.size());
        int pos = 0;
        for (int n : {1, 17, 3, 29, 20}) {
            deltanet_forward(w, c, x.f.data() + static_cast<size_t>(pos) * d.hidden, n, oc.data() + static_cast<size_t>(pos) * d.hidden);
            pos += n;
        }
        CHECK(pos == T && oc == ref_out && c == final_ckpt_order);
        std::printf("  prefill in chunks 1+17+3+29+20: identical to one shot\n");
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
