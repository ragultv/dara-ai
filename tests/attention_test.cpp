// Full-attention layer + KV cache (CPU FP32) against the real transformers Qwen3_5Attention with layer-3 weights.
// Requires build/golden/attention_layer3.bin (python -I tools/make_attention_golden.py).
//
//  1. Every stage vs the reference: gate split, q/k norm, RoPE, attention (before the gate), gated, layer output.
//  2. Prefill then single-token decode == one shot == chunked prefill == the reference.
//  3. Causality: perturbing later tokens cannot change earlier outputs.
//  4. KV cache: checkpoint = length; truncate + continue reproduces the same outputs; a different continuation
//     overwrites cleanly; bounds are enforced.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/attention.h"
#include "model/ops.h"
#include "support/golden.h"

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
static void expect(const char* what, const std::vector<float>& got, const GoldenTensor& want, double tol) {
    const double e = rel_err(got, want.f);
    std::printf("  %-26s max|err|/max|ref| = %.3e %s\n", what, e, e <= tol ? "" : "  <-- FAIL");
    CHECK(e <= tol);
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/attention_layer3.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data (%s): %s\n", path.c_str(), e.what()); return 2; }

    AttentionWeights w;
    w.q_proj = g["w.q_proj"].f; w.k_proj = g["w.k_proj"].f; w.v_proj = g["w.v_proj"].f; w.o_proj = g["w.o_proj"].f;
    w.q_norm = fold_plus_one(g["w.q_norm"].f);  // GGUF convention: (1 + w) folded, used as-is
    w.k_norm = fold_plus_one(g["w.k_norm"].f);
    w.finalize();
    const AttentionDims d = w.dims;
    const int T = static_cast<int>(g["x"].dims[0]);
    const std::vector<float>& x = g["x"].f;
    std::printf("Attention layer 3: T=%d tokens, %d heads / %d KV heads, head_dim %d, rotary %d\n", T, d.heads, d.kv_heads, d.head_dim, d.rot_dim);

    // ---- 1. stages --------------------------------------------------------------------------------
    KvCache cache(d, 256);
    std::vector<float> out(static_cast<size_t>(T) * d.hidden);
    AttentionTrace tr;
    attention_forward(w, cache, x.data(), T, out.data(), &tr);
    const double tol = 5e-5;
    expect("gate (split)", tr.gate, g["gate"], tol);
    expect("q after q_norm", tr.q_normed, g["q_normed"], tol);
    expect("k after k_norm", tr.k_normed, g["k_normed"], tol);
    expect("v", tr.v, g["v"], tol);
    expect("q after RoPE", tr.q_rot, g["q_rot"], tol);
    expect("k after RoPE", tr.k_rot, g["k_rot"], tol);
    expect("attention (pre-gate)", tr.attn, g["attn"], tol);
    expect("gated (post-sigmoid)", tr.gated, g["gated"], tol);
    expect("layer output", out, g["out"], tol);
    CHECK(cache.size() == static_cast<size_t>(T));
    const std::vector<float> ref_out = out;

    // ---- 2. prefill + decode == one shot == chunked ------------------------------------------------
    const int split = 40;
    {
        KvCache c(d, 256);
        std::vector<float> o(out.size());
        attention_forward(w, c, x.data(), split, o.data());                         // prefill
        for (int t = split; t < T; ++t)                                             // then single-token decode
            attention_forward(w, c, x.data() + static_cast<size_t>(t) * d.hidden, 1, o.data() + static_cast<size_t>(t) * d.hidden);
        CHECK(o == ref_out && c.same_prefix(cache));
        std::printf("  prefill %d + %d decode steps: identical to one shot\n", split, T - split);

        KvCache c2(d, 256);
        std::vector<float> o2(out.size());
        int pos = 0;
        for (int n : {1, 17, 3, 29, 20}) {
            attention_forward(w, c2, x.data() + static_cast<size_t>(pos) * d.hidden, n, o2.data() + static_cast<size_t>(pos) * d.hidden);
            pos += n;
        }
        CHECK(pos == T && o2 == ref_out && c2.same_prefix(cache));
        std::printf("  prefill in chunks 1+17+3+29+20: identical to one shot\n");
    }

    // ---- 3. causality -----------------------------------------------------------------------------
    {
        std::vector<float> x2 = x;
        for (size_t i = static_cast<size_t>(30) * d.hidden; i < x2.size(); ++i) x2[i] += 3.0f;  // perturb tokens 30..T-1
        KvCache c(d, 256);
        std::vector<float> o(out.size());
        attention_forward(w, c, x2.data(), T, o.data());
        CHECK(std::equal(o.begin(), o.begin() + 30 * d.hidden, ref_out.begin()));            // earlier tokens unchanged
        CHECK(!std::equal(o.begin() + 30 * d.hidden, o.end(), ref_out.begin() + 30 * d.hidden));  // later ones did change
        std::printf("  causality: tokens before 30 unaffected by perturbing tokens 30+\n");
    }

    // ---- 4. KV cache checkpoint / restore / bounds -------------------------------------------------
    {
        KvCache c(d, 256);
        std::vector<float> o(out.size());
        attention_forward(w, c, x.data(), split, o.data());
        const size_t mark = c.size();                                                // checkpoint = a length
        attention_forward(w, c, x.data() + static_cast<size_t>(split) * d.hidden, T - split, o.data() + static_cast<size_t>(split) * d.hidden);
        CHECK(c.size() == static_cast<size_t>(T));
        c.truncate(mark);                                                            // restore = truncate
        CHECK(c.size() == static_cast<size_t>(split));
        std::vector<float> o2(out.size());
        attention_forward(w, c, x.data() + static_cast<size_t>(split) * d.hidden, T - split, o2.data() + static_cast<size_t>(split) * d.hidden);
        CHECK(std::equal(o2.begin() + static_cast<long>(split) * d.hidden, o2.end(), ref_out.begin() + static_cast<long>(split) * d.hidden));
        CHECK(c.same_prefix(cache));
        // a different continuation after truncation really replaces the old entries
        c.truncate(mark);
        std::vector<float> o3(static_cast<size_t>(T - split) * d.hidden);
        attention_forward(w, c, x.data(), T - split, o3.data());                     // different tokens at the same positions
        KvCache fresh(d, 256);
        std::vector<float> o4(static_cast<size_t>(split) * d.hidden);
        attention_forward(w, fresh, x.data(), split, o4.data());
        CHECK(!c.same_prefix(cache));
        std::printf("  KV checkpoint at %zu: truncate + continue reproduces outputs and cache; a new continuation replaces entries\n", mark);

        bool threw_full = false, threw_trunc = false;
        KvCache tiny(d, 2);
        std::vector<float> kk(d.kv_width()), vv(d.kv_width());
        tiny.append(kk.data(), vv.data()); tiny.append(kk.data(), vv.data());
        try { tiny.append(kk.data(), vv.data()); } catch (const std::length_error&) { threw_full = true; }
        try { tiny.truncate(5); } catch (const std::out_of_range&) { threw_trunc = true; }
        CHECK(threw_full && threw_trunc);
    }

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
