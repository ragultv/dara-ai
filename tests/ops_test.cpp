// RMSNorm and RoPE (CPU FP32) against the real transformers modules. Requires build/golden/ops.bin
// (python -I tools/make_ops_golden.py).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model/ops.h"
#include "support/golden.h"

using namespace dara::model;
using dara::test::GoldenTensor;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

static int32_t bits(float f) { int32_t i; std::memcpy(&i, &f, 4); return i; }
static long ulp_diff(float a, float b) { return std::labs(static_cast<long>(bits(a)) - static_cast<long>(bits(b))); }

// ---- RMSNorm ------------------------------------------------------------------------------------

static void test_rmsnorm(std::map<std::string, GoldenTensor>& g, const char* tag) {
    const std::string t = tag;
    const auto& w = g["norm_" + t + ".w"].f;      // checkpoint weight (BF16 -> float): y = x_hat * (1 + w)
    const auto& X = g["norm_" + t + ".x"];
    const auto& Y = g["norm_" + t + ".y"].f;
    const auto& R = g["norm_" + t + ".inv_rms"].f;
    const int n = static_cast<int>(X.dims[1]), cases = static_cast<int>(X.dims[0]);
    const std::vector<float> scale = fold_plus_one(w);  // the GGUF convention: +1 folded in, used as-is
    const float eps = 1e-6f;
    double worst = 0, worst_r = 0;
    std::vector<float> y(n);
    for (int c = 0; c < cases; ++c) {
        const float* x = &X.f[static_cast<size_t>(c) * n];
        rmsnorm(x, scale.data(), n, eps, y.data());
        const float r = rmsnorm_inv_rms(x, n, eps);
        worst_r = std::max(worst_r, std::fabs(double(r) - R[c]) / R[c]);
        for (int i = 0; i < n; ++i) {
            const double ref = Y[static_cast<size_t>(c) * n + i];
            CHECK(std::isfinite(y[i]));
            if (ref == 0.0) { CHECK(y[i] == 0.0f); continue; }
            worst = std::max(worst, std::fabs(y[i] - ref) / std::fabs(ref));  // elementwise relative
        }
    }
    std::printf("  RMSNorm %-6s n=%d, %d cases: worst elementwise rel err %.2e, rsqrt statistic rel err %.2e\n", tag, n,
                cases, worst, worst_r);
    CHECK(worst < 1e-6 && worst_r < 1e-6);

    // The convention is load-bearing: raw w (no +1) and a double-applied (1 + (1 + w)) must both be wrong.
    const float* x0 = &X.f[0];
    std::vector<float> yr(n), yd(n);
    rmsnorm(x0, w.data(), n, eps, yr.data());                       // forgot the +1
    rmsnorm(x0, fold_plus_one(scale).data(), n, eps, yd.data());    // applied the +1 twice
    double dr = 0, dd = 0, mag = 0;
    for (int i = 0; i < n; ++i) { dr = std::max(dr, std::fabs(double(yr[i]) - Y[i])); dd = std::max(dd, std::fabs(double(yd[i]) - Y[i])); mag = std::max(mag, std::fabs(double(Y[i]))); }
    std::printf("    wrong conventions on the ordinary case: no +1 -> max abs diff %.3f, +1 twice -> %.3f (outputs ~%.2f)\n", dr, dd, mag);
    CHECK(dr > 0.05 * mag && dd > 0.05 * mag);
}

// ---- RoPE ---------------------------------------------------------------------------------------

static void test_rope(std::map<std::string, GoldenTensor>& g) {
    const auto& pos = g["rope.positions"].f;
    const auto& hf_inv = g["rope.inv_freq"].f;
    const GoldenTensor& cosg = g["rope.cos"];
    const GoldenTensor& sing = g["rope.sin"];
    const GoldenTensor& q = g["rope.q"];
    const GoldenTensor& k = g["rope.k"];
    const int T = static_cast<int>(pos.size()), rot = 64, half = rot / 2, hd = 256, QH = 16, KH = 4;
    CHECK(static_cast<int>(cosg.dims[1]) == rot);

    // inv_freq table: ours vs the reference's float32 table
    const std::vector<float> mine = rope_inv_freq(rot, 1e7);
    int exact = 0;
    long max_ulp = 0;
    for (int i = 0; i < half; ++i) {
        const long u = ulp_diff(mine[i], hf_inv[i]);
        exact += u == 0;
        max_ulp = std::max(max_ulp, u);
    }
    std::printf("  RoPE inv_freq (theta=1e7, %d frequencies): bitwise equal to the reference table for %d/%d, max %ld ulp\n",
                half, exact, half, max_ulp);

    std::vector<float> c(half), s(half), ch(half), sh(half);
    // golden duplicates halves: cos[T, 64] = [c0..c31, c0..c31]
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < half; ++i) CHECK(cosg.f[t * rot + i] == cosg.f[t * rot + i + half] && sing.f[t * rot + i] == sing.f[t * rot + i + half]);

    double worst_tab_mine = 0, worst_tab_hf = 0;
    double worst_q_hf = 0, worst_q_mine = 0;
    std::printf("  %9s | cos/sin vs ref: own table / ref table | rotated q,k vs ref: own table / ref table | fp32-angle vs exact-angle cos/sin\n", "position");
    for (int t = 0; t < T; ++t) {
        const int64_t p = static_cast<int64_t>(pos[t]);
        rope_cos_sin(p, mine, c.data(), s.data());
        rope_cos_sin(p, hf_inv, ch.data(), sh.data());
        double em = 0, eh = 0, ex = 0;
        for (int i = 0; i < half; ++i) {
            em = std::max({em, std::fabs(double(c[i]) - cosg.f[t * rot + i]), std::fabs(double(s[i]) - sing.f[t * rot + i])});
            eh = std::max({eh, std::fabs(double(ch[i]) - cosg.f[t * rot + i]), std::fabs(double(sh[i]) - sing.f[t * rot + i])});
            const double a = rope_angle_exact(p, 1.0 / std::pow(1e7, double(2 * i) / rot));
            ex = std::max({ex, std::fabs(std::cos(a) - cosg.f[t * rot + i]), std::fabs(std::sin(a) - sing.f[t * rot + i])});
        }
        worst_tab_mine = std::max(worst_tab_mine, em);
        worst_tab_hf = std::max(worst_tab_hf, eh);

        // apply to q and k at this position, with both tables
        for (int variant = 0; variant < 2; ++variant) {
            const float* cc = variant ? ch.data() : c.data();
            const float* ss = variant ? sh.data() : s.data();
            std::vector<float> qq(q.f.begin() + static_cast<long>(t) * QH * hd, q.f.begin() + static_cast<long>(t + 1) * QH * hd);
            std::vector<float> kk(k.f.begin() + static_cast<long>(t) * KH * hd, k.f.begin() + static_cast<long>(t + 1) * KH * hd);
            rope_apply(qq.data(), QH, hd, rot, cc, ss);
            rope_apply(kk.data(), KH, hd, rot, cc, ss);
            double e = 0;
            for (size_t i = 0; i < qq.size(); ++i) e = std::max(e, std::fabs(double(qq[i]) - g["rope.q_rot"].f[static_cast<size_t>(t) * QH * hd + i]));
            for (size_t i = 0; i < kk.size(); ++i) e = std::max(e, std::fabs(double(kk[i]) - g["rope.k_rot"].f[static_cast<size_t>(t) * KH * hd + i]));
            (variant ? worst_q_hf : worst_q_mine) = std::max(variant ? worst_q_hf : worst_q_mine, e);
            if (variant) {
                // dims [rot, head_dim) must be bitwise untouched
                bool untouched = true;
                for (int h = 0; h < QH; ++h)
                    for (int d = rot; d < hd; ++d) untouched &= qq[h * hd + d] == q.f[(static_cast<size_t>(t) * QH + h) * hd + d];
                CHECK(untouched);
                if (p == 0) {  // position 0 is the identity rotation
                    bool ident = true;
                    for (size_t i = 0; i < qq.size(); ++i) ident &= qq[i] == q.f[static_cast<size_t>(t) * QH * hd + i];
                    CHECK(ident);
                }
            }
        }
        std::printf("  %9lld | %9.2e / %9.2e            |", static_cast<long long>(p), em, eh);
        // per-position q error for both tables
        {
            double e0 = 0, e1 = 0;
            for (int variant = 0; variant < 2; ++variant) {
                std::vector<float> qq(q.f.begin() + static_cast<long>(t) * QH * hd, q.f.begin() + static_cast<long>(t + 1) * QH * hd);
                rope_apply(qq.data(), QH, hd, rot, variant ? ch.data() : c.data(), variant ? sh.data() : s.data());
                double e = 0;
                for (size_t i = 0; i < qq.size(); ++i) e = std::max(e, std::fabs(double(qq[i]) - g["rope.q_rot"].f[static_cast<size_t>(t) * QH * hd + i]));
                (variant ? e1 : e0) = e;
            }
            std::printf(" %9.2e / %9.2e             | %9.2e\n", e0, e1, ex);
        }
    }
    std::printf("  worst over all positions: cos/sin own-table %.2e, ref-table %.2e; rotated q/k own-table %.2e, ref-table %.2e\n",
                worst_tab_mine, worst_tab_hf, worst_q_mine, worst_q_hf);
    // With the reference's own inv_freq the fp32 pipeline must agree with the reference to rounding error.
    CHECK(worst_tab_hf < 5e-6 && worst_q_hf < 5e-5);
    // With our own table the result must too (this is the real requirement). If it fails the table must ship in the manifest.
    CHECK(worst_tab_mine < 5e-6 && worst_q_mine < 5e-5);
}

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "build/golden/ops.bin";
    std::map<std::string, GoldenTensor> g;
    try { g = dara::test::load_golden(path); }
    catch (const std::exception& e) { std::printf("cannot load golden data (%s): %s\n", path.c_str(), e.what()); return 2; }
    test_rmsnorm(g, "hidden");
    test_rmsnorm(g, "head");
    test_rope(g);
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
