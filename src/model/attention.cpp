#include "model/attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "model/ops.h"

namespace dara::model {

void AttentionWeights::finalize() { inv_freq = rope_inv_freq(dims.rot_dim, dims.rope_theta); }

void KvCache::append(const float* k, const float* v) {
    if (len_ >= cap_) throw std::length_error("KvCache full");
    std::copy_n(k, d_.kv_width(), &k_[len_ * d_.kv_width()]);
    std::copy_n(v, d_.kv_width(), &v_[len_ * d_.kv_width()]);
    ++len_;
}

void KvCache::truncate(size_t n) {
    if (n > len_) throw std::out_of_range("KvCache::truncate beyond size");
    len_ = n;
}

std::vector<KvSpan> KvCache::spans(size_t start, size_t length) const {
    if (start + length > len_) throw std::out_of_range("KvCache::spans beyond size");
    if (length == 0) return {};
    return {KvSpan{&k_[start * d_.kv_width()], &v_[start * d_.kv_width()], start, length}};
}

bool KvCache::same_prefix(const KvCache& o) const {
    return len_ == o.len_ && std::equal(k_.begin(), k_.begin() + static_cast<long>(len_ * d_.kv_width()), o.k_.begin()) &&
           std::equal(v_.begin(), v_.begin() + static_cast<long>(len_ * d_.kv_width()), o.v_.begin());
}

namespace {
void matvec(const std::vector<float>& W, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* w = &W[static_cast<size_t>(r) * cols];
        double acc = 0.0;  // double accumulation: reference path
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(w[c]) * x[c];
        y[r] = static_cast<float>(acc);
    }
}
}  // namespace

void attention_forward(const AttentionWeights& w, IAttentionCache& cache, const float* x, int T, float* out, AttentionTrace* tr) {
    const AttentionDims& d = w.dims;
    if (w.inv_freq.empty()) throw std::logic_error("AttentionWeights::finalize() not called");
    const int hd = d.head_dim, H = d.heads, KH = d.kv_heads, W_ = d.out_width();
    std::vector<float> qg(d.q_rows()), q(static_cast<size_t>(H) * hd), gate(W_), k(d.kv_width()), v(d.kv_width()), attn(W_), gated(W_);
    std::vector<float> cs(d.rot_dim / 2), sn(d.rot_dim / 2), scores;
    if (tr) {
        auto sz = [&](std::vector<float>& a, size_t per) { a.assign(static_cast<size_t>(T) * per, 0.f); };
        sz(tr->gate, W_); sz(tr->q_normed, static_cast<size_t>(H) * hd); sz(tr->k_normed, d.kv_width()); sz(tr->v, d.kv_width());
        sz(tr->q_rot, static_cast<size_t>(H) * hd); sz(tr->k_rot, d.kv_width()); sz(tr->attn, W_); sz(tr->gated, W_);
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<size_t>(t) * d.hidden;
        const int64_t pos = static_cast<int64_t>(cache.size());
        matvec(w.q_proj, d.q_rows(), d.hidden, xt, qg.data());
        matvec(w.k_proj, d.kv_width(), d.hidden, xt, k.data());
        matvec(w.v_proj, d.kv_width(), d.hidden, xt, v.data());
        // split per head: [query | gate]
        for (int h = 0; h < H; ++h) {
            std::copy_n(&qg[static_cast<size_t>(h) * 2 * hd], hd, &q[static_cast<size_t>(h) * hd]);
            std::copy_n(&qg[static_cast<size_t>(h) * 2 * hd + hd], hd, &gate[static_cast<size_t>(h) * hd]);
        }
        for (int h = 0; h < H; ++h) rmsnorm(&q[static_cast<size_t>(h) * hd], w.q_norm.data(), hd, d.eps, &q[static_cast<size_t>(h) * hd]);
        for (int h = 0; h < KH; ++h) rmsnorm(&k[static_cast<size_t>(h) * hd], w.k_norm.data(), hd, d.eps, &k[static_cast<size_t>(h) * hd]);
        if (tr) {
            std::copy(q.begin(), q.end(), &tr->q_normed[static_cast<size_t>(t) * H * hd]);
            std::copy(k.begin(), k.end(), &tr->k_normed[static_cast<size_t>(t) * d.kv_width()]);
            std::copy(v.begin(), v.end(), &tr->v[static_cast<size_t>(t) * d.kv_width()]);
            std::copy(gate.begin(), gate.end(), &tr->gate[static_cast<size_t>(t) * W_]);
        }
        rope_cos_sin(pos, w.inv_freq, cs.data(), sn.data());
        rope_apply(q.data(), H, hd, d.rot_dim, cs.data(), sn.data());
        rope_apply(k.data(), KH, hd, d.rot_dim, cs.data(), sn.data());
        if (tr) {
            std::copy(q.begin(), q.end(), &tr->q_rot[static_cast<size_t>(t) * H * hd]);
            std::copy(k.begin(), k.end(), &tr->k_rot[static_cast<size_t>(t) * d.kv_width()]);
        }
        cache.append(k.data(), v.data());

        const size_t n = cache.size();  // attend over positions 0..pos (causal: nothing beyond this token exists yet)
        scores.resize(n);
        const std::vector<KvSpan> spans = cache.spans(0, n);  // the cache is read through spans, never as a flat array
        for (int h = 0; h < H; ++h) {
            const int kh = h / d.groups();
            const float* qh = &q[static_cast<size_t>(h) * hd];
            float mx = -INFINITY;
            size_t j = 0;
            for (const KvSpan& sp : spans)
                for (size_t jj = 0; jj < sp.count; ++jj, ++j) {
                    const float* kj = sp.k + jj * d.kv_width() + static_cast<size_t>(kh) * hd;
                    double s = 0.0;
                    for (int i = 0; i < hd; ++i) s += static_cast<double>(qh[i]) * kj[i];
                    scores[j] = static_cast<float>(s) * scale;
                    mx = std::max(mx, scores[j]);
                }
            double denom = 0.0;
            for (size_t jj = 0; jj < n; ++jj) { scores[jj] = std::exp(scores[jj] - mx); denom += scores[jj]; }
            for (int i = 0; i < hd; ++i) {
                double o = 0.0;
                size_t jv = 0;
                for (const KvSpan& sp : spans)
                    for (size_t jj = 0; jj < sp.count; ++jj, ++jv)
                        o += static_cast<double>(scores[jv]) * sp.v[jj * d.kv_width() + static_cast<size_t>(kh) * hd + i];
                attn[static_cast<size_t>(h) * hd + i] = static_cast<float>(o / denom);
            }
        }
        for (int i = 0; i < W_; ++i) gated[i] = attn[i] * (1.0f / (1.0f + std::exp(-gate[i])));
        if (tr) {
            std::copy(attn.begin(), attn.end(), &tr->attn[static_cast<size_t>(t) * W_]);
            std::copy(gated.begin(), gated.end(), &tr->gated[static_cast<size_t>(t) * W_]);
        }
        matvec(w.o_proj, d.hidden, W_, gated.data(), out + static_cast<size_t>(t) * d.hidden);
    }
}

}  // namespace dara::model
