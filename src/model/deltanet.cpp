#include "model/deltanet.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dara::model {

int key_head_for_v(VHeadOrder order, int v, const DeltaNetDims& d) {
    return order == VHeadOrder::Checkpoint ? v / d.ratio() : v % d.k_heads;
}

// Checkpoint head h = ratio*kg + r  <->  GGUF head j = k_heads*r + kg.
int gguf_head_of_checkpoint(int h, const DeltaNetDims& d) { return d.k_heads * (h % d.ratio()) + h / d.ratio(); }
int checkpoint_head_of_gguf(int j, const DeltaNetDims& d) { return d.ratio() * (j % d.k_heads) + j / d.k_heads; }

DeltaNetState::DeltaNetState(const DeltaNetDims& d)
    : dims(d),
      conv(static_cast<size_t>(d.conv_dim()) * (d.conv_kernel - 1), 0.0f),
      S(static_cast<size_t>(d.v_heads) * d.k_dim * d.v_dim, 0.0f) {}

DeltaNetWeights reorder_v_heads(const DeltaNetWeights& w, VHeadOrder target) {
    if (w.order == target) return w;
    const DeltaNetDims& d = w.dims;
    DeltaNetWeights r = w;
    r.order = target;
    // dst head index -> src head index (in w's order).
    auto src_of = [&](int dst) { return target == VHeadOrder::Gguf ? checkpoint_head_of_gguf(dst, d) : gguf_head_of_checkpoint(dst, d); };
    const size_t H = d.hidden, vd = d.v_dim, kd2 = 2 * static_cast<size_t>(d.key_dim());
    for (int h = 0; h < d.v_heads; ++h) {
        const int s = src_of(h);
        for (size_t i = 0; i < vd; ++i) {
            std::copy_n(&w.in_qkv[(kd2 + static_cast<size_t>(s) * vd + i) * H], H, &r.in_qkv[(kd2 + static_cast<size_t>(h) * vd + i) * H]);
            std::copy_n(&w.in_z[(static_cast<size_t>(s) * vd + i) * H], H, &r.in_z[(static_cast<size_t>(h) * vd + i) * H]);
            std::copy_n(&w.conv_w[(kd2 + static_cast<size_t>(s) * vd + i) * d.conv_kernel], d.conv_kernel,
                        &r.conv_w[(kd2 + static_cast<size_t>(h) * vd + i) * d.conv_kernel]);
            for (int row = 0; row < d.hidden; ++row)
                r.out[static_cast<size_t>(row) * d.value_dim() + static_cast<size_t>(h) * vd + i] =
                    w.out[static_cast<size_t>(row) * d.value_dim() + static_cast<size_t>(s) * vd + i];
        }
        std::copy_n(&w.in_b[static_cast<size_t>(s) * H], H, &r.in_b[static_cast<size_t>(h) * H]);
        std::copy_n(&w.in_a[static_cast<size_t>(s) * H], H, &r.in_a[static_cast<size_t>(h) * H]);
        r.ssm_a[h] = w.ssm_a[s];
        r.dt_bias[h] = w.dt_bias[s];
    }
    return r;
}

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float siluf(float x) { return x * sigmoidf(x); }
inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // torch defaults

void matvec(const std::vector<float>& W, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* w = &W[static_cast<size_t>(r) * cols];
        double acc = 0.0;  // double accumulation: this is the reference path
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(w[c]) * x[c];
        y[r] = static_cast<float>(acc);
    }
}

}  // namespace

void deltanet_project(const DeltaNetWeights& w, const float* x, float* qkv, float* z, float* b, float* a) {
    const DeltaNetDims& d = w.dims;
    matvec(w.in_qkv, d.conv_dim(), d.hidden, x, qkv);
    matvec(w.in_z, d.value_dim(), d.hidden, x, z);
    matvec(w.in_b, d.v_heads, d.hidden, x, b);
    matvec(w.in_a, d.v_heads, d.hidden, x, a);
}

void deltanet_out_project(const DeltaNetWeights& w, const float* normed, float* out) {
    matvec(w.out, w.dims.hidden, w.dims.value_dim(), normed, out);
}

void deltanet_forward(const DeltaNetWeights& w, DeltaNetState& st, const float* x, int T, float* out, DeltaNetTrace* tr) {
    const DeltaNetDims& d = w.dims;
    if (st.dims.conv_dim() != d.conv_dim() || st.S.size() != static_cast<size_t>(d.v_heads) * d.k_dim * d.v_dim)
        throw std::invalid_argument("deltanet: state does not match weights");
    const int cd = d.conv_dim(), kd = d.key_dim(), vd = d.value_dim(), taps = d.conv_kernel, hist = taps - 1;
    std::vector<float> qkv(cd), conv(cd), z(vd), b(d.v_heads), a(d.v_heads), core(vd), normed(vd);
    std::vector<float> qn(static_cast<size_t>(d.k_heads) * d.k_dim), kn(qn.size());
    if (tr) {
        tr->q_conv.assign(static_cast<size_t>(T) * d.v_heads * d.k_dim, 0.f);
        tr->k_conv = tr->q_conv;
        tr->v_conv.assign(static_cast<size_t>(T) * d.v_heads * d.v_dim, 0.f);
        tr->beta.assign(static_cast<size_t>(T) * d.v_heads, 0.f);
        tr->g = tr->beta;
        tr->core.assign(static_cast<size_t>(T) * vd, 0.f);
        tr->normed = tr->core;
    }
    const float qscale = 1.0f / std::sqrt(static_cast<float>(d.k_dim));

    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<size_t>(t) * d.hidden;
        matvec(w.in_qkv, cd, d.hidden, xt, qkv.data());
        matvec(w.in_z, vd, d.hidden, xt, z.data());
        matvec(w.in_b, d.v_heads, d.hidden, xt, b.data());
        matvec(w.in_a, d.v_heads, d.hidden, xt, a.data());

        // Causal depthwise conv: window = [3 stored inputs, current input], tap 0 on the oldest.
        for (int c = 0; c < cd; ++c) {
            float* h = &st.conv[static_cast<size_t>(c) * hist];
            const float* wc = &w.conv_w[static_cast<size_t>(c) * taps];
            float acc = 0.0f;
            for (int k = 0; k < hist; ++k) acc += wc[k] * h[k];
            acc += wc[taps - 1] * qkv[c];
            conv[c] = siluf(acc);
            for (int k = 0; k + 1 < hist; ++k) h[k] = h[k + 1];
            h[hist - 1] = qkv[c];
        }

        const float* qc = conv.data();
        const float* kc = conv.data() + kd;
        const float* vc = conv.data() + 2 * kd;

        // l2norm(q), l2norm(k) per key head, then q scaling.
        for (int h = 0; h < d.k_heads; ++h) {
            double sq = 0, sk = 0;
            for (int i = 0; i < d.k_dim; ++i) {
                sq += static_cast<double>(qc[h * d.k_dim + i]) * qc[h * d.k_dim + i];
                sk += static_cast<double>(kc[h * d.k_dim + i]) * kc[h * d.k_dim + i];
            }
            const float rq = static_cast<float>(1.0 / std::sqrt(sq + 1e-6)), rk = static_cast<float>(1.0 / std::sqrt(sk + 1e-6));
            for (int i = 0; i < d.k_dim; ++i) {
                qn[h * d.k_dim + i] = qc[h * d.k_dim + i] * rq * qscale;
                kn[h * d.k_dim + i] = kc[h * d.k_dim + i] * rk;
            }
        }

        for (int h = 0; h < d.v_heads; ++h) {
            const int kh = key_head_for_v(w.order, h, d);
            const float beta = sigmoidf(b[h]);
            const float g = w.ssm_a[h] * softplusf(a[h] + w.dt_bias[h]);
            const float decay = std::exp(g);
            float* S = &st.S[static_cast<size_t>(h) * d.k_dim * d.v_dim];
            const float* k = &kn[static_cast<size_t>(kh) * d.k_dim];
            const float* q = &qn[static_cast<size_t>(kh) * d.k_dim];
            const float* v = vc + static_cast<size_t>(h) * d.v_dim;
            if (tr) {
                const size_t o = (static_cast<size_t>(t) * d.v_heads + h);
                tr->beta[o] = beta;
                tr->g[o] = g;
                // raw (pre-l2norm) q/k as the reference sees them after repeat_interleave
                std::copy_n(qc + static_cast<size_t>(kh) * d.k_dim, d.k_dim, &tr->q_conv[o * d.k_dim]);
                std::copy_n(kc + static_cast<size_t>(kh) * d.k_dim, d.k_dim, &tr->k_conv[o * d.k_dim]);
                std::copy_n(v, d.v_dim, &tr->v_conv[o * d.v_dim]);
            }
            for (int vi = 0; vi < d.v_dim; ++vi) {
                double kv = 0.0;
                for (int ki = 0; ki < d.k_dim; ++ki) {
                    float& s = S[static_cast<size_t>(ki) * d.v_dim + vi];
                    s *= decay;
                    kv += static_cast<double>(s) * k[ki];
                }
                const float delta = (v[vi] - static_cast<float>(kv)) * beta;
                double o = 0.0;
                for (int ki = 0; ki < d.k_dim; ++ki) {
                    float& s = S[static_cast<size_t>(ki) * d.v_dim + vi];
                    s += k[ki] * delta;
                    o += static_cast<double>(s) * q[ki];
                }
                core[static_cast<size_t>(h) * d.v_dim + vi] = static_cast<float>(o);
            }
        }

        // Gated RMSNorm per V head: (o * rsqrt(mean(o^2) + eps)) * w * silu(z)
        for (int h = 0; h < d.v_heads; ++h) {
            const float* o = &core[static_cast<size_t>(h) * d.v_dim];
            double ms = 0.0;
            for (int i = 0; i < d.v_dim; ++i) ms += static_cast<double>(o[i]) * o[i];
            const float r = static_cast<float>(1.0 / std::sqrt(ms / d.v_dim + d.eps));
            for (int i = 0; i < d.v_dim; ++i)
                normed[static_cast<size_t>(h) * d.v_dim + i] = w.norm_w[i] * (o[i] * r) * siluf(z[static_cast<size_t>(h) * d.v_dim + i]);
        }
        if (tr) {
            std::copy(core.begin(), core.end(), &tr->core[static_cast<size_t>(t) * vd]);
            std::copy(normed.begin(), normed.end(), &tr->normed[static_cast<size_t>(t) * vd]);
        }
        matvec(w.out, d.hidden, vd, normed.data(), out + static_cast<size_t>(t) * d.hidden);
    }
}

}  // namespace dara::model
