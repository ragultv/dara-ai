#include "model/layer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "model/ops.h"

namespace dara::model {

LayerState make_layer_state(const DecoderLayerWeights& w, size_t kv_capacity) {
    LayerState s;
    s.kind = w.kind;
    if (w.kind == LayerKind::DeltaNet) s.dn = std::make_unique<DeltaNetState>(w.dn.dims);
    else s.kv = std::make_unique<KvCache>(w.at.dims, kv_capacity);
    return s;
}

namespace {
void matvec(const std::vector<float>& W, int rows, int cols, const float* x, float* y) {
    for (int r = 0; r < rows; ++r) {
        const float* w = &W[static_cast<size_t>(r) * cols];
        double acc = 0.0;
        for (int c = 0; c < cols; ++c) acc += static_cast<double>(w[c]) * x[c];
        y[r] = static_cast<float>(acc);
    }
}
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }
}  // namespace

void decoder_layer_forward(const DecoderLayerWeights& w, LayerState& st, const float* x, int T, float* out, LayerTrace* tr) {
    if (st.kind != w.kind) throw std::invalid_argument("layer state kind does not match the layer");
    const int H = w.hidden, I = w.mlp.inter;
    if (tr) {
        for (auto* v : {&tr->normed1, &tr->mixer, &tr->resid1, &tr->normed2, &tr->mlp, &tr->out}) v->assign(static_cast<size_t>(T) * H, 0.f);
    }
    std::vector<float> h1(H), m(H), x1(H), h2(H), g(I), u(I), f(H);
    for (int t = 0; t < T; ++t) {
        const float* xt = x + static_cast<size_t>(t) * H;
        rmsnorm(xt, w.input_norm.data(), H, w.eps, h1.data());
        if (w.kind == LayerKind::DeltaNet) {
            if (w.dn.dims.hidden != H) throw std::invalid_argument("DeltaNet hidden size mismatch");
            deltanet_forward(w.dn, *st.dn, h1.data(), 1, m.data());
        } else {
            if (w.at.dims.hidden != H) throw std::invalid_argument("attention hidden size mismatch");
            attention_forward(w.at, *st.kv, h1.data(), 1, m.data());
        }
        for (int i = 0; i < H; ++i) x1[i] = xt[i] + m[i];
        rmsnorm(x1.data(), w.post_norm.data(), H, w.eps, h2.data());
        matvec(w.mlp.gate, I, H, h2.data(), g.data());
        matvec(w.mlp.up, I, H, h2.data(), u.data());
        for (int i = 0; i < I; ++i) g[i] = siluf(g[i]) * u[i];
        matvec(w.mlp.down, H, I, g.data(), f.data());
        float* o = out + static_cast<size_t>(t) * H;
        for (int i = 0; i < H; ++i) o[i] = x1[i] + f[i];
        if (tr) {
            const size_t b = static_cast<size_t>(t) * H;
            std::copy(h1.begin(), h1.end(), &tr->normed1[b]);
            std::copy(m.begin(), m.end(), &tr->mixer[b]);
            std::copy(x1.begin(), x1.end(), &tr->resid1[b]);
            std::copy(h2.begin(), h2.end(), &tr->normed2[b]);
            std::copy(f.begin(), f.end(), &tr->mlp[b]);
            std::copy(o, o + H, &tr->out[b]);
        }
    }
}

}  // namespace dara::model
