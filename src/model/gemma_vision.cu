#include "model/gemma_vision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels/half.h"
#include "model/gemma_ops_cuda.h"
#include "model/ops_cuda.h"
#include "model/vision_ops_cuda.h"

namespace dara::model {
namespace {

std::string meta(const dara::io::GgufCheckpoint& g, const std::string& key, const std::string& fallback = "") {
    for (const auto& kv : g.header().metadata) if (kv.first == key) return kv.second;
    if (!fallback.empty()) return fallback;
    throw std::runtime_error("mmproj GGUF metadata is missing " + key);
}

// F16 (or F32) tensor -> BF16 bits: exact for weights that were BF16 before the F16 conversion
std::vector<uint16_t> bf16_bits(const dara::io::GgufCheckpoint& g, const std::string& name) {
    const auto& info = g.info(name);
    const std::vector<uint8_t> raw = g.read(name);
    const size_t n = info.type == 1 ? raw.size() / 2 : raw.size() / 4;
    std::vector<uint16_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        float f;
        if (info.type == 1) { uint16_t h; std::memcpy(&h, &raw[i * 2], 2); f = dara::kernels::half_to_float(h); }
        else if (info.type == 0) std::memcpy(&f, &raw[i * 4], 4);
        else throw std::runtime_error("mmproj tensor " + name + ": expected F16 or F32");
        out[i] = dara::kernels::float_to_bf16(f);
    }
    return out;
}

std::vector<float> f32(const dara::io::GgufCheckpoint& g, const std::string& name) {
    const auto& info = g.info(name);
    if (info.type == 0) return g.read_f32(name);
    if (info.type != 1) throw std::runtime_error("mmproj tensor " + name + ": expected F16 or F32");
    const std::vector<uint8_t> raw = g.read(name);
    std::vector<float> out(raw.size() / 2);
    for (size_t i = 0; i < out.size(); ++i) { uint16_t h; std::memcpy(&h, &raw[i * 2], 2); out[i] = dara::kernels::half_to_float(h); }
    return out;
}

}  // namespace

GemmaVision::GemmaVision(const dara::io::GgufCheckpoint& g, int max_soft_tokens) : max_soft_(max_soft_tokens) {
    if (meta(g, "general.architecture") != "clip" || meta(g, "clip.vision.projector_type", "?") != "gemma4v")
        throw std::runtime_error("not a Gemma 4 vision mmproj (expected clip / gemma4v)");
    hidden_ = std::stoi(meta(g, "clip.vision.embedding_length"));
    inter_ = std::stoi(meta(g, "clip.vision.feed_forward_length"));
    layers_n_ = std::stoi(meta(g, "clip.vision.block_count"));
    heads_ = std::stoi(meta(g, "clip.vision.attention.head_count"));
    patch_ = std::stoi(meta(g, "clip.vision.patch_size"));
    proj_dim_ = std::stoi(meta(g, "clip.vision.projection_dim"));
    eps_ = std::stof(meta(g, "clip.vision.attention.layer_norm_epsilon", "1e-6"));
    hd_ = hidden_ / heads_;
    if (hd_ != 64) throw std::runtime_error("vision attention kernel supports head_dim 64 only");

    // patch embedding: GGUF conv layout [out][c][y][x] -> linear [out][(y * patch + x) * 3 + c] (the processor's patch order)
    {
        const std::vector<uint16_t> pe = bf16_bits(g, "v.patch_embd.weight");
        const int P = patch_, C = 3, in = P * P * C;
        std::vector<uint16_t> lin(static_cast<size_t>(hidden_) * in);
        const char* order = std::getenv("DARA_VISION_PATCH_ORDER");
        const bool raw = order && std::strcmp(order, "yxc") == 0;
        for (int o = 0; o < hidden_; ++o)
            for (int c = 0; c < C; ++c)
                for (int y = 0; y < P; ++y)
                    for (int x = 0; x < P; ++x) {
                        const size_t src = raw ? static_cast<size_t>(o) * in + (static_cast<size_t>(y) * P + x) * C + c
                                               : ((static_cast<size_t>(o) * C + c) * P + y) * P + x;
                        lin[static_cast<size_t>(o) * in + (static_cast<size_t>(y) * P + x) * C + c] = pe[src];
                    }
        patch_w_ = upload_bf16_raw(lin.data(), static_cast<size_t>(hidden_), static_cast<size_t>(in));
    }
    pos_table_ = f32(g, "v.position_embd.weight");
    pos_size_ = static_cast<int>(pos_table_.size() / (2 * static_cast<size_t>(hidden_)));
    {
        const std::vector<uint16_t> pw = bf16_bits(g, "mm.input_projection.weight");
        proj_w_ = upload_bf16_raw(pw.data(), static_cast<size_t>(proj_dim_), static_cast<size_t>(hidden_));
    }
    auto lin = [&](const std::string& p, size_t rows, size_t cols) {
        Linear l;
        const std::vector<uint16_t> w = bf16_bits(g, p + ".weight");
        l.w = upload_bf16_raw(w.data(), rows, cols);
        auto sc = [&](const char* s, float dflt) { const std::string n = p + "." + s; return g.has(n) ? g.read_f32(n)[0] : dflt; };
        l.in_lo = sc("input_min", -INFINITY); l.in_hi = sc("input_max", INFINITY);
        l.out_lo = sc("output_min", -INFINITY); l.out_hi = sc("output_max", INFINITY);
        return l;
    };
    const size_t H = static_cast<size_t>(hidden_), I = static_cast<size_t>(inter_);
    layers_.reserve(static_cast<size_t>(layers_n_));
    for (int i = 0; i < layers_n_; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        Layer L;
        L.q = lin(b + "attn_q", H, H); L.k = lin(b + "attn_k", H, H); L.v = lin(b + "attn_v", H, H); L.o = lin(b + "attn_out", H, H);
        L.gate = lin(b + "ffn_gate", I, H); L.up = lin(b + "ffn_up", I, H); L.down = lin(b + "ffn_down", H, I);
        L.ln1 = DevBuf<float>(g.read_f32(b + "ln1.weight")); L.attn_post = DevBuf<float>(g.read_f32(b + "attn_post_norm.weight"));
        L.ln2 = DevBuf<float>(g.read_f32(b + "ln2.weight")); L.ffn_post = DevBuf<float>(g.read_f32(b + "ffn_post_norm.weight"));
        L.q_norm = DevBuf<float>(g.read_f32(b + "attn_q_norm.weight")); L.k_norm = DevBuf<float>(g.read_f32(b + "attn_k_norm.weight"));
        layers_.push_back(std::move(L));
    }
    ones_ = DevBuf<float>(std::vector<float>(std::max<size_t>(H, static_cast<size_t>(hd_)), 1.0f));
}

VisionInput GemmaVision::preprocess(const dara::io::RgbImage& img) const {
    // get_aspect_ratio_preserving_size (transformers Gemma4 image processor), max_patches = max_soft_tokens * pool^2
    const int max_patches = max_soft_ * pool_ * pool_;
    const double total_px = static_cast<double>(img.height) * img.width;
    const double target_px = static_cast<double>(max_patches) * patch_ * patch_;
    const double factor = std::sqrt(target_px / total_px);
    const int side = pool_ * patch_;
    int th = static_cast<int>(std::floor(factor * img.height / side)) * side;
    int tw = static_cast<int>(std::floor(factor * img.width / side)) * side;
    if (th == 0 && tw == 0) throw std::runtime_error("image too small to encode");
    const int max_side = (max_patches / (pool_ * pool_)) * side;
    if (th == 0) { th = side; tw = std::min(static_cast<int>(std::floor(static_cast<double>(img.width) / img.height)) * side, max_side); }
    else if (tw == 0) { tw = side; th = std::min(static_cast<int>(std::floor(static_cast<double>(img.height) / img.width)) * side, max_side); }
    const dara::io::RgbImage r = (th == img.height && tw == img.width) ? img : dara::io::resize_bicubic(img, tw, th);

    VisionInput v;
    v.grid_w = tw / patch_;
    v.grid_h = th / patch_;
    v.n = v.grid_w * v.grid_h;
    const int pd = patch_ * patch_ * 3;
    v.patches.resize(static_cast<size_t>(v.n) * pd);
    v.px.resize(static_cast<size_t>(v.n));
    v.py.resize(static_cast<size_t>(v.n));
    for (int gy = 0; gy < v.grid_h; ++gy)
        for (int gx = 0; gx < v.grid_w; ++gx) {
            const int t = gy * v.grid_w + gx;
            v.px[static_cast<size_t>(t)] = gx;
            v.py[static_cast<size_t>(t)] = gy;
            float* dst = &v.patches[static_cast<size_t>(t) * pd];
            for (int y = 0; y < patch_; ++y)
                for (int x = 0; x < patch_; ++x)
                    for (int c = 0; c < 3; ++c) {
                        const uint8_t p = r.px[(static_cast<size_t>(gy * patch_ + y) * r.width + (gx * patch_ + x)) * 3 + c];
                        dst[(y * patch_ + x) * 3 + c] = static_cast<float>(p * (1.0 / 255.0));   // rescale (float64 product, then float32)
                    }
        }
    return v;
}

void GemmaVision::linear(const Linear& l, const float* x, float* tmp, float* y, int n) {
    const size_t in = l.w.cols, out = l.w.rows;
    cuda_check(clamp_copy_cuda(x, tmp, static_cast<size_t>(n) * in, l.in_lo, l.in_hi), "vision clamp in");
    cuda_check(matvec_multi_cuda(l.w, tmp, in, y, out, n), "vision linear");
    cuda_check(clamp_cuda(y, static_cast<size_t>(n) * out, l.out_lo, l.out_hi), "vision clamp out");
}

std::vector<float> GemmaVision::encode(const VisionInput& in) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = in.n;
    const size_t H = static_cast<size_t>(hidden_), I = static_cast<size_t>(inter_), N = static_cast<size_t>(n);
    const int pd = patch_ * patch_ * 3;
    if (n <= 0 || n % (pool_ * pool_) != 0) throw std::invalid_argument("vision input: patch count must be a positive multiple of 9");

    // patch embedding: input_proj(2 * (pixels - 0.5)) + pos_x row + pos_y row
    std::vector<float> xin(N * pd), pos(N * H);
    for (size_t i = 0; i < xin.size(); ++i) xin[i] = 2.0f * (in.patches[i] - 0.5f);
    for (int t = 0; t < n; ++t) {
        const int x = in.px[static_cast<size_t>(t)], y = in.py[static_cast<size_t>(t)];
        if (x < 0 || y < 0 || x >= pos_size_ || y >= pos_size_) throw std::out_of_range("patch position outside the position table");
        const float* tx = &pos_table_[static_cast<size_t>(x) * H];
        const float* ty = &pos_table_[(static_cast<size_t>(pos_size_) + y) * H];
        for (size_t d = 0; d < H; ++d) pos[static_cast<size_t>(t) * H + d] = tx[d] + ty[d];
    }
    DevBuf<float> d_in(xin), d_pos(pos), h(N * H), a(N * H), t(N * std::max(H, I)), q(N * H), k(N * H), v(N * H), att(N * H), g(N * I), u(N * I);
    DevBuf<int> d_px(in.px), d_py(in.py);
    cuda_check(matvec_multi_cuda(patch_w_, d_in.p, static_cast<size_t>(pd), h.p, H, n), "patch embedding");
    cuda_check(add_cuda(h.p, d_pos.p, h.p, N * H), "position embedding");

    for (const Layer& L : layers_) {
        cuda_check(rmsnorm_cuda(h.p, L.ln1.p, n, hidden_, eps_, a.p), "vision ln1");
        linear(L.q, a.p, t.p, q.p, n);
        linear(L.k, a.p, t.p, k.p, n);
        linear(L.v, a.p, t.p, v.p, n);
        cuda_check(rmsnorm_cuda(q.p, L.q_norm.p, n * heads_, hd_, eps_, q.p), "vision q norm");
        cuda_check(rmsnorm_cuda(k.p, L.k_norm.p, n * heads_, hd_, eps_, k.p), "vision k norm");
        cuda_check(rmsnorm_cuda(v.p, ones_.p, n * heads_, hd_, eps_, v.p), "vision v norm");
        cuda_check(rope2d_cuda(q.p, n, heads_, hd_, d_px.p, d_py.p, theta_), "vision rope q");
        cuda_check(rope2d_cuda(k.p, n, heads_, hd_, d_px.p, d_py.p, theta_), "vision rope k");
        cuda_check(vision_attention_cuda(q.p, k.p, v.p, n, heads_, 1.0f, att.p), "vision attention");
        linear(L.o, att.p, t.p, a.p, n);
        cuda_check(norm_residual_cuda(a.p, L.attn_post.p, n, hidden_, eps_, h.p, 1.0f, false, h.p), "vision post-attention norm + residual");
        cuda_check(rmsnorm_cuda(h.p, L.ln2.p, n, hidden_, eps_, a.p), "vision ln2");
        linear(L.gate, a.p, t.p, g.p, n);
        linear(L.up, a.p, t.p, u.p, n);
        cuda_check(geglu_cuda(g.p, u.p, g.p, N * I), "vision geglu");
        linear(L.down, g.p, t.p, a.p, n);
        cuda_check(norm_residual_cuda(a.p, L.ffn_post.p, n, hidden_, eps_, h.p, 1.0f, false, h.p), "vision post-feed-forward norm + residual");
    }
    const std::vector<float> hs = h.get();

    // 3x3 average pooling by patch position, x sqrt(hidden)
    const int pw = in.grid_w / pool_, M = n / (pool_ * pool_);
    std::vector<float> pooled(static_cast<size_t>(M) * H, 0.0f);
    const float w9 = 1.0f / static_cast<float>(pool_ * pool_);
    for (int tt = 0; tt < n; ++tt) {
        const int idx = in.px[static_cast<size_t>(tt)] / pool_ + pw * (in.py[static_cast<size_t>(tt)] / pool_);
        if (idx < 0 || idx >= M) throw std::out_of_range("pooling index out of range");
        float* o = &pooled[static_cast<size_t>(idx) * H];
        const float* s = &hs[static_cast<size_t>(tt) * H];
        for (size_t d = 0; d < H; ++d) o[d] += w9 * s[d];
    }
    const float root = static_cast<float>(std::sqrt(static_cast<double>(hidden_)));
    for (float& x : pooled) x *= root;

    // multimodal embedder: unscaled RMSNorm, projection to the text width
    DevBuf<float> d_pool(pooled), d_out(static_cast<size_t>(M) * proj_dim_);
    cuda_check(rmsnorm_cuda(d_pool.p, ones_.p, M, hidden_, eps_, d_pool.p), "embedder norm");
    cuda_check(matvec_multi_cuda(proj_w_, d_pool.p, H, d_out.p, static_cast<size_t>(proj_dim_), M), "embedder projection");
    std::vector<float> outv = d_out.get();
    last_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return outv;
}

}  // namespace dara::model
