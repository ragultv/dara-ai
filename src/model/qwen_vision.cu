#include "model/qwen_vision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "kernels/half.h"
#include "model/ops_cuda.h"
#include "model/vision_ops_cuda.h"

namespace dara::model {
namespace {

std::string meta(const dara::io::GgufCheckpoint& g, const std::string& key, const std::string& fallback = "") {
    for (const auto& kv : g.header().metadata) if (kv.first == key) return kv.second;
    if (!fallback.empty()) return fallback;
    throw std::runtime_error("mmproj GGUF metadata is missing " + key);
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

// Weights FP16 for the tensor-core GEMM (default: the mmproj F16 values exactly) or BF16 for the FP32-FMA GEMM (DARA_VISION_F16=0)
bool vision_f16() {
    static const bool on = [] {   // default: GPUs with tensor cores (compute capability 8.0+); DARA_VISION_F16=1/0 forces
        if (const char* e = std::getenv("DARA_VISION_F16")) return e[0] == '1';
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 8;
    }();
    return on;
}

// [rows][cols] weights, optionally zero-padded to cols_pad columns / rows_pad rows (padding adds zero terms only)
DeviceMatrix bf16_matrix(const std::vector<float>& w, size_t rows, size_t cols, size_t rows_pad, size_t cols_pad) {
    std::vector<uint16_t> bits(rows_pad * cols_pad, 0);
    const bool h = vision_f16();
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < cols; ++c) bits[r * cols_pad + c] = h ? dara::kernels::float_to_half(w[r * cols + c]) : dara::kernels::float_to_bf16(w[r * cols + c]);
    return h ? upload_f16_raw(bits.data(), rows_pad, cols_pad) : upload_bf16_raw(bits.data(), rows_pad, cols_pad);
}

// torch.linspace(start, end, steps) in float32 (CPU kernel: symmetric fill from both ends)
std::vector<float> linspace_f32(float start, float end, int steps) {
    std::vector<float> v(static_cast<size_t>(steps));
    if (steps == 1) { v[0] = start; return v; }
    const float step = (end - start) / static_cast<float>(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i) v[static_cast<size_t>(i)] = i < halfway ? start + step * static_cast<float>(i) : end - step * static_cast<float>(steps - i - 1);
    return v;
}

}  // namespace

QwenVision::QwenVision(const dara::io::GgufCheckpoint& g, int max_tokens) : max_tokens_(max_tokens) {
    if (meta(g, "general.architecture") != "clip" || meta(g, "clip.projector_type", "?") != "qwen3vl_merger")
        throw std::runtime_error("not a Qwen3.5 vision mmproj (expected clip / qwen3vl_merger)");
    hidden_ = std::stoi(meta(g, "clip.vision.embedding_length"));
    inter_ = std::stoi(meta(g, "clip.vision.feed_forward_length"));
    depth_ = std::stoi(meta(g, "clip.vision.block_count"));
    heads_ = std::stoi(meta(g, "clip.vision.attention.head_count"));
    patch_ = std::stoi(meta(g, "clip.vision.patch_size"));
    merge_ = std::stoi(meta(g, "clip.vision.spatial_merge_size"));
    out_dim_ = std::stoi(meta(g, "clip.vision.projection_dim"));
    eps_ = std::stof(meta(g, "clip.vision.attention.layer_norm_epsilon", "1e-6"));
    hd_ = hidden_ / heads_;
    if (hd_ != 64 && hd_ != 72) throw std::runtime_error("vision attention kernel supports head_dim 64 or 72");
    if (patch_ != 16 || merge_ != 2) throw std::runtime_error("unsupported Qwen vision patch/merge size");
    inter_pad_ = (inter_ + 31) / 32 * 32;   // the BF16 GEMM takes multiples of 32 columns; zero padding leaves the products unchanged
    const size_t H = static_cast<size_t>(hidden_), I = static_cast<size_t>(inter_), IP = static_cast<size_t>(inter_pad_);

    {   // Conv3d [out][c][t][y][x] as a linear over the (c, t, y, x) patch vector; GGUF splits the two frames
        const std::vector<float> w0 = f32(g, "v.patch_embd.weight"), w1 = f32(g, "v.patch_embd.weight.1");
        const size_t P2 = static_cast<size_t>(patch_) * patch_, in = 3 * 2 * P2;
        std::vector<float> lin(H * in);
        for (size_t o = 0; o < H; ++o)
            for (size_t c = 0; c < 3; ++c)
                for (size_t p = 0; p < P2; ++p) {
                    lin[o * in + (c * 2 + 0) * P2 + p] = w0[(o * 3 + c) * P2 + p];
                    lin[o * in + (c * 2 + 1) * P2 + p] = w1[(o * 3 + c) * P2 + p];
                }
        patch_lin_.w = bf16_matrix(lin, H, in, H, in);
        patch_lin_.b = DevBuf<float>(f32(g, "v.patch_embd.bias"));
    }
    pos_table_ = f32(g, "v.position_embd.weight");
    grid_side_ = static_cast<int>(std::lround(std::sqrt(static_cast<double>(pos_table_.size() / H))));
    post_w_ = DevBuf<float>(f32(g, "v.post_ln.weight"));
    post_b_ = DevBuf<float>(f32(g, "v.post_ln.bias"));
    const size_t M4 = 4 * H;
    mm0_.w = bf16_matrix(f32(g, "mm.0.weight"), M4, M4, M4, M4); mm0_.b = DevBuf<float>(f32(g, "mm.0.bias"));
    mm2_.w = bf16_matrix(f32(g, "mm.2.weight"), static_cast<size_t>(out_dim_), M4, static_cast<size_t>(out_dim_), M4); mm2_.b = DevBuf<float>(f32(g, "mm.2.bias"));
    blocks_.reserve(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        Block B;
        B.qkv.w = bf16_matrix(f32(g, b + "attn_qkv.weight"), 3 * H, H, 3 * H, H); B.qkv.b = DevBuf<float>(f32(g, b + "attn_qkv.bias"));
        B.proj.w = bf16_matrix(f32(g, b + "attn_out.weight"), H, H, H, H); B.proj.b = DevBuf<float>(f32(g, b + "attn_out.bias"));
        std::vector<float> b1 = f32(g, b + "ffn_up.bias");
        b1.resize(IP, 0.0f);
        B.fc1.w = bf16_matrix(f32(g, b + "ffn_up.weight"), I, H, IP, H); B.fc1.b = DevBuf<float>(b1);
        B.fc2.w = bf16_matrix(f32(g, b + "ffn_down.weight"), H, I, H, IP); B.fc2.b = DevBuf<float>(f32(g, b + "ffn_down.bias"));
        B.ln1_w = DevBuf<float>(f32(g, b + "ln1.weight")); B.ln1_b = DevBuf<float>(f32(g, b + "ln1.bias"));
        B.ln2_w = DevBuf<float>(f32(g, b + "ln2.weight")); B.ln2_b = DevBuf<float>(f32(g, b + "ln2.bias"));
        blocks_.push_back(std::move(B));
    }
}

QwenVisionInput QwenVision::preprocess(const dara::io::RgbImage& img) const {
    // smart_resize(height, width, factor = patch * merge, min_pixels, max_pixels) -- Python round() is round-half-to-even
    const int factor = patch_ * merge_;
    const double min_pixels = 65536.0, max_pixels = static_cast<double>(max_tokens_) * factor * factor;
    const double h = img.height, w = img.width;
    if (std::max(h, w) / std::min(h, w) > 200) throw std::runtime_error("image aspect ratio must be below 200");
    int hb = static_cast<int>(std::nearbyint(h / factor)) * factor, wb = static_cast<int>(std::nearbyint(w / factor)) * factor;
    if (static_cast<double>(hb) * wb > max_pixels) {
        const double beta = std::sqrt(h * w / max_pixels);
        hb = std::max(factor, static_cast<int>(std::floor(h / beta / factor)) * factor);
        wb = std::max(factor, static_cast<int>(std::floor(w / beta / factor)) * factor);
    } else if (static_cast<double>(hb) * wb < min_pixels) {
        const double beta = std::sqrt(min_pixels / (h * w));
        hb = static_cast<int>(std::ceil(h * beta / factor)) * factor;
        wb = static_cast<int>(std::ceil(w * beta / factor)) * factor;
    }
    const dara::io::RgbImage r = dara::io::resize_bicubic(img, wb, hb);

    QwenVisionInput v;
    v.grid_h = hb / patch_;
    v.grid_w = wb / patch_;
    v.n = v.grid_h * v.grid_w;
    const int P = patch_, P2 = P * P, pd = 3 * temporal_ * P2;
    v.patches.resize(static_cast<size_t>(v.n) * pd);
    v.row.resize(static_cast<size_t>(v.n));
    v.col.resize(static_cast<size_t>(v.n));
    int t = 0;
    for (int bh = 0; bh < v.grid_h / merge_; ++bh)
        for (int bw = 0; bw < v.grid_w / merge_; ++bw)
            for (int mh = 0; mh < merge_; ++mh)
                for (int mw = 0; mw < merge_; ++mw, ++t) {
                    const int gy = bh * merge_ + mh, gx = bw * merge_ + mw;
                    v.row[static_cast<size_t>(t)] = gy;
                    v.col[static_cast<size_t>(t)] = gx;
                    float* dst = &v.patches[static_cast<size_t>(t) * pd];
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < P; ++y)
                            for (int x = 0; x < P; ++x) {
                                const uint8_t p = r.px[(static_cast<size_t>(gy * P + y) * r.width + (gx * P + x)) * 3 + c];
                                const float s = static_cast<float>(p * (1.0 / 255.0));   // rescale (float64 product, then float32)
                                const float nv = (s - 0.5f) / 0.5f;                        // normalize
                                for (int f = 0; f < temporal_; ++f) dst[(c * temporal_ + f) * P2 + y * P + x] = nv;
                            }
                }
    return v;
}

void QwenVision::linear(const Lin& l, const float* x, size_t ldx, float* y, int n) {
    cuda_check(matvec_multi_cuda(l.w, x, ldx, y, l.w.rows, n), "vision linear");
    cuda_check(bias_add_cuda(y, l.b.p, n, static_cast<int>(l.w.rows)), "vision bias");
}

std::vector<float> QwenVision::encode(const QwenVisionInput& in) {
    const auto t0 = std::chrono::steady_clock::now();
    const int n = in.n;
    if (n <= 0 || n % 4 != 0) throw std::invalid_argument("vision input: patch count must be a positive multiple of 4");
    const size_t H = static_cast<size_t>(hidden_), N = static_cast<size_t>(n), IP = static_cast<size_t>(inter_pad_);
    const int pd = 3 * temporal_ * patch_ * patch_;

    // learned position grid, bilinearly resampled to the patch grid (fast_pos_embed_interpolate), in the patches' merge-block order
    std::vector<float> pos(N * H);
    {
        const int S = grid_side_;
        const std::vector<float> hi = linspace_f32(0.0f, static_cast<float>(S - 1), in.grid_h), wi = linspace_f32(0.0f, static_cast<float>(S - 1), in.grid_w);
        for (int t = 0; t < n; ++t) {
            const int gy = in.row[static_cast<size_t>(t)], gx = in.col[static_cast<size_t>(t)];
            const float hy = hi[static_cast<size_t>(gy)], wx = wi[static_cast<size_t>(gx)];
            const int hf = static_cast<int>(hy), wf = static_cast<int>(wx);
            const int hc = std::min(hf + 1, S - 1), wc = std::min(wf + 1, S - 1);
            const float dh = hy - static_cast<float>(hf), dw = wx - static_cast<float>(wf);
            const float w00 = (1 - dh) * (1 - dw), w01 = (1 - dh) * dw, w10 = dh * (1 - dw), w11 = dh * dw;
            const float* r00 = &pos_table_[static_cast<size_t>(hf * S + wf) * H];
            const float* r01 = &pos_table_[static_cast<size_t>(hf * S + wc) * H];
            const float* r10 = &pos_table_[static_cast<size_t>(hc * S + wf) * H];
            const float* r11 = &pos_table_[static_cast<size_t>(hc * S + wc) * H];
            float* o = &pos[static_cast<size_t>(t) * H];
            for (size_t d = 0; d < H; ++d) o[d] = r00[d] * w00 + r01[d] * w01 + r10[d] * w10 + r11[d] * w11;
        }
    }
    DevBuf<float> d_in(in.patches), d_pos(pos), x(N * H), a(N * H), qkv(N * 3 * H), att(N * H), m(N * IP);
    DevBuf<int> d_row(in.row), d_col(in.col);
    linear(patch_lin_, d_in.p, static_cast<size_t>(pd), x.p, n);
    cuda_check(add_cuda(x.p, d_pos.p, x.p, N * H), "position embedding");
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd_));
    for (const Block& B : blocks_) {
        cuda_check(layernorm_cuda(x.p, B.ln1_w.p, B.ln1_b.p, n, hidden_, eps_, a.p), "vision norm1");
        linear(B.qkv, a.p, H, qkv.p, n);
        cuda_check(rope2d_rh_cuda(qkv.p, 3 * H, n, heads_, hd_, d_row.p, d_col.p, 10000.0f), "vision rope q");
        cuda_check(rope2d_rh_cuda(qkv.p + H, 3 * H, n, heads_, hd_, d_row.p, d_col.p, 10000.0f), "vision rope k");
        cuda_check(vision_attention_cuda(qkv.p, qkv.p + H, qkv.p + 2 * H, n, heads_, scale, att.p, nullptr, hd_, 3 * H), "vision attention");
        linear(B.proj, att.p, H, a.p, n);
        cuda_check(add_cuda(x.p, a.p, x.p, N * H), "vision residual 1");
        cuda_check(layernorm_cuda(x.p, B.ln2_w.p, B.ln2_b.p, n, hidden_, eps_, a.p), "vision norm2");
        linear(B.fc1, a.p, H, m.p, n);
        cuda_check(gelu_tanh_cuda(m.p, N * IP), "vision gelu");
        linear(B.fc2, m.p, IP, a.p, n);
        cuda_check(add_cuda(x.p, a.p, x.p, N * H), "vision residual 2");
        // large images: let the GPU drain per layer, so no run of queued work gets long enough for the Windows timeout (TDR) on slow GPUs
        if (n > 1024) cuda_check(cudaStreamSynchronize(nullptr), "vision layer");
    }
    // merger: LayerNorm per patch, 2x2 patches (consecutive in merge-block order) concatenated, fc1 + exact GELU + fc2
    const int M = n / 4;
    cuda_check(layernorm_cuda(x.p, post_w_.p, post_b_.p, n, hidden_, eps_, a.p), "merger norm");
    DevBuf<float> h1(static_cast<size_t>(M) * 4 * H), out(static_cast<size_t>(M) * out_dim_);
    linear(mm0_, a.p, 4 * H, h1.p, M);
    cuda_check(gelu_erf_cuda(h1.p, static_cast<size_t>(M) * 4 * H), "merger gelu");
    linear(mm2_, h1.p, 4 * H, out.p, M);
    std::vector<float> r = out.get();
    last_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

}  // namespace dara::model
