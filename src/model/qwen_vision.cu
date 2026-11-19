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

QwenVision::QwenVision(const dara::io::GgufCheckpoint& g, int max_tokens, int stream_weights) : max_tokens_(max_tokens) {
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
    const size_t M4 = 4 * H, OUT = static_cast<size_t>(out_dim_);

    // Streaming needs every big matrix to be F16 in the file (the mmproj-F16 files are); anything else keeps the matrices resident.
    bool all_f16 = true;
    auto f16_tensor = [&](const std::string& n) { all_f16 = all_f16 && g.has(n) && g.info(n).type == 1; };
    f16_tensor("mm.0.weight"); f16_tensor("mm.2.weight");
    for (int i = 0; i < depth_; ++i) for (const char* t : {"attn_qkv", "attn_out", "ffn_up", "ffn_down"}) f16_tensor("v.blk." + std::to_string(i) + "." + t + ".weight");
    if (stream_weights < 0) { const char* e = std::getenv("DARA_VISION_STREAM"); stream_weights = e ? (e[0] != '0') : 1; }
    streamed_ = stream_weights != 0 && all_f16;

    auto upload_sizes = [&](const Lin& l) { resident_bytes_ += l.w.bytes + l.b.n * sizeof(float); };
    // one weight matrix: a reference into the mapped file (streaming) or an upload (resident)
    auto weight = [&](Lin& l, const std::string& name, size_t rows, size_t cols, size_t rows_pad, size_t cols_pad) {
        if (streamed_) {
            l.s = StreamRef{g.map(name), rows, cols, rows_pad, cols_pad};
        } else {
            l.w = bf16_matrix(f32(g, name), rows, cols, rows_pad, cols_pad);
        }
    };

    {   // Conv3d [out][c][t][y][x] as a linear over the (c, t, y, x) patch vector; GGUF splits the two frames (always resident: re-laid-out, 3 MB)
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
        upload_sizes(patch_lin_);
    }
    {   // the learned position grid is read row by row from the mapped file when an image is encoded
        const auto& pi = g.info("v.position_embd.weight");
        if (pi.type != 0 && pi.type != 1) throw std::runtime_error("mmproj tensor v.position_embd.weight: expected F16 or F32");
        pos_map_ = g.map("v.position_embd.weight");
        pos_type_ = pi.type;
        grid_side_ = static_cast<int>(std::lround(std::sqrt(static_cast<double>(pi.rows() * pi.cols() / H))));
    }
    post_w_ = DevBuf<float>(f32(g, "v.post_ln.weight"));
    post_b_ = DevBuf<float>(f32(g, "v.post_ln.bias"));
    resident_bytes_ += (post_w_.n + post_b_.n) * sizeof(float);
    weight(mm0_, "mm.0.weight", M4, M4, M4, M4); mm0_.b = DevBuf<float>(f32(g, "mm.0.bias"));
    weight(mm2_, "mm.2.weight", OUT, M4, OUT, M4); mm2_.b = DevBuf<float>(f32(g, "mm.2.bias"));
    upload_sizes(mm0_); upload_sizes(mm2_);
    blocks_.reserve(static_cast<size_t>(depth_));
    for (int i = 0; i < depth_; ++i) {
        const std::string b = "v.blk." + std::to_string(i) + ".";
        Block B;
        weight(B.qkv, b + "attn_qkv.weight", 3 * H, H, 3 * H, H); B.qkv.b = DevBuf<float>(f32(g, b + "attn_qkv.bias"));
        weight(B.proj, b + "attn_out.weight", H, H, H, H); B.proj.b = DevBuf<float>(f32(g, b + "attn_out.bias"));
        std::vector<float> b1 = f32(g, b + "ffn_up.bias");
        b1.resize(IP, 0.0f);
        weight(B.fc1, b + "ffn_up.weight", I, H, IP, H); B.fc1.b = DevBuf<float>(b1);
        weight(B.fc2, b + "ffn_down.weight", H, I, H, IP); B.fc2.b = DevBuf<float>(f32(g, b + "ffn_down.bias"));
        B.ln1_w = DevBuf<float>(f32(g, b + "ln1.weight")); B.ln1_b = DevBuf<float>(f32(g, b + "ln1.bias"));
        B.ln2_w = DevBuf<float>(f32(g, b + "ln2.weight")); B.ln2_b = DevBuf<float>(f32(g, b + "ln2.bias"));
        for (const Lin* l : {&B.qkv, &B.proj, &B.fc1, &B.fc2}) upload_sizes(*l);
        resident_bytes_ += (B.ln1_w.n + B.ln1_b.n + B.ln2_w.n + B.ln2_b.n) * sizeof(float);
        blocks_.push_back(std::move(B));
    }
    if (streamed_) build_plan();
}

// The chunks of one image in the order encode() consumes them (the three q / k / v row ranges of the fused qkv matrix are separate steps).
void QwenVision::build_plan() {
    const size_t H = static_cast<size_t>(hidden_);
    auto add = [&](const StreamRef& w, size_t r0, size_t r1) { for (const ChunkReq& c : chunks_for(w, r0, r1)) plan_.push_back(c); };
    for (const Block& B : blocks_) {
        add(B.qkv.s, 0, H); add(B.qkv.s, H, 2 * H); add(B.qkv.s, 2 * H, 3 * H);
        add(B.proj.s, 0, H);
        add(B.fc1.s, 0, B.fc1.s.rows_pad);
        add(B.fc2.s, 0, H);
    }
    add(mm0_.s, 0, mm0_.s.rows_pad);
    add(mm2_.s, 0, mm2_.s.rows_pad);
}

size_t QwenVision::scratch_bytes(int n_patches) const {
    auto a = [](size_t bytes) { return (bytes + 255) & ~size_t(255); };
    const size_t N = static_cast<size_t>(n_patches), H = static_cast<size_t>(hidden_), IP = static_cast<size_t>(inter_pad_), M = N / 4;
    const size_t pd = 3 * static_cast<size_t>(temporal_) * patch_ * patch_;
    size_t b = a(N * pd * 4) + a(N * H * 4) * 7 /* pos, x, a, q, k, v, att */ + a(N * IP * 4) + 2 * a(N * 4) + a(M * 4 * H * 4) + a(M * static_cast<size_t>(out_dim_) * 4);
    if (streamed_) b += WeightStream::device_bytes();
    return b;
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

void QwenVision::linear_rows(WeightStream* ws, const Lin& l, size_t r0, size_t r1, const float* x, size_t ldx, float* y, size_t ldy, int n) {
    if (!ws) {   // resident: a view of the rows
        DeviceMatrix v = l.w;
        v.rows = r1 - r0;
        v.bytes = v.rows * l.w.ld * 2;
        v.data = std::shared_ptr<void>(l.w.data, static_cast<uint8_t*>(l.w.data.get()) + r0 * l.w.ld * 2);
        cuda_check(matvec_multi_cuda(v, x, ldx, y, ldy, n), "vision linear");
    } else {
        for (const ChunkReq& c : chunks_for(l.s, r0, r1)) {
            const DeviceMatrix& m = ws->next(c);
            cuda_check(matvec_multi_cuda(m, x, ldx, y + (c.r0 - r0), ldy, n), "vision linear");
            ws->done();
        }
    }
    if (ldy != r1 - r0) throw std::logic_error("vision linear: the bias add needs a contiguous output");
    cuda_check(bias_add_cuda(y, l.b.p + r0, n, static_cast<int>(r1 - r0)), "vision bias");
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
        std::vector<float> r00(H), r01(H), r10(H), r11(H);
        auto row = [&](int idx, std::vector<float>& dst) {   // F16 or F32 row of the mapped table
            if (pos_type_ == 0) std::memcpy(dst.data(), pos_map_ + static_cast<size_t>(idx) * H * 4, H * 4);
            else for (size_t d = 0; d < H; ++d) { uint16_t h; std::memcpy(&h, pos_map_ + (static_cast<size_t>(idx) * H + d) * 2, 2); dst[d] = dara::kernels::half_to_float(h); }
        };
        for (int t = 0; t < n; ++t) {
            const int gy = in.row[static_cast<size_t>(t)], gx = in.col[static_cast<size_t>(t)];
            const float hy = hi[static_cast<size_t>(gy)], wx = wi[static_cast<size_t>(gx)];
            const int hf = static_cast<int>(hy), wf = static_cast<int>(wx);
            const int hc = std::min(hf + 1, S - 1), wc = std::min(wf + 1, S - 1);
            const float dh = hy - static_cast<float>(hf), dw = wx - static_cast<float>(wf);
            const float w00 = (1 - dh) * (1 - dw), w01 = (1 - dh) * dw, w10 = dh * (1 - dw), w11 = dh * dw;
            row(hf * S + wf, r00); row(hf * S + wc, r01); row(hc * S + wf, r10); row(hc * S + wc, r11);
            float* o = &pos[static_cast<size_t>(t) * H];
            for (size_t d = 0; d < H; ++d) o[d] = r00[d] * w00 + r01[d] * w01 + r10[d] * w10 + r11[d] * w11;
        }
    }
    std::unique_ptr<WeightStream> ws;
    if (streamed_) ws = std::make_unique<WeightStream>(plan_, vision_f16());
    DevBuf<float> d_in(in.patches), d_pos(pos), x(N * H), a(N * H), q(N * H), k(N * H), v(N * H), att(N * H), m(N * IP);
    DevBuf<int> d_row(in.row), d_col(in.col);
    linear_rows(nullptr, patch_lin_, 0, H, d_in.p, static_cast<size_t>(pd), x.p, H, n);
    cuda_check(add_cuda(x.p, d_pos.p, x.p, N * H), "position embedding");
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd_));
    for (const Block& B : blocks_) {
        cuda_check(layernorm_cuda(x.p, B.ln1_w.p, B.ln1_b.p, n, hidden_, eps_, a.p), "vision norm1");
        linear_rows(ws.get(), B.qkv, 0, H, a.p, H, q.p, H, n);          // the fused qkv matrix: q, k and v rows are separate steps (own buffers)
        linear_rows(ws.get(), B.qkv, H, 2 * H, a.p, H, k.p, H, n);
        linear_rows(ws.get(), B.qkv, 2 * H, 3 * H, a.p, H, v.p, H, n);
        cuda_check(rope2d_rh_cuda(q.p, H, n, heads_, hd_, d_row.p, d_col.p, 10000.0f), "vision rope q");
        cuda_check(rope2d_rh_cuda(k.p, H, n, heads_, hd_, d_row.p, d_col.p, 10000.0f), "vision rope k");
        cuda_check(vision_attention_cuda(q.p, k.p, v.p, n, heads_, scale, att.p, nullptr, hd_, H), "vision attention");
        linear_rows(ws.get(), B.proj, 0, H, att.p, H, a.p, H, n);
        cuda_check(add_cuda(x.p, a.p, x.p, N * H), "vision residual 1");
        cuda_check(layernorm_cuda(x.p, B.ln2_w.p, B.ln2_b.p, n, hidden_, eps_, a.p), "vision norm2");
        linear_rows(ws.get(), B.fc1, 0, B.fc1.s.rows_pad ? B.fc1.s.rows_pad : IP, a.p, H, m.p, IP, n);
        cuda_check(gelu_tanh_cuda(m.p, N * IP), "vision gelu");
        linear_rows(ws.get(), B.fc2, 0, H, m.p, IP, a.p, H, n);
        cuda_check(add_cuda(x.p, a.p, x.p, N * H), "vision residual 2");
        // large images: let the GPU drain per layer, so no run of queued work gets long enough for the Windows timeout (TDR) on slow GPUs
        if (n > 1024) cuda_check(cudaStreamSynchronize(nullptr), "vision layer");
    }
    // merger: LayerNorm per patch, 2x2 patches (consecutive in merge-block order) concatenated, fc1 + exact GELU + fc2
    const int M = n / 4;
    cuda_check(layernorm_cuda(x.p, post_w_.p, post_b_.p, n, hidden_, eps_, a.p), "merger norm");
    DevBuf<float> h1(static_cast<size_t>(M) * 4 * H), out(static_cast<size_t>(M) * out_dim_);
    linear_rows(ws.get(), mm0_, 0, 4 * H, a.p, 4 * H, h1.p, 4 * H, M);
    cuda_check(gelu_erf_cuda(h1.p, static_cast<size_t>(M) * 4 * H), "merger gelu");
    linear_rows(ws.get(), mm2_, 0, static_cast<size_t>(out_dim_), h1.p, 4 * H, out.p, static_cast<size_t>(out_dim_), M);
    std::vector<float> r = out.get();
    last_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}
}  // namespace dara::model
