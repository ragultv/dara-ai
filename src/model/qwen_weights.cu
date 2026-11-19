#include "model/qwen_weights.h"

#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/quant.h"
#include "model/ops.h"
#include "model/qwen_model.h"

namespace dara::model {
namespace {

void fill_dn_dims(DeltaNetDims& d, const QwenConfig& cfg) {
    d.hidden = cfg.hidden; d.k_heads = cfg.k_heads; d.v_heads = cfg.v_heads; d.k_dim = cfg.k_dim; d.v_dim = cfg.v_dim;
    d.conv_kernel = cfg.conv_kernel; d.eps = cfg.eps;
}
void fill_at_dims(AttentionDims& d, const QwenConfig& cfg) {
    d.hidden = cfg.hidden; d.heads = cfg.heads; d.kv_heads = cfg.kv_heads; d.head_dim = cfg.head_dim;
    d.rot_dim = cfg.rot_dim(); d.eps = cfg.eps; d.rope_theta = cfg.rope_theta;
}

dara::kernels::QuantType gguf_qtype(uint32_t type, const std::string& name) {
    switch (type) {
        case 8: return dara::kernels::QuantType::Q8_0;
        case 12: return dara::kernels::QuantType::Q4_K;
        case 13: return dara::kernels::QuantType::Q5_K;
        case 14: return dara::kernels::QuantType::Q6_K;
    }
    throw std::runtime_error("unsupported GGUF matrix type " + std::to_string(type) + " for " + name);
}

// One projection matrix of a layer: which GGUF tensor, which member of GpuLayerWeights it feeds, and its expected shape.
struct MatDesc {
    std::string tensor;
    DeviceMatrix GpuLayerWeights::* member;
    size_t rows, cols;
};

std::vector<MatDesc> layer_matrices(const QwenConfig& cfg, LayerKind kind, const std::string& b) {
    const size_t H = static_cast<size_t>(cfg.hidden), I = static_cast<size_t>(cfg.inter);
    std::vector<MatDesc> m;
    m.push_back({b + "ffn_gate.weight", &GpuLayerWeights::gate, I, H});
    m.push_back({b + "ffn_up.weight", &GpuLayerWeights::up, I, H});
    m.push_back({b + "ffn_down.weight", &GpuLayerWeights::down, H, I});
    if (kind == LayerKind::DeltaNet) {
        DeltaNetDims d;
        fill_dn_dims(d, cfg);
        m.push_back({b + "attn_qkv.weight", &GpuLayerWeights::in_qkv, static_cast<size_t>(d.conv_dim()), H});
        m.push_back({b + "attn_gate.weight", &GpuLayerWeights::in_z, static_cast<size_t>(d.value_dim()), H});
        m.push_back({b + "ssm_beta.weight", &GpuLayerWeights::in_b, static_cast<size_t>(d.v_heads), H});
        m.push_back({b + "ssm_alpha.weight", &GpuLayerWeights::in_a, static_cast<size_t>(d.v_heads), H});
        m.push_back({b + "ssm_out.weight", &GpuLayerWeights::dn_out, H, static_cast<size_t>(d.value_dim())});
    } else {
        AttentionDims d;
        fill_at_dims(d, cfg);
        m.push_back({b + "attn_q.weight", &GpuLayerWeights::q_proj, static_cast<size_t>(d.q_rows()), H});
        m.push_back({b + "attn_k.weight", &GpuLayerWeights::k_proj, static_cast<size_t>(d.kv_width()), H});
        m.push_back({b + "attn_v.weight", &GpuLayerWeights::v_proj, static_cast<size_t>(d.kv_width()), H});
        m.push_back({b + "attn_output.weight", &GpuLayerWeights::o_proj, H, static_cast<size_t>(d.out_width())});
    }
    return m;
}

}  // namespace

GgufWeights::GgufWeights(const dara::io::GgufCheckpoint& g, const QwenConfig& cfg) : g_(g), cfg_(cfg) {}

size_t GgufWeights::layer_matrix_bytes(int layer) const {
    size_t off = 0;
    for (const MatDesc& d : layer_matrices(cfg_, cfg_.kinds.at(layer), "blk." + std::to_string(layer) + "."))
        off = align_up(off + g_.info(d.tensor).bytes);
    return off;
}

size_t GgufWeights::head_bytes() const { return g_.info(cfg_.tied_embeddings ? "token_embd.weight" : "output.weight").bytes; }
size_t GgufWeights::embedding_bytes() const { return g_.info("token_embd.weight").bytes; }

PackedLayer GgufWeights::load_layer_packed(int layer) const {
    const std::string b = "blk." + std::to_string(layer) + ".";
    PackedLayer p;
    GpuLayerWeights& w = p.weights;
    w.kind = cfg_.kinds.at(layer);
    w.hidden = cfg_.hidden; w.inter = cfg_.inter; w.eps = cfg_.eps;
    w.input_norm = DevBuf<float>(g_.read_f32(b + "attn_norm.weight"));            // +1 already folded in
    w.post_norm = DevBuf<float>(g_.read_f32(b + "post_attention_norm.weight"));
    if (w.kind == LayerKind::DeltaNet) {
        fill_dn_dims(w.dn, cfg_);
        w.order = VHeadOrder::Gguf;  // V heads de-interleaved (audited on every layer)
        w.conv_w = DevBuf<float>(g_.read_f32(b + "ssm_conv1d.weight"));   // [conv_dim][taps]
        w.ssm_a = DevBuf<float>(g_.read_f32(b + "ssm_a"));                // already -exp(A_log)
        w.dt_bias = DevBuf<float>(g_.read_f32(b + "ssm_dt.bias"));
        w.norm_w = DevBuf<float>(g_.read_f32(b + "ssm_norm.weight"));
    } else {
        fill_at_dims(w.at, cfg_);
        w.q_norm = DevBuf<float>(g_.read_f32(b + "attn_q_norm.weight"));
        w.k_norm = DevBuf<float>(g_.read_f32(b + "attn_k_norm.weight"));
        w.inv_freq = DevBuf<float>(rope_inv_freq(w.at.rot_dim, w.at.rope_theta));
    }
    // Pack every matrix back to back, reading each tensor straight into the image (no temporary copies).
    const std::vector<MatDesc> descs = layer_matrices(cfg_, w.kind, b);
    size_t off = 0;
    for (const MatDesc& d : descs) {
        const auto& in = g_.info(d.tensor);
        if (in.cols() != d.cols || in.rows() != d.rows) throw std::runtime_error("unexpected shape for " + d.tensor);
        p.slots.push_back({d.member, gguf_qtype(in.type, d.tensor), d.rows, d.cols, off, static_cast<size_t>(in.bytes)});
        off = align_up(off + in.bytes);
    }
    p.matrix_bytes = off;
    p.host = HostBuffer(off);
    for (size_t i = 0; i < descs.size(); ++i) g_.read_into(descs[i].tensor, p.host.data() + p.slots[i].offset);
    return p;
}

std::vector<float> GgufWeights::embed_row(int32_t token) const {
    // Rows come straight from the memory-mapped GGUF (file-backed pages: no private copy of the 497 MB table; a row is read on first touch).
    // DARA_EMBD_RAM=1 restores the old full copy in host RAM.
    const auto& in = g_.info("token_embd.weight");
    static const bool ram = [] { const char* e = std::getenv("DARA_EMBD_RAM"); return e && e[0] == '1'; }();
    if (!embd_ptr_) {
        if (ram) { embd_ = g_.read("token_embd.weight"); embd_ptr_ = embd_.data(); }
        else embd_ptr_ = g_.map("token_embd.weight");
    }
    const auto& f = dara::kernels::quant_format(gguf_qtype(in.type, "token_embd.weight"));
    const size_t blocks = in.cols() / f.block_weights;
    std::vector<float> out(in.cols());
    f.dequant_f32(embd_ptr_ + static_cast<size_t>(token) * blocks * f.block_bytes, blocks, out.data());
    return out;
}

std::vector<float> GgufWeights::final_norm_scale() const {
    if (final_norm_.empty()) final_norm_ = g_.read_f32("output_norm.weight");
    return final_norm_;
}

DeviceMatrix GgufWeights::load_output_head() const {
    // Untied models carry their own output.weight (possibly a different quantization than token_embd, e.g. Q6_K vs Q4_K);
    // tied models use the embedding matrix itself.
    const std::string name = cfg_.tied_embeddings ? "token_embd.weight" : "output.weight";
    const auto& in = g_.info(name);
    if (in.cols() != static_cast<uint64_t>(cfg_.hidden) || in.rows() != static_cast<uint64_t>(cfg_.vocab)) throw std::runtime_error("unexpected shape for " + name);
    const auto bytes = g_.read(name);
    return upload_quant_raw(gguf_qtype(in.type, name), bytes.data(), cfg_.vocab, cfg_.hidden);
}


std::vector<uint8_t> GgufWeights::read_output_head(dara::kernels::QuantType* qt) const {
    const std::string name = cfg_.tied_embeddings ? "token_embd.weight" : "output.weight";
    const auto& in = g_.info(name);
    if (in.cols() != static_cast<uint64_t>(cfg_.hidden) || in.rows() != static_cast<uint64_t>(cfg_.vocab)) throw std::runtime_error("unexpected shape for " + name);
    if (qt) *qt = gguf_qtype(in.type, name);
    return g_.read(name);
}

const uint8_t* GgufWeights::map_output_head(dara::kernels::QuantType* qt, size_t* bytes) const {
    const std::string name = cfg_.tied_embeddings ? "token_embd.weight" : "output.weight";
    const auto& in = g_.info(name);
    if (in.cols() != static_cast<uint64_t>(cfg_.hidden) || in.rows() != static_cast<uint64_t>(cfg_.vocab)) throw std::runtime_error("unexpected shape for " + name);
    if (qt) *qt = gguf_qtype(in.type, name);
    if (bytes) *bytes = static_cast<size_t>(in.bytes);
    return g_.map(name);
}

}  // namespace dara::model
