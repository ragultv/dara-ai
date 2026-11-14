#include "model/gemma_model.h"

#include <algorithm>
#include <optional>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "core/placement.h"
#include "core/system.h"
#include "kernels/quant.h"
#include "model/gemma_ops_cuda.h"
#include "model/ops.h"
#include "model/ops_cuda.h"
#include "model/attn_prefill.h"
#include "model/attn_splitkv.h"
#include "model/tq_kv.h"

#define PM(label) prof_mark(label)

namespace dara::model {
namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
constexpr size_t kMiB = size_t(1) << 20;

std::string meta(const dara::io::GgufCheckpoint& g, const std::string& key) {
    for (const auto& kv : g.header().metadata)
        if (kv.first == key) return kv.second;
    throw std::runtime_error("GGUF metadata is missing " + key);
}
bool has_meta(const dara::io::GgufCheckpoint& g, const std::string& key) {
    for (const auto& kv : g.header().metadata)
        if (kv.first == key) return true;
    return false;
}
long long meta_int(const dara::io::GgufCheckpoint& g, const std::string& key) { return std::stoll(meta(g, key)); }
double meta_num(const dara::io::GgufCheckpoint& g, const std::string& key) { return std::stod(meta(g, key)); }
const std::vector<int64_t>& meta_ints(const dara::io::GgufCheckpoint& g, const std::string& key) {
    const auto it = g.header().int_arrays.find(key);
    if (it == g.header().int_arrays.end()) throw std::runtime_error("GGUF metadata is missing array " + key);
    return it->second;
}

dara::kernels::QuantType qtype_of(uint32_t type, const std::string& name) {
    switch (type) {
        case 8: return dara::kernels::QuantType::Q8_0;
        case 12: return dara::kernels::QuantType::Q4_K;
        case 13: return dara::kernels::QuantType::Q5_K;
        case 14: return dara::kernels::QuantType::Q6_K;
    }
    throw std::runtime_error("unsupported GGUF matrix type " + std::to_string(type) + " for " + name);
}

// F32 or BF16 tensor -> device BF16 matrix, bit-exact. The checkpoint is BF16, so an F32 copy of a weight is exactly representable;
// anything that is not is rejected rather than silently rounded.
DeviceMatrix upload_exact_bf16(const dara::io::GgufCheckpoint& g, const std::string& name, size_t rows, size_t cols) {
    const auto& in = g.info(name);
    if (in.cols() != cols || in.rows() != rows) throw std::runtime_error("unexpected shape for " + name);
    const std::vector<uint8_t> raw = g.read(name);
    std::vector<uint16_t> bits(rows * cols);
    if (in.type == 30) {  // BF16
        std::memcpy(bits.data(), raw.data(), bits.size() * 2);
    } else if (in.type == 0) {  // F32
        for (size_t i = 0; i < bits.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, raw.data() + i * 4, 4);
            if (u & 0xFFFFu) throw std::runtime_error(name + " holds F32 values that are not BF16-exact");
            bits[i] = static_cast<uint16_t>(u >> 16);
        }
    } else {
        throw std::runtime_error(name + ": expected an F32 or BF16 matrix");
    }
    return upload_bf16_raw(bits.data(), rows, cols);
}

DevBuf<float> f32_tensor(const dara::io::GgufCheckpoint& g, const std::string& name, size_t expect) {
    const std::vector<float> v = g.read_f32(name);
    if (v.size() != expect) throw std::runtime_error("unexpected size of " + name);
    return DevBuf<float>(v);
}

struct MatDesc { std::string tensor; int index; size_t rows, cols; };
constexpr int kQ = 0, kK = 1, kV = 2, kO = 3, kGate = 4, kUp = 5, kDown = 6, kNumMats = 7;

std::vector<MatDesc> layer_matrices(const GemmaConfig& c, int l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    const size_t H = static_cast<size_t>(c.hidden), hd = static_cast<size_t>(c.head_dim(l)), I = static_cast<size_t>(c.inter[l]);
    std::vector<MatDesc> m;
    m.push_back({b + "attn_q.weight", kQ, static_cast<size_t>(c.heads) * hd, H});
    if (c.owns_kv(l)) {  // KV-sharing layers carry unused k/v tensors in the file; they are not loaded
        m.push_back({b + "attn_k.weight", kK, static_cast<size_t>(c.kv_heads) * hd, H});
        m.push_back({b + "attn_v.weight", kV, static_cast<size_t>(c.kv_heads) * hd, H});
    }
    m.push_back({b + "attn_output.weight", kO, H, static_cast<size_t>(c.heads) * hd});
    m.push_back({b + "ffn_gate.weight", kGate, I, H});
    m.push_back({b + "ffn_up.weight", kUp, I, H});
    m.push_back({b + "ffn_down.weight", kDown, H, I});
    return m;
}

}  // namespace

// ---- configuration --------------------------------------------------------------------------------------------------------------
GemmaConfig GemmaConfig::from_gguf(const dara::io::GgufCheckpoint& g) {
    const std::string arch = meta(g, "general.architecture");
    if (arch != "gemma4") throw std::runtime_error("unsupported GGUF architecture '" + arch + "' (this backend runs 'gemma4')");
    const std::string a = arch + ".";
    GemmaConfig c;
    c.layers = static_cast<int>(meta_int(g, a + "block_count"));
    c.hidden = static_cast<int>(meta_int(g, a + "embedding_length"));
    c.heads = static_cast<int>(meta_int(g, a + "attention.head_count"));
    c.kv_heads = static_cast<int>(meta_int(g, a + "attention.head_count_kv"));
    c.head_dim_full = static_cast<int>(meta_int(g, a + "attention.key_length"));
    c.head_dim_swa = static_cast<int>(meta_int(g, a + "attention.key_length_swa"));
    if (meta_int(g, a + "attention.value_length") != c.head_dim_full || meta_int(g, a + "attention.value_length_swa") != c.head_dim_swa)
        throw std::runtime_error("key_length != value_length is not supported");
    c.window = static_cast<int>(meta_int(g, a + "attention.sliding_window"));
    c.per_layer_dim = static_cast<int>(meta_int(g, a + "embedding_length_per_layer_input"));
    c.shared_kv_layers = static_cast<int>(meta_int(g, a + "attention.shared_kv_layers"));
    c.eps = static_cast<float>(meta_num(g, a + "attention.layer_norm_rms_epsilon"));
    c.softcap = has_meta(g, a + "final_logit_softcapping") ? static_cast<float>(meta_num(g, a + "final_logit_softcapping")) : 0.0f;
    c.theta_full = meta_num(g, a + "rope.freq_base");
    c.theta_swa = meta_num(g, a + "rope.freq_base_swa");
    const auto& pat = meta_ints(g, a + "attention.sliding_window_pattern");
    const auto& ffn = meta_ints(g, a + "feed_forward_length");
    if (static_cast<int>(pat.size()) != c.layers || static_cast<int>(ffn.size()) != c.layers) throw std::runtime_error("per-layer metadata does not match block_count");
    for (int l = 0; l < c.layers; ++l) { c.sliding.push_back(pat[l] ? 1 : 0); c.inter.push_back(static_cast<int>(ffn[l])); }

    const int first_shared = c.layers - c.shared_kv_layers;
    if (c.shared_kv_layers < 0 || first_shared <= 0) throw std::runtime_error("bad shared_kv_layers");
    c.kv_src.resize(c.layers);
    for (int l = 0; l < c.layers; ++l) {
        if (l < first_shared) { c.kv_src[l] = l; continue; }
        int src = -1;
        for (int j = first_shared - 1; j >= 0; --j) if (c.sliding[j] == c.sliding[l]) { src = j; break; }
        if (src < 0) throw std::runtime_error("a KV-sharing layer has no earlier layer of its kind to read from");
        c.kv_src[l] = src;
    }
    const auto& emb = g.info("token_embd.weight");
    if (emb.cols() != static_cast<uint64_t>(c.hidden)) throw std::runtime_error("token_embd width does not match embedding_length");
    c.vocab = static_cast<int>(emb.rows());
    if (g.has("output.weight")) throw std::runtime_error("an untied output head is not supported for gemma4 yet");
    if (c.head_dim_swa % 32 || c.head_dim_full % 32 || c.heads % c.kv_heads) throw std::runtime_error("unsupported head geometry");

    // Global layers use "proportional" RoPE: only the leading frequency pairs rotate, the rest are position independent. The file
    // states this through rope_freqs.weight (a divisor per frequency pair: 1 = rotate, huge = no rotation).
    const auto& rf = g.info("rope_freqs.weight");
    if (rf.cols() != static_cast<uint64_t>(c.head_dim_full / 2)) throw std::runtime_error("rope_freqs.weight has an unexpected length");
    const std::vector<float> freq_factors = g.read_f32("rope_freqs.weight");
    int rotated = 0;
    while (rotated < static_cast<int>(freq_factors.size()) && freq_factors[rotated] == 1.0f) ++rotated;
    for (size_t i = rotated; i < freq_factors.size(); ++i)
        if (freq_factors[i] < 1e20f) throw std::runtime_error("rope_freqs.weight is not of the form [1,...,1,huge,...,huge]");
    c.rotated_pairs_full = rotated;
    return c;
}

// ---- construction ---------------------------------------------------------------------------------------------------------------
// DARA_ATTN_REF=1 selects the original single-block attention kernels (bitwise regression against the RC1/RC2 baselines); the default is split-KV.
static bool attention_reference_mode() {
    static const bool r = [] { const char* e = std::getenv("DARA_ATTN_REF"); return e && e[0] == '1'; }();
    return r;
}

GemmaModel::GemmaModel(const dara::io::GgufCheckpoint& g, size_t kv_capacity, const ResidencyOptions& opt)
    : g_(g), cfg_(GemmaConfig::from_gguf(g)), opt_(opt), capacity_(kv_capacity) {
  try {
    const auto t_start = Clock::now();
    cudaFree(nullptr);
    report_.gpu = require_supported_gpu();
    size_t free_b = 0, total_b = 0;
    cuda_check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    if (kv_capacity == 0 || kv_capacity > static_cast<size_t>(kGemmaMaxContext)) throw std::invalid_argument("kv_capacity must be in [1, kGemmaMaxContext]");
    cuda_check(int8_init(), "int8_init");
    cuda_check(tq_init(), "tq_init");  // codebook (also read by the split-KV TurboQuant kernels)
    cuda_check(attn_splitkv_init(), "attn_splitkv_init");
    if (opt.kv_format == KvFormat::TQ4) {
        if (kv_capacity > static_cast<size_t>(kTqMaxPositions)) throw std::invalid_argument("TurboQuant KV: kv_capacity above kTqMaxPositions");
        cuda_check(tq_init(), "tq_init");  // codebook to the device, outside any graph capture
    }

    const int L = cfg_.layers;
    // ---- sizes --------------------------------------------------------------------------------------------------------------
    std::vector<size_t> mat_bytes(L, 0);
    size_t max_layer = 0, kv_bytes = 0, small_bytes = 0;
    for (int l = 0; l < L; ++l) {
        size_t off = 0;
        for (const MatDesc& d : layer_matrices(cfg_, l)) off = align_up(off + g.info(d.tensor).bytes);
        mat_bytes[l] = off;
        max_layer = std::max(max_layer, off);
        if (cfg_.owns_kv(l)) kv_bytes += 2 * kv_capacity * static_cast<size_t>(cfg_.kv_heads) * kv_bytes_per_head_vector(opt.kv_format, cfg_.head_dim(l));
        // norms, rope table, two small BF16 matrices
        small_bytes += (5 * static_cast<size_t>(cfg_.hidden) + 2 * cfg_.head_dim(l) + cfg_.head_dim(l) / 2) * sizeof(float) +
                       2 * static_cast<size_t>(cfg_.hidden) * cfg_.per_layer_dim * 2;
    }
    const size_t head_bytes = g.info("token_embd.weight").bytes;
    const size_t pli_table_bytes = g.info("per_layer_token_embd.weight").bytes;
    const size_t C = static_cast<size_t>(std::max(1, opt.max_chunk_tokens));
    int max_inter = 0, max_qw = 0;
    for (int l = 0; l < L; ++l) { max_inter = std::max(max_inter, cfg_.inter[l]); max_qw = std::max(max_qw, cfg_.heads * cfg_.head_dim(l)); }
    const size_t pli_floats_per_token = static_cast<size_t>(L) * cfg_.per_layer_dim;
    const size_t workspace = C * (10 * static_cast<size_t>(cfg_.hidden) + 3 * static_cast<size_t>(max_inter) + 3 * static_cast<size_t>(max_qw) +
                                  3 * pli_floats_per_token + 4 * static_cast<size_t>(cfg_.per_layer_dim)) * sizeof(float) + 32 * kMiB +
                                 static_cast<size_t>(kGemmaMaxContext) * sizeof(float) * 0;
    const size_t safety = opt.safety_bytes ? opt.safety_bytes : std::max<size_t>(256 * kMiB, total_b / 16);

    // ---- plan ---------------------------------------------------------------------------------------------------------------
    dara::core::ModelManifest man;
    man.family = "gemma4";
    man.recurrent_state_bytes = kv_bytes;  // the exact K/V bytes of the layers that own K/V (the planner needs no per-layer model of them)
    man.embed_output_bytes = head_bytes + small_bytes;
    man.host_resident_bytes = 0;  // the per-layer embedding table is accounted for below, against the RAM budget
    for (int l = 0; l < L; ++l) {
        dara::core::LayerSpec ls;
        ls.kind = dara::core::LayerKind::DeltaNet;  // "no per-token KV in the planner's formula": KV is already counted exactly above
        ls.stored_bytes = ls.weight_bytes = mat_bytes[l];
        man.layers.push_back(ls);
    }
    const auto sys = dara::core::query_system_memory();
    dara::core::PlacementRequest req;
    req.model = man;
    req.kv_mode = dara::core::KvMode::F32;
    req.context_tokens = kv_capacity;
    req.vision_resident = false;
    req.budget.vram_total = total_b;
    req.budget.vram_free = free_b;
    req.budget.runtime_reserve = 0;
    req.budget.safety_reserve = safety;
    req.budget.workspace_bytes = workspace;
    const size_t ram_margin = 1536 * kMiB;
    // The per-layer embedding table lives in host RAM when it fits next to the margin; otherwise its rows are read from the file.
    const bool table_in_ram = sys.available_bytes > ram_margin + pli_table_bytes + 2 * head_bytes;
    const size_t ram_for_layers_base = sys.available_bytes > ram_margin ? sys.available_bytes - ram_margin : 0;
    const size_t reserved_host = (table_in_ram ? pli_table_bytes : 0) + head_bytes;  // token_embd copy for row gathers
    req.budget.ram_free = ram_for_layers_base > reserved_host ? ram_for_layers_base - reserved_host : 0;
    dara::core::PlacementPlan plan = dara::core::plan_placement(req);
    size_t staging = 0;
    if (plan.host_layers > 0) {
        staging = 2 * align_up(max_layer);
        req.budget.workspace_bytes = workspace + staging;
        plan = dara::core::plan_placement(req);
    }
    int resident = static_cast<int>(plan.gpu_layers);
    if (opt.force_resident_layers >= 0) resident = std::min(L, opt.force_resident_layers);
    report_.plan = plan.describe();
    if (opt.force_resident_layers >= 0)
        report_.plan = "FORCED " + std::to_string(resident) + " VRAM / " + std::to_string(L - resident) + " host layers (planner would have chosen: " + report_.plan + ")";
    if (opt.force_resident_layers < 0 && (plan.kind == dara::core::Placement::DoesNotFit || plan.kind == dara::core::Placement::StorageBacked)) {
        std::string why = std::string("cannot place the model: ") + plan.describe();
        for (const auto& w : plan.warnings) why += "; " + w;
        why += " | free VRAM " + std::to_string(free_b / kMiB) + " MiB of " + std::to_string(total_b / kMiB) + " MiB, head " + std::to_string(head_bytes / kMiB) +
               " MiB, K/V " + std::to_string(kv_bytes / kMiB) + " MiB, workspace " + std::to_string(workspace / kMiB) + " MiB, safety " +
               std::to_string(safety / kMiB) + " MiB, available RAM " + std::to_string(sys.available_bytes / kMiB) + " MiB";
        throw std::runtime_error(why);
    }
    resident_ = resident;

    // ---- load once ----------------------------------------------------------------------------------------------------------
    kv_.resize(L);
    for (int l = 0; l < L; ++l)
        if (cfg_.owns_kv(l)) {
            const size_t w = static_cast<size_t>(cfg_.kv_heads) * cfg_.head_dim(l);
            if (opt.kv_format == KvFormat::K16V8) {
                kv_[l].Kq = DevBuf<uint8_t>(kv_capacity * w * 2); kv_[l].Vq = DevBuf<uint8_t>(kv_capacity * w);
                kv_[l].Vn = DevBuf<float>(kv_capacity * w / 32);
            } else if (opt.kv_format == KvFormat::Q8) {
                kv_[l].Kq = DevBuf<uint8_t>(kv_capacity * w); kv_[l].Vq = DevBuf<uint8_t>(kv_capacity * w);
                kv_[l].Kn = DevBuf<float>(kv_capacity * w / 32); kv_[l].Vn = DevBuf<float>(kv_capacity * w / 32);
            } else if (opt.kv_format == KvFormat::TQ4) {
                const size_t codes = kv_capacity * static_cast<size_t>(cfg_.kv_heads) * (cfg_.head_dim(l) / 2), norms = kv_capacity * static_cast<size_t>(cfg_.kv_heads);
                kv_[l].Kq = DevBuf<uint8_t>(codes); kv_[l].Vq = DevBuf<uint8_t>(codes);
                kv_[l].Kn = DevBuf<float>(norms); kv_[l].Vn = DevBuf<float>(norms);
            } else {
                kv_[l].K = DevBuf<float>(kv_capacity * w);
                kv_[l].V = DevBuf<float>(kv_capacity * w);
            }
        }
    head_ = upload_quant_raw(qtype_of(g.info("token_embd.weight").type, "token_embd.weight"), g.read("token_embd.weight").data(), static_cast<size_t>(cfg_.vocab), cfg_.hidden);
    const size_t PL = pli_floats_per_token;
    pli_proj_ = upload_exact_bf16(g, "per_layer_model_proj.weight", PL, static_cast<size_t>(cfg_.hidden));
    pli_proj_norm_ = f32_tensor(g, "per_layer_proj_norm.weight", static_cast<size_t>(cfg_.per_layer_dim));
    out_norm_ = f32_tensor(g, "output_norm.weight", static_cast<size_t>(cfg_.hidden));
    ones_ = DevBuf<float>(std::vector<float>(static_cast<size_t>(std::max(cfg_.head_dim_full, cfg_.head_dim_swa)), 1.0f));

    small_.resize(L);
    layers_.reserve(L);
    report_.hosted_pinned = true;
    for (int l = 0; l < L; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const size_t H = static_cast<size_t>(cfg_.hidden), hd = static_cast<size_t>(cfg_.head_dim(l));
        Small& s = small_[l];
        s.input_norm = f32_tensor(g, b + "attn_norm.weight", H);
        s.post_attn_norm = f32_tensor(g, b + "post_attention_norm.weight", H);
        s.pre_ff_norm = f32_tensor(g, b + "ffn_norm.weight", H);
        s.post_ff_norm = f32_tensor(g, b + "post_ffw_norm.weight", H);
        s.post_pli_norm = f32_tensor(g, b + "post_norm.weight", H);
        s.q_norm = f32_tensor(g, b + "attn_q_norm.weight", hd);
        if (cfg_.owns_kv(l)) s.k_norm = f32_tensor(g, b + "attn_k_norm.weight", hd);
        // inverse frequencies exactly as the reference builds them (float32)
        std::vector<float> inv;
        if (cfg_.sliding[l]) {
            inv = rope_inv_freq(cfg_.head_dim_swa, cfg_.theta_swa);
        } else {
            inv = rope_inv_freq(cfg_.head_dim_full, cfg_.theta_full);
            for (size_t i = static_cast<size_t>(cfg_.rotated_pairs_full); i < inv.size(); ++i) inv[i] = 0.0f;
        }
        s.inv_freq = DevBuf<float>(inv);
        s.inp_gate = upload_exact_bf16(g, b + "inp_gate.weight", static_cast<size_t>(cfg_.per_layer_dim), H);
        s.proj = upload_exact_bf16(g, b + "proj.weight", H, static_cast<size_t>(cfg_.per_layer_dim));
        const std::vector<float> sc = g.read_f32(b + "layer_output_scale.weight");
        if (sc.size() != 1) throw std::runtime_error(b + "layer_output_scale.weight must hold one value");
        s.layer_scalar = sc[0];

        PackedLayer p;
        p.mats.resize(kNumMats);
        size_t off = 0;
        const std::vector<MatDesc> descs = layer_matrices(cfg_, l);
        for (const MatDesc& d : descs) {
            const auto& in = g.info(d.tensor);
            if (in.cols() != d.cols || in.rows() != d.rows) throw std::runtime_error("unexpected shape for " + d.tensor);
            MatrixSlot slot{nullptr, qtype_of(in.type, d.tensor), d.rows, d.cols, off, static_cast<size_t>(in.bytes), d.index};
            p.slots.push_back(slot);
            off = align_up(off + in.bytes);
        }
        p.matrix_bytes = off;
        p.host = HostBuffer(off);
        for (size_t i = 0; i < descs.size(); ++i) g.read_into(descs[i].tensor, p.host.data() + p.slots[i].offset);
        layers_.push_back(std::move(p));
        PackedLayer& pl = layers_.back();
        if (l < resident_) {
            make_resident(pl);
            report_.resident_weight_bytes += pl.matrix_bytes;
        } else {
            report_.hosted_weight_bytes += pl.matrix_bytes;
            report_.hosted_pinned = report_.hosted_pinned && pl.host.pinned();
        }
    }

    // host-side embedding tables, read once now (cold load), not on the first token
    embd_type_ = g.info("token_embd.weight").type;
    embd_ = g.read("token_embd.weight");
    pli_type_ = g.info("per_layer_token_embd.weight").type;
    if (table_in_ram) pli_table_ = g.read("per_layer_token_embd.weight");

    report_.resident_layers = resident_;
    report_.hosted_layers = L - resident_;
    report_.head_bytes = head_.bytes;
    report_.embedding_host_bytes = embd_.size() + pli_table_.size();
    report_.state_bytes = kv_bytes;
    report_.workspace_bytes = workspace;
    report_.vram_free_at_start = free_b;
    report_.vram_total = total_b;

    if (resident_ < L) {
        size_t hosted_max = 0;
        for (int i = resident_; i < L; ++i) hosted_max = std::max(hosted_max, layers_[i].matrix_bytes);
        report_.staging_bytes = 2 * hosted_max;
        for (int k = 0; k < 2; ++k) {
            arena_[k] = static_cast<uint8_t*>(counted_malloc(hosted_max));
            const size_t bytes = hosted_max;
            arena_owner_[k] = std::shared_ptr<void>(arena_[k], [bytes](void* p) { counted_free(p, bytes); });
        }
        cudaStream_t s;
        cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "create copy stream");
        copy_stream_ = s;
        for (int k = 0; k < 2; ++k) {
            cudaEvent_t e;
            cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create event");
            copy_done_[k] = e;
            cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "create event");
            compute_done_[k] = e;
        }
    }
    {   // work stream, device-resident decode position, scratch pool sized for the largest chunk (so a captured graph never sees an allocation)
        graph_.create_stream();
        d_pos_ = DevBuf<int64_t>(1);
        d_argmax_ = DevBuf<unsigned char>(argmax_scratch_bytes() + 16);
        void* hp = nullptr;
        if (cudaHostAlloc(&hp, 2 * sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); hp = std::malloc(2 * sizeof(int64_t)); }
        h_pos_ = static_cast<int64_t*>(hp);
        h_next_ = reinterpret_cast<int*>(h_pos_ + 1);  // shares the pinned allocation (freed with h_pos_)
        if (const char* e = std::getenv("DARA_NO_GRAPH")) use_graphs_ = !(e[0] == '1');
        reserve_pool();
    }
    load_seconds_ = since(t_start);
  } catch (...) {
    release_raw();  // a constructor that throws never runs the destructor
    throw;
  }
}

void GemmaModel::reserve_pool() {
    const size_t C = static_cast<size_t>(std::max(1, opt_.max_chunk_tokens));
    const size_t H = static_cast<size_t>(cfg_.hidden), PD = static_cast<size_t>(cfg_.per_layer_dim);
    size_t max_inter = 0, max_qw = 0, max_kvw = 0;
    for (int l = 0; l < cfg_.layers; ++l) {
        max_inter = std::max<size_t>(max_inter, cfg_.inter[l]);
        max_qw = std::max<size_t>(max_qw, static_cast<size_t>(cfg_.heads) * cfg_.head_dim(l));
        max_kvw = std::max<size_t>(max_kvw, static_cast<size_t>(cfg_.kv_heads) * cfg_.head_dim(l));
    }
    pool_.get<float>(1, C * H); pool_.get<float>(2, C * max_qw); pool_.get<float>(3, C * max_qw); pool_.get<float>(4, C * H);
    pool_.get<float>(5, C * H); pool_.get<float>(6, C * max_inter); pool_.get<float>(7, C * max_inter); pool_.get<float>(8, C * H);
    pool_.get<float>(9, C * PD); pool_.get<float>(10, C * H); pool_.get<int64_t>(11, C);
    pool_.get<float>(12, C * max_kvw); pool_.get<float>(13, C * max_kvw);
    pool_.get<float>(14, 8 * attn_splitkv_scratch_floats_per_token(opt_.kv_format == KvFormat::TQ4 ? AttnFmt::TQ4 : opt_.kv_format == KvFormat::Q8 ? AttnFmt::Q8 : opt_.kv_format == KvFormat::K16V8 ? AttnFmt::K16V8 : AttnFmt::F32, cfg_.heads, cfg_.head_dim_full, static_cast<int>(capacity_)));  // split-KV partials
}

void GemmaModel::release_raw() {
    // Idempotent: runs from the destructor AND from a constructor that throws part-way (its destructor never runs).
    for (void*& e : prof_events_) if (e) { cudaEventDestroy(static_cast<cudaEvent_t>(e)); e = nullptr; }
    if (h_pos_) { if (cudaFreeHost(h_pos_) != cudaSuccess) cudaGetLastError(); h_pos_ = nullptr; }
    for (int k = 0; k < 2; ++k) {
        if (copy_done_[k]) { cudaEventDestroy(static_cast<cudaEvent_t>(copy_done_[k])); copy_done_[k] = nullptr; }
        if (compute_done_[k]) { cudaEventDestroy(static_cast<cudaEvent_t>(compute_done_[k])); compute_done_[k] = nullptr; }
    }
    if (copy_stream_) { cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream_)); copy_stream_ = nullptr; }
}

GemmaModel::~GemmaModel() { release_raw(); }

void GemmaModel::rewind(size_t pos) {
    if (pos > pos_) throw std::invalid_argument("cannot rewind forward");
    pos_ = pos;
}

void GemmaModel::start_copy(int layer) {
    const int slot = (layer - resident_) % 2;
    PackedLayer& p = layers_[layer];
    if (slot_used_[slot]) cuda_check(cudaStreamWaitEvent(static_cast<cudaStream_t>(copy_stream_), static_cast<cudaEvent_t>(compute_done_[slot]), 0), "copy waits for compute");
    cuda_check(cudaMemcpyAsync(arena_[slot], p.host.data(), p.matrix_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(copy_stream_)), "stage layer");
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(copy_done_[slot]), static_cast<cudaStream_t>(copy_stream_)), "record copy");
    streamed_bytes_ += p.matrix_bytes;
}

void GemmaModel::prof_mark(const char* label) {
    if (!prof_active_) return;
    if (prof_used_ >= prof_events_.size()) {
        cudaEvent_t e;
        cuda_check(cudaEventCreate(&e), "profiler event");
        prof_events_.push_back(e);
    }
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(prof_events_[prof_used_]), graph_.stream()), "profiler mark");
    if (prof_labels_.size() <= prof_used_) prof_labels_.resize(prof_used_ + 1);
    prof_labels_[prof_used_] = label;
    ++prof_used_;
}

std::string GemmaModel::profile_report() const {
    std::string r;
    char b[256];
    if (prof_tokens_ == 0) return "profiler: no single-token forwards were recorded\n";
    const double toks = static_cast<double>(prof_tokens_);
    std::snprintf(b, sizeof b, "profile over %ld decode tokens (per token): wall %.2f ms | GPU timeline %.2f ms | host enqueue %.2f ms\n", prof_tokens_, prof_wall_ms_ / toks, prof_gpu_ms_ / toks, prof_host_enqueue_ms_ / toks);
    r += b;
    std::snprintf(b, sizeof b, "  %-24s %9s %9s %7s\n", "operation", "ms/token", "marks/tok", "share");
    r += b;
    for (const auto& kv : prof_agg_) {
        std::snprintf(b, sizeof b, "  %-24s %9.3f %9.1f %6.1f%%\n", kv.first.c_str(), kv.second.ms / toks, kv.second.count / toks, 100.0 * kv.second.ms / prof_gpu_ms_);
        r += b;
    }
    return r;
}

// ---- one decoder layer on n tokens (a chunk) ------------------------------------------------------------------------------------
// x -> out: both device [n][hidden], distinct buffers. pli_l: this layer's per-layer input, device [n][per_layer_dim].
void GemmaModel::run_layer(int l, int T, int c0, int n, const float* x, float* out, const float* pli_l, size_t pos0, bool use_dev_pos) {
    (void)T; (void)c0;
    const GemmaConfig& c = cfg_;
    const Small& s = small_[l];
    PackedLayer& p = layers_[l];
    void* st = graph_.stream_ptr();
    const int H = c.hidden, heads = c.heads, kvh = c.kv_heads, hd = c.head_dim(l), I = c.inter[l], PD = c.per_layer_dim;
    const size_t nH = static_cast<size_t>(n) * H;
    float* h1 = pool_.get<float>(1, nH);
    float* q = pool_.get<float>(2, static_cast<size_t>(n) * heads * hd);
    float* att = pool_.get<float>(3, static_cast<size_t>(n) * heads * hd);
    float* o = pool_.get<float>(4, nH);
    float* x1 = pool_.get<float>(5, nH);
    float* g = pool_.get<float>(6, static_cast<size_t>(n) * I);
    float* u = pool_.get<float>(7, static_cast<size_t>(n) * I);
    float* d = pool_.get<float>(8, nH);
    float* pg = pool_.get<float>(9, static_cast<size_t>(n) * PD);
    float* pp = pool_.get<float>(10, nH);

    // Token positions as a device array: the decode token reads the one uploaded before the step (graph-friendly); chunked prefill fills it.
    const int64_t* dpos;
    if (use_dev_pos && n == 1) {
        dpos = d_pos_.p;
    } else {
        int64_t* dp = pool_.get<int64_t>(11, static_cast<size_t>(n));
        cuda_check(fill_positions_cuda(dp, static_cast<int64_t>(pos0), n, st), "positions");
        dpos = dp;
    }
    // Score slots per attention block: fixed by capacity for the decode step (a graph cannot change it), exact otherwise.
    const int win = c.sliding[l] ? c.window : 0;
    const int cap = static_cast<int>(capacity_);
    const int smem_positions = (use_dev_pos && n == 1) ? (win > 0 ? std::min(win, cap) : cap)
                                                       : (win > 0 ? std::min(win, static_cast<int>(pos0) + n) : static_cast<int>(pos0) + n);

    cuda_check(rmsnorm_cuda(x, s.input_norm.p, n, H, c.eps, h1, st), "input norm");
    PM("norm");

    std::optional<Int8SharedInput> share_qkv(std::in_place, h1);   // q, k, v all read h1 (unchanged until the pre-feed-forward norm)
    cuda_check(matvec_multi_cuda(p.mats[kQ], h1, H, q, static_cast<size_t>(heads) * hd, n, st), "q_proj");
    PM("attn matmul (q/k/v/o)");
    cuda_check(rmsnorm_cuda(q, s.q_norm.p, n * heads, hd, c.eps, q, st), "q_norm");
    cuda_check(rope_cuda(q, n, heads, hd, hd, dpos, s.inv_freq.p, st), "rope q");
    PM("rope+qk norm");

    const int src = c.kv_src[l];
    if (c.owns_kv(l)) {
        const size_t kvw = static_cast<size_t>(kvh) * hd;
        float* k = pool_.get<float>(12, static_cast<size_t>(n) * kvw);
        float* v = pool_.get<float>(13, static_cast<size_t>(n) * kvw);
        cuda_check(matvec_multi_cuda(p.mats[kK], h1, H, k, kvw, n, st), "k_proj");
        cuda_check(matvec_multi_cuda(p.mats[kV], h1, H, v, kvw, n, st), "v_proj");
        PM("attn matmul (q/k/v/o)");
        cuda_check(rmsnorm_cuda(k, s.k_norm.p, n * kvh, hd, c.eps, k, st), "k_norm");
        cuda_check(rope_cuda(k, n, kvh, hd, hd, dpos, s.inv_freq.p, st), "rope k");
        cuda_check(rmsnorm_cuda(v, ones_.p, n * kvh, hd, c.eps, v, st), "v_norm");  // parameter-free RMSNorm
        if (kv_q8_values(opt_.kv_format)) {
            if (opt_.kv_format == KvFormat::K16V8) cuda_check(f16kv_append_cuda(k, dpos, n, kvh, hd, kv_[l].Kq.p, st), "f16 append K");
            else cuda_check(q8kv_append_cuda(k, dpos, n, kvh, hd, kv_[l].Kq.p, kv_[l].Kn.p, st), "q8 append K");
            cuda_check(q8kv_append_cuda(v, dpos, n, kvh, hd, kv_[l].Vq.p, kv_[l].Vn.p, st), "q8 append V");
        } else if (opt_.kv_format == KvFormat::TQ4) {
            cuda_check(tq_append_cuda(k, dpos, n, kvh, hd, kv_[l].Kq.p, kv_[l].Kn.p, st), "tq append K");
            cuda_check(tq_append_cuda(v, dpos, n, kvh, hd, kv_[l].Vq.p, kv_[l].Vn.p, st), "tq append V");
        } else {
            cuda_check(kv_append_cuda(kv_[l].K.p, kv_[l].V.p, k, v, dpos, n, static_cast<int>(kvw), st), "append K/V");
        }
        PM("kv append / v norm");
    }
    if (attention_reference_mode() && !kv_q8_values(opt_.kv_format)) {   // (INT8 KV has no reference kernel: always split-KV)
        if (opt_.kv_format == KvFormat::TQ4)
            cuda_check(tq_attention_cuda(q, nullptr, kv_[src].Kq.p, kv_[src].Kn.p, kv_[src].Vq.p, kv_[src].Vn.p, dpos, n, heads, kvh, hd, win, 1.0f, smem_positions, att, st), "tq attention");
        else
            cuda_check(gemma_attention_cuda(q, kv_[src].K.p, kv_[src].V.p, dpos, n, heads, kvh, hd, win, smem_positions, att, st), "attention");
    } else {
        // positions any token can attend to: window-bounded for sliding layers; fixed by the capacity for the decode step (a graph cannot change the grid)
        const int maxpos = (use_dev_pos && n == 1) ? (win > 0 ? std::min(win, cap) : cap) : (win > 0 ? std::min(win, static_cast<int>(pos0) + n) : static_cast<int>(pos0) + n);
        const AttnFmt afmt = opt_.kv_format == KvFormat::TQ4 ? AttnFmt::TQ4 : opt_.kv_format == KvFormat::Q8 ? AttnFmt::Q8 : opt_.kv_format == KvFormat::K16V8 ? AttnFmt::K16V8 : AttnFmt::F32;
        const size_t per = attn_splitkv_scratch_floats_per_token(afmt, heads, hd, maxpos);
        // decode (n == 1) uses the slot sized at construction; prefill batches up to 64 tokens per launch (the slot grows outside any graph capture)
        const size_t floats = per * static_cast<size_t>(std::min(n, 8));
        if (afmt != AttnFmt::TQ4 && n >= kPrefillAttnMinTokens && !(use_dev_pos && n == 1)) {
            // prefill chunk on an FP32 or INT8 cache: GEMM -> masked softmax -> GEMM (K/V tiles reused across 64 query rows)
            const size_t pf = attn_prefill_scratch_floats(n, static_cast<int>(pos0), win, heads);
            float* ps = pool_.get<float>(20, pf);
            const bool q8 = afmt != AttnFmt::F32;   // K/V in Kq/Vq (Q8 or K16V8)
            cuda_check(attn_prefill_cuda(afmt, q, nullptr, q8 ? static_cast<const void*>(kv_[src].Kq.p) : kv_[src].K.p, q8 ? kv_[src].Kn.p : nullptr,
                                         q8 ? static_cast<const void*>(kv_[src].Vq.p) : kv_[src].V.p, q8 ? kv_[src].Vn.p : nullptr, n, static_cast<int>(pos0), heads, kvh,
                                         hd, win, 1.0f, ps, pf, att, st),
                       "prefill attention");
        } else {
        float* scr = pool_.get<float>(14, floats);
        cuda_check(afmt != AttnFmt::F32
                       ? attn_splitkv_cuda(afmt, q, nullptr, kv_[src].Kq.p, kv_[src].Kn.p, kv_[src].Vq.p, kv_[src].Vn.p, dpos, n, heads, kvh, hd, win, 1.0f, maxpos, scr, floats, att, st)
                       : attn_splitkv_cuda(afmt, q, nullptr, kv_[src].K.p, nullptr, kv_[src].V.p, nullptr, dpos, n, heads, kvh, hd, win, 1.0f, maxpos, scr, floats, att, st),
                   "split-KV attention");
        }
    }
    PM("attention core");
    share_qkv.reset();
    cuda_check(matvec_multi_cuda(p.mats[kO], att, static_cast<size_t>(heads) * hd, o, H, n, st), "o_proj");
    PM("attn matmul (q/k/v/o)");
    cuda_check(rmsnorm_cuda(o, s.post_attn_norm.p, n, H, c.eps, o, st), "post-attention norm");
    cuda_check(add_cuda(x, o, x1, nH, st), "residual 1");
    PM("norm");

    cuda_check(rmsnorm_cuda(x1, s.pre_ff_norm.p, n, H, c.eps, h1, st), "pre-feed-forward norm");
    {
    Int8SharedInput share(h1);
    cuda_check(matvec_multi_cuda(p.mats[kGate], h1, H, g, static_cast<size_t>(I), n, st), "gate_proj");
    cuda_check(matvec_multi_cuda(p.mats[kUp], h1, H, u, static_cast<size_t>(I), n, st), "up_proj");
    }
    PM("mlp matmul (gate/up/down)");
    cuda_check(geglu_cuda(g, u, g, static_cast<size_t>(n) * I, st), "geglu");
    PM("geglu");
    cuda_check(matvec_multi_cuda(p.mats[kDown], g, static_cast<size_t>(I), d, H, n, st), "down_proj");
    PM("mlp matmul (gate/up/down)");
    cuda_check(rmsnorm_cuda(d, s.post_ff_norm.p, n, H, c.eps, d, st), "post-feed-forward norm");
    cuda_check(add_cuda(x1, d, x1, nH, st), "residual 2");
    PM("norm");

    // per-layer input: gate -> GELU * per-layer embedding -> projection -> norm -> residual
    cuda_check(matvec_multi_cuda(s.inp_gate, x1, H, pg, static_cast<size_t>(PD), n, st), "per-layer input gate");
    cuda_check(geglu_cuda(pg, pli_l, pg, static_cast<size_t>(n) * PD, st), "per-layer gate activation");
    cuda_check(matvec_multi_cuda(s.proj, pg, static_cast<size_t>(PD), pp, H, n, st), "per-layer projection");
    PM("per-layer input block");
    cuda_check(rmsnorm_cuda(pp, s.post_pli_norm.p, n, H, c.eps, pp, st), "post per-layer norm");
    cuda_check(add_cuda(x1, pp, out, nH, st), "residual 3");
    cuda_check(scale_cuda(out, s.layer_scalar, nH, st), "layer scalar");
    PM("per-layer input block");
}

// ---- forward --------------------------------------------------------------------------------------------------------------------
std::vector<float> GemmaModel::forward(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats) {
    return forward_impl(tokens, T, logits_from, layer_outputs, stats, true, nullptr);
}

std::vector<int32_t> GemmaModel::forward_argmax(const int32_t* tokens, int T, RunStats* stats) {
    std::vector<int32_t> ids;
    forward_impl(tokens, T, 0, nullptr, stats, false, nullptr, &ids);
    return ids;
}

int GemmaModel::forward_greedy(int32_t token, RunStats* stats, std::vector<float>* logits) {
    int next = 0;
    std::vector<float> lg = forward_impl(&token, 1, 0, nullptr, stats, logits != nullptr, &next);
    if (logits) *logits = std::move(lg);
    return next;
}

std::vector<float> GemmaModel::forward_impl(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats,
                                            bool copy_logits, int* next_token, std::vector<int32_t>* ids_out) {
    if (T <= 0 || logits_from < 0 || logits_from > T) throw std::invalid_argument("bad forward arguments");
    if (pos_ + static_cast<size_t>(T) > capacity_) throw std::length_error("K/V cache capacity exceeded");
    const auto t_call = Clock::now();
    const int H = cfg_.hidden, L = cfg_.layers, PD = cfg_.per_layer_dim;
    RunStats rs;
    device_stats().reset_peak();
    size_t free_min = ~size_t(0);
    auto sample_free = [&] {
        size_t fr = 0, tot = 0;
        const cudaError_t e = cudaMemGetInfo(&fr, &tot);
        if (e != cudaSuccess) std::fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(e));
        else free_min = std::min(free_min, fr);
    };
    sample_free();
    streamed_bytes_ = 0;
    if (resident_ < L) start_copy(resident_);

    // ---- host: embedding rows (x sqrt(hidden)) and per-layer token embeddings (x sqrt(per_layer_dim)) ---------------------------
    auto t0 = Clock::now();
    const size_t PL = static_cast<size_t>(L) * PD;
    std::vector<float> emb(static_cast<size_t>(T) * H), pli_tok(static_cast<size_t>(T) * PL);
    {
        const auto& ef = dara::kernels::quant_format(qtype_of(embd_type_, "token_embd.weight"));
        const size_t eb = static_cast<size_t>(H) / ef.block_weights;
        const auto& pf = dara::kernels::quant_format(qtype_of(pli_type_, "per_layer_token_embd.weight"));
        const size_t pb = PL / pf.block_weights;
        const float es = std::sqrt(static_cast<float>(H)), ps = std::sqrt(static_cast<float>(PD));
        for (int t = 0; t < T; ++t)
            if (tokens[t] < 0 || tokens[t] >= cfg_.vocab) throw std::out_of_range("token id outside the vocabulary");
        // Prompt chunks: tokens are independent, so the dequantisation (~8960 + 1536 values per token from host RAM) is split across CPU threads. Each token's
        // rows are computed exactly as in the serial loop, so the result is identical. The streamed-table path (read_rows) stays serial.
        const unsigned hw = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
        if (!pli_table_.empty() && T >= 32 && hw > 1) {
            std::vector<std::thread> pool;
            const int per = (T + static_cast<int>(hw) - 1) / static_cast<int>(hw);
            for (unsigned w = 0; w < hw; ++w) {
                const int a = static_cast<int>(w) * per, b = std::min(T, a + per);
                if (a >= b) break;
                pool.emplace_back([&, a, b] {
                    for (int t = a; t < b; ++t) {
                        float* e = &emb[static_cast<size_t>(t) * H];
                        ef.dequant_f32(embd_.data() + static_cast<size_t>(tokens[t]) * eb * ef.block_bytes, eb, e);
                        for (int i = 0; i < H; ++i) e[i] *= es;
                        float* pt = &pli_tok[static_cast<size_t>(t) * PL];
                        pf.dequant_f32(pli_table_.data() + static_cast<size_t>(tokens[t]) * pb * pf.block_bytes, pb, pt);
                        for (size_t i = 0; i < PL; ++i) pt[i] *= ps;
                    }
                });
            }
            for (auto& th : pool) th.join();
        } else {
        std::vector<uint8_t> row;
        for (int t = 0; t < T; ++t) {
            float* e = &emb[static_cast<size_t>(t) * H];
            ef.dequant_f32(embd_.data() + static_cast<size_t>(tokens[t]) * eb * ef.block_bytes, eb, e);
            for (int i = 0; i < H; ++i) e[i] *= es;
            float* pt = &pli_tok[static_cast<size_t>(t) * PL];
            if (!pli_table_.empty()) {
                pf.dequant_f32(pli_table_.data() + static_cast<size_t>(tokens[t]) * pb * pf.block_bytes, pb, pt);
            } else {
                row = g_.read_rows("per_layer_token_embd.weight", static_cast<uint64_t>(tokens[t]), 1);
                pf.dequant_f32(row.data(), pb, pt);
            }
            for (size_t i = 0; i < PL; ++i) pt[i] *= ps;
        }
        }
    }
    rs.embed_seconds = since(t0);

    // ---- device: per-layer inputs for all T tokens, laid out [layer][T][per_layer_dim] --------------------------------------------
    const size_t TH = static_cast<size_t>(T) * H;
    float* hid[2] = {pool_.get<float>(40, TH), pool_.get<float>(41, TH)};
    float* d_pli = pool_.get<float>(42, static_cast<size_t>(T) * PL);
    {
        float* d_emb = pool_.get<float>(43, TH);
        float* d_proj = pool_.get<float>(44, static_cast<size_t>(T) * PL);
        float* d_tok = pool_.get<float>(45, static_cast<size_t>(T) * PL);
        cuda_check(cudaMemcpy(d_emb, emb.data(), TH * 4, cudaMemcpyHostToDevice), "upload embeddings");
        cuda_check(cudaMemcpy(d_tok, pli_tok.data(), static_cast<size_t>(T) * PL * 4, cudaMemcpyHostToDevice), "upload per-layer token embeddings");
        cuda_check(matvec_multi_cuda(pli_proj_, d_emb, H, d_proj, PL, T), "per-layer model projection");
        cuda_check(scale_cuda(d_proj, 1.0f / std::sqrt(static_cast<float>(H)), static_cast<size_t>(T) * PL), "projection scale");
        cuda_check(rmsnorm_cuda(d_proj, pli_proj_norm_.p, T * L, PD, cfg_.eps, d_proj), "per-layer projection norm");
        cuda_check(combine_per_layer_inputs_cuda(d_proj, d_tok, d_pli, T, L, PD, 1.0f / std::sqrt(2.0f)), "combine per-layer inputs");
        cuda_check(cudaMemcpy(hid[0], d_emb, TH * 4, cudaMemcpyDeviceToDevice), "hidden init");
    }

    // ---- layers ---------------------------------------------------------------------------------------------------------------
    if (layer_outputs) layer_outputs->clear();
    const int C = std::max(1, opt_.max_chunk_tokens);
    int cur = 0;
    const bool decode1 = (T == 1);
    if (decode1) {  // the position is data, not a launch argument: upload it before the step
        *h_pos_ = static_cast<int64_t>(pos_);
        cuda_check(cudaMemcpyAsync(d_pos_.p, h_pos_, sizeof(int64_t), cudaMemcpyHostToDevice, graph_.stream()), "upload position");
    }
    const bool use_graph = use_graphs_ && decode1 && resident_ == L && !layer_outputs && !prof_on_;
    prof_active_ = prof_on_;  // single-token steps and prefill chunks alike (a prefill forward counts as one profiled call)
    prof_used_ = 0;
    const auto t_prof_start = Clock::now();
    PM("start");
    t0 = Clock::now();
    bool graph_done = false;
    if (use_graph) {
        try {
            const std::vector<void*> key = {hid[0], hid[1], d_pli, pool_.get<float>(14, 1)};
            if (!graph_.ready_for(key)) {  // (re)capture when the buffers the graph was built for moved
                graph_.capture(key, [&] {
                    int c = 0;
                    for (int l = 0; l < L; ++l) {
                        run_layer(l, 1, 0, 1, hid[c], hid[1 - c], d_pli + static_cast<size_t>(l) * PD, pos_, true);
                        c = 1 - c;
                    }
                });
            }
            graph_.launch();
            graph_done = true;
            graph_failures_ = 0;
        } catch (const GraphError&) {
            // Graph preparation or replay failed: the graph is discarded and nothing of this step ran, so it runs eagerly below.
            graph_.invalidate();
            if (++graph_failures_ >= 3) use_graphs_ = false;
        }
    }
    if (graph_done) {
        cur = L % 2;  // each layer flips the ping-pong buffers
    } else {
        for (int l = 0; l < L; ++l) {
            PackedLayer& p = layers_[l];
            if (l >= resident_) {
                const int slot = (l - resident_) % 2;
                const auto tw = Clock::now();
                cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(copy_done_[slot])), "wait for staged layer");
                rs.copy_wait_seconds += since(tw);
                bind_matrices(p, arena_owner_[slot], arena_[slot]);
                if (l + 1 < L) start_copy(l + 1);
            }
            for (int c0 = 0; c0 < T; c0 += C) {
                const int n = std::min(C, T - c0);
                run_layer(l, T, c0, n, hid[cur] + static_cast<size_t>(c0) * H, hid[1 - cur] + static_cast<size_t>(c0) * H,
                          d_pli + (static_cast<size_t>(l) * T + c0) * PD, pos_ + static_cast<size_t>(c0), decode1);
            }
            if (l >= resident_) {
                const int slot = (l - resident_) % 2;
                cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(compute_done_[slot]), graph_.stream()), "record compute done");
                slot_used_[slot] = true;
            }
            cur = 1 - cur;
            if (layer_outputs) {
                std::vector<float> lo(TH);
                cuda_check(cudaMemcpy(lo.data(), hid[cur], TH * 4, cudaMemcpyDeviceToHost), "layer output");
                layer_outputs->push_back(std::move(lo));
            }
        }
    }
    PM("end");
    const double host_enqueue_ms = since(t_prof_start) * 1000.0;
    cuda_check(cudaDeviceSynchronize(), "layers");
    if (prof_active_ && prof_used_ >= 2) {
        const double wall_ms = since(t_prof_start) * 1000.0;
        float total = 0;
        cuda_check(cudaEventElapsedTime(&total, static_cast<cudaEvent_t>(prof_events_[0]), static_cast<cudaEvent_t>(prof_events_[prof_used_ - 1])), "profiler total");
        for (size_t i = 1; i + 1 < prof_used_; ++i) {
            float ms = 0;
            cuda_check(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(prof_events_[i - 1]), static_cast<cudaEvent_t>(prof_events_[i])), "profiler delta");
            auto it = std::find_if(prof_agg_.begin(), prof_agg_.end(), [&](const auto& kv) { return kv.first == prof_labels_[i]; });
            if (it == prof_agg_.end()) { prof_agg_.push_back({prof_labels_[i], {}}); it = prof_agg_.end() - 1; }
            it->second.ms += ms;
            it->second.count += 1;
        }
        prof_gpu_ms_ += total;
        prof_host_enqueue_ms_ += host_enqueue_ms;
        prof_wall_ms_ += wall_ms;
        prof_tokens_ += 1;
    }
    prof_active_ = false;
    rs.compute_seconds = since(t0) - rs.copy_wait_seconds;
    sample_free();
    pos_ += static_cast<size_t>(T);

    // ---- final norm, tied head, soft-cap ----------------------------------------------------------------------------------------
    t0 = Clock::now();
    const int n_out = T - logits_from;
    std::vector<float> logits(copy_logits ? static_cast<size_t>(n_out) * cfg_.vocab : 0);
    if (n_out > 0) {
        float* xn = pool_.get<float>(46, static_cast<size_t>(n_out) * H);
        float* dy = pool_.get<float>(47, static_cast<size_t>(n_out) * cfg_.vocab);
        cuda_check(rmsnorm_cuda(hid[cur] + static_cast<size_t>(logits_from) * H, out_norm_.p, n_out, H, cfg_.eps, xn), "final norm");
        cuda_check(matvec_multi_cuda(head_, xn, static_cast<size_t>(H), dy, static_cast<size_t>(cfg_.vocab), n_out), "lm_head");
        if (cfg_.softcap > 0.0f) cuda_check(softcap_cuda(dy, cfg_.softcap, static_cast<size_t>(n_out) * cfg_.vocab), "softcap");
        if (next_token) {  // greedy: argmax of the last position on the device, 4 bytes back
            const float* last = dy + static_cast<size_t>(n_out - 1) * cfg_.vocab;
            int* d_out = reinterpret_cast<int*>(d_argmax_.p + argmax_scratch_bytes());
            cuda_check(argmax_cuda(last, cfg_.vocab, d_argmax_.p, d_out), "argmax");
            cuda_check(cudaMemcpy(h_next_, d_out, sizeof(int), cudaMemcpyDeviceToHost), "next token copy");
            *next_token = *h_next_;
        }
        if (ids_out) {   // greedy token after every position, on the device
            const size_t scr = argmax_scratch_bytes();
            if (d_argmax_.n < scr + static_cast<size_t>(n_out) * sizeof(int)) d_argmax_ = DevBuf<unsigned char>(scr + static_cast<size_t>(std::max(n_out, 16)) * sizeof(int));
            int* d_ids = reinterpret_cast<int*>(d_argmax_.p + scr);
            for (int t = 0; t < n_out; ++t) cuda_check(argmax_cuda(dy + static_cast<size_t>(t) * cfg_.vocab, cfg_.vocab, d_argmax_.p, d_ids + t), "argmax");
            ids_out->assign(static_cast<size_t>(n_out), 0);
            cuda_check(cudaMemcpy(ids_out->data(), d_ids, static_cast<size_t>(n_out) * sizeof(int), cudaMemcpyDeviceToHost), "ids copy");
        }
        if (copy_logits) cuda_check(cudaMemcpy(logits.data(), dy, logits.size() * 4, cudaMemcpyDeviceToHost), "logits copy");
    }
    rs.head_seconds = since(t0);
    sample_free();
    rs.seconds = since(t_call);
    rs.streamed_bytes = streamed_bytes_;
    rs.device_peak_bytes = device_stats().peak.load();
    rs.device_free_min_bytes = free_min;
    if (stats) *stats = rs;
    return logits;
}

}  // namespace dara::model
