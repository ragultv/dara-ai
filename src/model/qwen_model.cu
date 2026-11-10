#include "model/qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "core/placement.h"
#include "core/system.h"
#include "model/attention_cuda.h"
#include "model/attn_splitkv.h"
#include "model/tq_kv.h"
#include "model/ops.h"

namespace dara::model {

static std::string meta(const dara::io::GgufCheckpoint& g, const std::string& key) {
    for (const auto& kv : g.header().metadata)
        if (kv.first == key) return kv.second;
    throw std::runtime_error("GGUF metadata is missing " + key);
}
static long long meta_int(const dara::io::GgufCheckpoint& g, const std::string& key) { return std::stoll(meta(g, key)); }
static double meta_num(const dara::io::GgufCheckpoint& g, const std::string& key) { return std::stod(meta(g, key)); }

QwenConfig QwenConfig::from_gguf(const dara::io::GgufCheckpoint& g) {
    const std::string arch = meta(g, "general.architecture");
    if (arch != "qwen35") throw std::runtime_error("unsupported GGUF architecture '" + arch + "' (this backend runs 'qwen35')");
    const std::string a = arch + ".";
    QwenConfig c;
    c.layers = static_cast<int>(meta_int(g, a + "block_count"));
    c.hidden = static_cast<int>(meta_int(g, a + "embedding_length"));
    c.inter = static_cast<int>(meta_int(g, a + "feed_forward_length"));
    c.heads = static_cast<int>(meta_int(g, a + "attention.head_count"));
    c.kv_heads = static_cast<int>(meta_int(g, a + "attention.head_count_kv"));
    c.head_dim = static_cast<int>(meta_int(g, a + "attention.key_length"));
    if (meta_int(g, a + "attention.value_length") != c.head_dim) throw std::runtime_error("key_length != value_length is not supported");
    c.eps = static_cast<float>(meta_num(g, a + "attention.layer_norm_rms_epsilon"));
    c.rope_theta = meta_num(g, a + "rope.freq_base");
    c.partial_rotary = meta_num(g, a + "rope.dimension_count") / c.head_dim;
    c.conv_kernel = static_cast<int>(meta_int(g, a + "ssm.conv_kernel"));
    c.k_dim = static_cast<int>(meta_int(g, a + "ssm.state_size"));
    c.k_heads = static_cast<int>(meta_int(g, a + "ssm.group_count"));
    c.v_heads = static_cast<int>(meta_int(g, a + "ssm.time_step_rank"));
    const int inner = static_cast<int>(meta_int(g, a + "ssm.inner_size"));
    if (c.v_heads <= 0 || inner % c.v_heads != 0) throw std::runtime_error("ssm.inner_size is not a multiple of the V head count");
    c.v_dim = inner / c.v_heads;
    const int interval = static_cast<int>(meta_int(g, a + "full_attention_interval"));
    if (interval <= 0) throw std::runtime_error("bad full_attention_interval");
    for (int i = 0; i < c.layers; ++i) c.kinds.push_back((i + 1) % interval == 0 ? LayerKind::Attention : LayerKind::DeltaNet);
    // Vocabulary and head tying come from the tensors themselves.
    const auto& emb = g.info("token_embd.weight");
    if (emb.cols() != static_cast<uint64_t>(c.hidden)) throw std::runtime_error("token_embd width does not match embedding_length");
    c.vocab = static_cast<int>(emb.rows());
    c.tied_embeddings = !g.has("output.weight");
    if (c.k_dim != 128 || c.v_dim != 128 || c.head_dim != 256) throw std::runtime_error("unsupported head dimensions");
    if (c.v_heads % c.k_heads != 0 || c.heads % c.kv_heads != 0) throw std::runtime_error("inconsistent head counts");
    return c;
}

namespace {
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
constexpr size_t kMiB = size_t(1) << 20;
constexpr size_t kSmallParamsPerLayer = kMiB;  // norms, conv, gates, rope table: well under this, reserved per layer
}  // namespace

QwenModel::QwenModel(const QwenConfig& cfg, const IModelWeights& weights, size_t kv_capacity, const ResidencyOptions& opt)
    : cfg_(cfg), weights_(weights), opt_(opt) {
  try {
    kv_capacity_ = kv_capacity;
    const auto t_start = Clock::now();
    cudaFree(nullptr);  // create the CUDA context first so the free-memory reading below already accounts for it
    report_.gpu = require_supported_gpu();
    size_t free_b = 0, total_b = 0;
    cuda_check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");

    // ---- sizes ----------------------------------------------------------------------------------------------
    const int L = cfg.layers;
    std::vector<size_t> layer_bytes(L);
    size_t max_layer = 0, state_bytes = 0, kv_bytes = 0;
    for (int i = 0; i < L; ++i) {
        layer_bytes[i] = weights.layer_matrix_bytes(i) + kSmallParamsPerLayer;
        max_layer = std::max(max_layer, weights.layer_matrix_bytes(i));
        if (cfg.kinds[i] == LayerKind::DeltaNet) {
            state_bytes += (static_cast<size_t>(cfg.v_heads) * cfg.k_dim * cfg.v_dim + static_cast<size_t>(cfg.k_heads * cfg.k_dim * 2 + cfg.v_heads * cfg.v_dim) * (cfg.conv_kernel - 1)) * sizeof(float);
        } else {
            kv_bytes += 2 * kv_capacity * static_cast<size_t>(cfg.kv_heads) * kv_bytes_per_head_vector(opt.kv_format, cfg.head_dim);
        }
    }
    const size_t head_bytes = weights.head_bytes();
    const size_t C = static_cast<size_t>(std::max(1, opt.max_chunk_tokens));
    const size_t workspace = C * (14 * static_cast<size_t>(cfg.hidden) + 2 * static_cast<size_t>(cfg.inter) + 2 * static_cast<size_t>(cfg.heads) * cfg.head_dim * 2) * sizeof(float) + 32 * kMiB;
    const size_t safety = opt.safety_bytes ? opt.safety_bytes : std::max<size_t>(256 * kMiB, total_b / 16);

    // ---- plan -----------------------------------------------------------------------------------------------
    dara::core::ModelManifest man;
    man.family = "qwen35";
    man.kv_heads = static_cast<uint32_t>(cfg.kv_heads);
    man.head_dim = static_cast<uint32_t>(cfg.head_dim);
    man.recurrent_state_bytes = state_bytes;
    man.embed_output_bytes = head_bytes;
    for (int i = 0; i < L; ++i) {
        dara::core::LayerSpec ls;
        ls.kind = cfg.kinds[i] == LayerKind::DeltaNet ? dara::core::LayerKind::DeltaNet : dara::core::LayerKind::Attention;
        ls.stored_bytes = ls.weight_bytes = layer_bytes[i];
        man.layers.push_back(ls);
    }
    const auto sys = dara::core::query_system_memory();
    dara::core::PlacementRequest req;
    req.model = man;
    req.kv_mode = opt.kv_format == KvFormat::TQ4 ? dara::core::KvMode::TQ4 : kv_q8_values(opt.kv_format) ? dara::core::KvMode::Q8 : dara::core::KvMode::F32;
    req.tq_overhead = opt.kv_format == KvFormat::K16V8 ? 0.5 + 2.0 / 32 : opt.kv_format == KvFormat::Q8 ? 4.0 / 32 : 4.0 / cfg.head_dim;   // K16V8: avg 1.5 B + V scales  // Q8: one float scale per 32 values; TQ4: the per-vector float norm
    req.context_tokens = kv_capacity;
    req.vision_resident = false;
    req.budget.vram_total = total_b;
    req.budget.vram_free = free_b;
    req.budget.runtime_reserve = 0;  // the CUDA context already exists, so free_b is net of it
    req.budget.safety_reserve = safety;
    req.budget.workspace_bytes = workspace;
    const size_t ram_margin = 1536 * kMiB;
    req.budget.ram_free = sys.available_bytes > ram_margin ? sys.available_bytes - ram_margin : 0;
    dara::core::PlacementPlan plan = dara::core::plan_placement(req);
    size_t staging = 0;
    if (plan.host_layers > 0) {  // hosted layers need a double-buffered staging arena in VRAM: plan again with it counted
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
               " MiB, state+KV " + std::to_string((state_bytes + kv_bytes) / kMiB) + " MiB, workspace " + std::to_string(workspace / kMiB) +
               " MiB, safety " + std::to_string(safety / kMiB) + " MiB, available RAM " + std::to_string(sys.available_bytes / kMiB) + " MiB";
        throw std::runtime_error(why);
    }
    resident_ = resident;

    // ---- load once: states, head, then every layer ------------------------------------------------------------
    for (int i = 0; i < L; ++i) {
        GpuLayerState s;
        s.kind = cfg.kinds[i];
        if (s.kind == LayerKind::DeltaNet) {
            DeltaNetDims d;
            d.hidden = cfg.hidden; d.k_heads = cfg.k_heads; d.v_heads = cfg.v_heads; d.k_dim = cfg.k_dim; d.v_dim = cfg.v_dim;
            d.conv_kernel = cfg.conv_kernel;
            s.conv_state = DevBuf<float>(static_cast<size_t>(d.conv_dim()) * (d.conv_kernel - 1));
            s.S = DevBuf<float>(static_cast<size_t>(d.v_heads) * d.k_dim * d.v_dim);
        } else {
            s.kv_cap = kv_capacity;
            s.kv_format = opt.kv_format;
            if (kv_q8_values(opt.kv_format)) {
                if (cfg.head_dim % 128 != 0) throw std::invalid_argument("INT8 KV: head_dim must be a multiple of 128");
                const size_t vals = kv_capacity * static_cast<size_t>(cfg.kv_heads) * cfg.head_dim;
                const bool k16 = opt.kv_format == KvFormat::K16V8;
                s.Kq = DevBuf<uint8_t>(k16 ? vals * 2 : vals); s.Vq = DevBuf<uint8_t>(vals);
                if (!k16) s.Kn = DevBuf<float>(vals / 32);
                s.Vn = DevBuf<float>(vals / 32);
            } else if (opt.kv_format == KvFormat::TQ4) {
                if (!tq_supported_head_dim(cfg.head_dim) || kv_capacity > static_cast<size_t>(kTqMaxPositions)) throw std::invalid_argument("TurboQuant KV: unsupported head dimension or capacity above kTqMaxPositions");
                const size_t codes = kv_capacity * static_cast<size_t>(cfg.kv_heads) * (cfg.head_dim / 2), norms = kv_capacity * static_cast<size_t>(cfg.kv_heads);
                s.Kq = DevBuf<uint8_t>(codes); s.Vq = DevBuf<uint8_t>(codes);
                s.Kn = DevBuf<float>(norms); s.Vn = DevBuf<float>(norms);
            } else {
                const size_t kvw = static_cast<size_t>(cfg.kv_heads) * cfg.head_dim;
                s.Kc = DevBuf<float>(kv_capacity * kvw);
                s.Vc = DevBuf<float>(kv_capacity * kvw);
            }
        }
        states_.push_back(std::move(s));
    }
    head_ = weights.load_output_head();
    layers_.reserve(L);
    report_.hosted_pinned = true;
    for (int i = 0; i < L; ++i) {
        layers_.push_back(weights.load_layer_packed(i));
        PackedLayer& p = layers_.back();
        if (i < resident_) {
            make_resident(p);
            report_.resident_weight_bytes += p.matrix_bytes + p.small_param_bytes();
        } else {
            report_.hosted_weight_bytes += p.matrix_bytes;
            report_.hosted_pinned = report_.hosted_pinned && p.host.pinned();
        }
    }
    report_.resident_layers = resident_;
    report_.hosted_layers = L - resident_;
    report_.head_bytes = head_.bytes;
    weights.embed_row(0);  // read the embedding table into host RAM now (cold load), not on the first token
    report_.embedding_host_bytes = weights.embedding_bytes();
    report_.state_bytes = state_bytes + kv_bytes;
    report_.workspace_bytes = workspace;
    report_.vram_free_at_start = free_b;
    report_.vram_total = total_b;

    if (resident_ < L) {  // staging arenas + copy stream for hosted layers
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
    cuda_check(int8_init(), "int8_init");
    cuda_check(tq_init(), "tq_init");            // codebook (also read by the split-KV TurboQuant kernels)
    cuda_check(attn_splitkv_init(), "attn_splitkv_init");
    if (false) cuda_check(tq_init(), "tq_init");  // codebook to the device, outside any graph capture
    graph_.create_stream();
    d_pos_ = DevBuf<int64_t>(1);
    {
        void* hp = nullptr;
        if (cudaHostAlloc(&hp, sizeof(int64_t), cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); hp = std::malloc(sizeof(int64_t)); }
        h_pos_ = static_cast<int64_t*>(hp);
    }
    if (const char* e = std::getenv("DARA_NO_GRAPH")) use_graphs_ = !(e[0] == '1');
    reserve_pool();  // size every scratch slot for the largest chunk now, so a captured graph never sees an allocation or a moved buffer
    load_seconds_ = since(t_start);
  } catch (...) {
    release_raw();  // a constructor that throws never runs the destructor
    throw;
  }
}

void QwenModel::reserve_pool() {
    const size_t C = static_cast<size_t>(std::max(1, opt_.max_chunk_tokens));
    const size_t H = static_cast<size_t>(cfg_.hidden), I = static_cast<size_t>(cfg_.inter);
    const size_t CD = static_cast<size_t>(cfg_.k_heads) * cfg_.k_dim * 2 + static_cast<size_t>(cfg_.v_heads) * cfg_.v_dim;
    const size_t VD = static_cast<size_t>(cfg_.v_heads) * cfg_.v_dim, VH = static_cast<size_t>(cfg_.v_heads);
    const size_t QR = static_cast<size_t>(cfg_.heads) * cfg_.head_dim * 2, KV = static_cast<size_t>(cfg_.kv_heads) * cfg_.head_dim;
    const size_t OW = static_cast<size_t>(cfg_.heads) * cfg_.head_dim;
    pool_.get<float>(1, C * H); pool_.get<float>(2, C * H); pool_.get<float>(3, C * H); pool_.get<float>(4, C * H);
    pool_.get<float>(5, C * I); pool_.get<float>(6, C * I); pool_.get<float>(7, C * H);
    pool_.get<float>(10, C * std::max(CD, QR)); pool_.get<float>(11, C * std::max(VD, KV)); pool_.get<float>(12, C * std::max(VH, KV));
    pool_.get<float>(13, C * std::max(VH, OW)); pool_.get<float>(14, std::max(CD, C * OW)); pool_.get<float>(15, C * std::max(VD, OW));
    pool_.get<int64_t>(16, C);
    pool_.get<float>(17, 8 * attn_splitkv_scratch_floats_per_token(opt_.kv_format == KvFormat::TQ4 ? AttnFmt::TQ4 : opt_.kv_format == KvFormat::Q8 ? AttnFmt::Q8 : opt_.kv_format == KvFormat::K16V8 ? AttnFmt::K16V8 : AttnFmt::F32, cfg_.heads, cfg_.head_dim, static_cast<int>(kv_capacity_)));  // split-KV partials
}

void QwenModel::release_raw() {
    // Idempotent: runs from the destructor AND from a constructor that throws part-way (its destructor never runs).
    if (h_pos_) { if (cudaFreeHost(h_pos_) != cudaSuccess) cudaGetLastError(); h_pos_ = nullptr; }
    for (int k = 0; k < 2; ++k) {
        if (copy_done_[k]) { cudaEventDestroy(static_cast<cudaEvent_t>(copy_done_[k])); copy_done_[k] = nullptr; }
        if (compute_done_[k]) { cudaEventDestroy(static_cast<cudaEvent_t>(compute_done_[k])); compute_done_[k] = nullptr; }
    }
    if (copy_stream_) { cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream_)); copy_stream_ = nullptr; }
}

QwenModel::~QwenModel() { release_raw(); }

QwenModel::Checkpoint QwenModel::checkpoint() const {
    Checkpoint c;
    c.pos = pos_;
    for (const auto& s : states_) c.layers.push_back(gpu_layer_checkpoint(s));
    return c;
}

void QwenModel::restore(const Checkpoint& c) {
    if (c.layers.size() != states_.size()) throw std::invalid_argument("checkpoint does not match this model");
    for (size_t i = 0; i < states_.size(); ++i) gpu_layer_restore(states_[i], c.layers[i]);
    pos_ = c.pos;
}

void QwenModel::snapshot() {
    if (shadow_conv_.empty()) {  // first use: allocate the in-VRAM shadow of every DeltaNet layer's state
        shadow_conv_.resize(states_.size());
        shadow_S_.resize(states_.size());
        shadow_len_.assign(states_.size(), 0);
        for (size_t i = 0; i < states_.size(); ++i)
            if (states_[i].kind == LayerKind::DeltaNet) { shadow_conv_[i] = DevBuf<float>(states_[i].conv_state.n); shadow_S_[i] = DevBuf<float>(states_[i].S.n); }
    }
    for (size_t i = 0; i < states_.size(); ++i) {
        shadow_len_[i] = states_[i].pos;
        if (states_[i].kind != LayerKind::DeltaNet) continue;  // KV: rewinding the length is enough, entries past it are overwritten
        cuda_check(cudaMemcpyAsync(shadow_conv_[i].p, states_[i].conv_state.p, states_[i].conv_state.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "snapshot conv");
        cuda_check(cudaMemcpyAsync(shadow_S_[i].p, states_[i].S.p, states_[i].S.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "snapshot S");
    }
    shadow_pos_ = pos_;
    if (tape_.empty()) {
        tape_.resize(states_.size());
        for (size_t i = 0; i < states_.size(); ++i)
            if (states_[i].kind == LayerKind::DeltaNet) tape_[i] = DevBuf<float>(deltanet_tape_floats(layers_[i].weights, kMaxSpecTokens));
    }
    record_next_ = true;
    tape_tokens_ = 0;
}

void QwenModel::rollback(int keep) {
    if (shadow_conv_.empty()) throw std::logic_error("rollback without a snapshot");
    if (keep < 0 || (keep > 0 && keep > tape_tokens_)) throw std::invalid_argument("rollback: cannot keep more tokens than the recorded forward");
    for (size_t i = 0; i < states_.size(); ++i) {
        states_[i].pos = shadow_len_[i] + static_cast<size_t>(keep);
        if (states_[i].kind != LayerKind::DeltaNet) continue;
        cuda_check(cudaMemcpyAsync(states_[i].conv_state.p, shadow_conv_[i].p, states_[i].conv_state.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "rollback conv");
        cuda_check(cudaMemcpyAsync(states_[i].S.p, shadow_S_[i].p, states_[i].S.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "rollback S");
        if (keep > 0) gpu_deltanet_replay(layers_[i].weights, states_[i], tape_[i].p, tape_tokens_, keep, pool_);
    }
    pos_ = shadow_pos_ + static_cast<size_t>(keep);
    record_next_ = false;
}

void QwenModel::start_copy(int layer) {
    const int slot = (layer - resident_) % 2;
    PackedLayer& p = layers_[layer];
    // The host runs ahead of the GPU now: do not overwrite this staging slot until compute that read it has finished.
    if (slot_used_[slot]) cuda_check(cudaStreamWaitEvent(static_cast<cudaStream_t>(copy_stream_), static_cast<cudaEvent_t>(compute_done_[slot]), 0), "copy waits for compute");
    cuda_check(cudaMemcpyAsync(arena_[slot], p.host.data(), p.matrix_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(copy_stream_)), "stage layer");
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(copy_done_[slot]), static_cast<cudaStream_t>(copy_stream_)), "record copy");
    streamed_bytes_ += p.matrix_bytes;
}

std::vector<float> QwenModel::forward(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats) {
    if (T <= 0 || logits_from < 0 || logits_from > T) throw std::invalid_argument("bad forward arguments");
    const auto t_call = Clock::now();
    const int H = cfg_.hidden, L = cfg_.layers;
    RunStats rs;
    device_stats().reset_peak();
    size_t free_min = ~size_t(0);
    auto sample_free = [&] { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); free_min = std::min(free_min, fr); };
    sample_free();
    streamed_bytes_ = 0;

    if (resident_ < L) start_copy(resident_);  // prefetch the first hosted layer while the host gathers embeddings

    auto t0 = Clock::now();
    std::vector<float> hidden(static_cast<size_t>(T) * H);
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= cfg_.vocab) throw std::out_of_range("token id outside the vocabulary");
        const auto row = weights_.embed_row(tokens[t]);
        std::copy(row.begin(), row.end(), hidden.begin() + static_cast<long>(t) * H);
    }
    rs.embed_seconds = since(t0);

    if (layer_outputs) layer_outputs->clear();
    const int C = std::max(1, opt_.max_chunk_tokens);
    // The hidden state stays on the GPU for the whole token: one upload here, ping-pong buffers between layers, one download after
    // the last layer. All layer work is queued on the default stream; the host only blocks to wait for a hosted layer's weights.
    const size_t TH_all = static_cast<size_t>(T) * H;
    float* hid_dev[2] = {pool_.get<float>(40, TH_all), pool_.get<float>(41, TH_all)};
    cuda_check(cudaMemcpy(hid_dev[0], hidden.data(), TH_all * sizeof(float), cudaMemcpyHostToDevice), "upload embeddings");
    int cur = 0;
    t0 = Clock::now();
    // CUDA-graph decode: one captured launch for all layers (resident weights, single token, nothing recording or reading per-layer data).
    const bool use_graph = use_graphs_ && T == 1 && resident_ == L && !layer_outputs && !record_next_ &&
                           pos_ + 1 <= static_cast<size_t>(kAttentionMaxKvLen);
    bool graph_done = false;
    if (use_graph) {
        for (const GpuLayerState& st : states_)
            if (st.kind == LayerKind::Attention && st.pos + 1 > st.kv_cap) throw std::length_error("KV cache capacity exceeded");
        try {
            *h_pos_ = static_cast<int64_t>(pos_);
            cuda_check(cudaMemcpyAsync(d_pos_.p, h_pos_, sizeof(int64_t), cudaMemcpyHostToDevice, graph_.stream()), "upload position");
            const std::vector<void*> key = {hid_dev[0], hid_dev[1], pool_.get<float>(1, 1), pool_.get<float>(10, 1), pool_.get<float>(17, 1)};
            if (!graph_.ready_for(key)) {
                graph_.capture(key, [&] {
                    int c = 0;
                    for (int l = 0; l < L; ++l) {
                        gpu_layer_forward_dev(layers_[l].weights, states_[l], hid_dev[c], hid_dev[1 - c], 1, pool_, nullptr, graph_.stream_ptr(), d_pos_.p);
                        c = 1 - c;
                    }
                });
            }
            graph_.launch();
            graph_done = true;
            graph_failures_ = 0;
        } catch (const GraphError&) {
            // Graph preparation or replay failed: the graph is discarded and nothing of this step ran, so the step simply runs eagerly below.
            // Repeated failures switch graphs off instead of retrying forever.
            graph_.invalidate();
            if (++graph_failures_ >= 3) use_graphs_ = false;
        }
    }
    if (graph_done) {
        for (GpuLayerState& st : states_) st.pos += 1;  // host bookkeeping the replayed graph cannot do
        cur = L % 2;                                    // each layer flips the ping-pong buffers
    } else {
    for (int l = 0; l < L; ++l) {
        PackedLayer& p = layers_[l];
        if (l >= resident_) {  // hosted layer: wait for its copy, bind to the arena, and start the next layer's copy
            const int slot = (l - resident_) % 2;
            const auto tw = Clock::now();
            cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(copy_done_[slot])), "wait for staged layer");
            rs.copy_wait_seconds += since(tw);
            bind_matrices(p, arena_owner_[slot], arena_[slot]);
            if (l + 1 < L) start_copy(l + 1);
        }
        float* tape = (record_next_ && T <= kMaxSpecTokens && T <= C && states_[l].kind == LayerKind::DeltaNet) ? tape_[l].p : nullptr;
        for (int c0 = 0; c0 < T; c0 += C) {
            const int n = std::min(C, T - c0);
            gpu_layer_forward_dev(p.weights, states_[l], hid_dev[cur] + static_cast<size_t>(c0) * H, hid_dev[1 - cur] + static_cast<size_t>(c0) * H, n, pool_, tape);
        }
        if (l >= resident_) {  // this layer's weights stay in use until its compute drains; tell the copy stream when that happens
            const int slot = (l - resident_) % 2;
            cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(compute_done_[slot]), 0), "record compute done");
            slot_used_[slot] = true;
        }
        cur = 1 - cur;
        if (layer_outputs) {
            std::vector<float> lo(TH_all);
            cuda_check(cudaMemcpy(lo.data(), hid_dev[cur], TH_all * sizeof(float), cudaMemcpyDeviceToHost), "layer output");
            layer_outputs->push_back(std::move(lo));
        }
    }
    }
    cuda_check(cudaMemcpy(hidden.data(), hid_dev[cur], TH_all * sizeof(float), cudaMemcpyDeviceToHost), "download hidden");  // waits for all layers
    layer_prof_flush();  // no-op unless DARA_LAYER_PROF=1; the copy above synchronised the stream
    rs.compute_seconds = since(t0) - rs.copy_wait_seconds;
    if (record_next_) { tape_tokens_ = (T <= kMaxSpecTokens && T <= C) ? T : 0; record_next_ = false; }
    sample_free();
    pos_ += T;

    t0 = Clock::now();
    const std::vector<float> final_scale = weights_.final_norm_scale();
    const int n_out = T - logits_from;
    std::vector<float> logits(static_cast<size_t>(n_out) * cfg_.vocab);
    if (n_out > 0) {
        std::vector<float> normed(static_cast<size_t>(n_out) * H);
        for (int t = 0; t < n_out; ++t)
            rmsnorm(&hidden[static_cast<size_t>(t + logits_from) * H], final_scale.data(), H, cfg_.eps, &normed[static_cast<size_t>(t) * H]);
        // One batched projection: the output head is read once per group of 8 rows instead of once per row.
        float* dx = pool_.get<float>(30, static_cast<size_t>(n_out) * H);
        float* dy = pool_.get<float>(31, static_cast<size_t>(n_out) * cfg_.vocab);
        cuda_check(cudaMemcpy(dx, normed.data(), static_cast<size_t>(n_out) * H * sizeof(float), cudaMemcpyHostToDevice), "head input");
        cuda_check(matvec_multi_cuda(head_, dx, static_cast<size_t>(H), dy, static_cast<size_t>(cfg_.vocab), n_out), "lm_head");
        cuda_check(cudaMemcpy(logits.data(), dy, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "logits copy");
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
