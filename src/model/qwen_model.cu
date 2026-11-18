#include "model/qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "core/placement.h"
#include "core/system.h"
#include "model/attention_cuda.h"
#include "model/attn_prefill.h"
#include "model/attn_splitkv.h"
#include "model/tq_kv.h"
#include "model/ops.h"
#include "model/ops_cuda.h"

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
    for (const auto& kv : g.header().metadata) if (kv.first == a + "context_length") c.context_length = static_cast<size_t>(std::stoll(kv.second));
    {
        const auto it = g.header().int_arrays.find(a + "rope.dimension_sections");
        if (it != g.header().int_arrays.end() && it->second.size() >= 3) {
            c.mrope_t = static_cast<int>(it->second[0]); c.mrope_h = static_cast<int>(it->second[1]); c.mrope_w = static_cast<int>(it->second[2]);
        }
    }
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
    size_t max_layer = 0, state_bytes = 0, kv_per_token = 0;
    for (int i = 0; i < L; ++i) {
        layer_bytes[i] = weights.layer_matrix_bytes(i) + kSmallParamsPerLayer;
        max_layer = std::max(max_layer, weights.layer_matrix_bytes(i));
        if (cfg.kinds[i] == LayerKind::DeltaNet) {
            state_bytes += (static_cast<size_t>(cfg.v_heads) * cfg.k_dim * cfg.v_dim + static_cast<size_t>(cfg.k_heads * cfg.k_dim * 2 + cfg.v_heads * cfg.v_dim) * (cfg.conv_kernel - 1)) * sizeof(float);
        } else {
            kv_per_token += 2 * static_cast<size_t>(cfg.kv_heads) * kv_bytes_per_head_vector(opt.kv_format, cfg.head_dim);
        }
    }
    const size_t head_bytes = weights.head_bytes();
    const size_t C = static_cast<size_t>(std::max(1, opt.max_chunk_tokens));
    const size_t workspace = C * (14 * static_cast<size_t>(cfg.hidden) + 2 * static_cast<size_t>(cfg.inter) + 2 * static_cast<size_t>(cfg.heads) * cfg.head_dim * 2) * sizeof(float) + 32 * kMiB;
    // reserve: max(128 MiB, VRAM/32) -- on a 4 GB GPU 128 MiB (measured: ~150 MiB stays free in use at 21+2 9B layers, enough for 8K-token prefill scratch)
    size_t safety = opt.safety_bytes ? opt.safety_bytes : std::max<size_t>(128 * kMiB, total_b / 32);
    if (const char* e = std::getenv("DARA_VRAM_SAFETY_MB")) safety = static_cast<size_t>(std::atoll(e)) * kMiB;   // tuning / experiments
    const bool auto_ctx = kv_capacity == 0;
    const AttnFmt afmt = opt.kv_format == KvFormat::TQ4 ? AttnFmt::TQ4 : opt.kv_format == KvFormat::Q8 ? AttnFmt::Q8 : opt.kv_format == KvFormat::K16V8 ? AttnFmt::K16V8 : AttnFmt::F32;
    // VRAM that grows with the context outside the K/V caches: the caller's per-position extra (drafter K/V) and, in auto mode, the
    // scratch the pool grows into at long contexts (prefill attention scores, split-KV partials).
    auto ctx_extra = [&](size_t ctx) -> size_t {
        size_t x = opt.extra_vram_bytes + opt.extra_bytes_per_token * ctx;
        if (auto_ctx) {
            const int c = static_cast<int>(std::min<size_t>(ctx, size_t{1} << 30));
            x += (attn_prefill_scratch_floats(static_cast<int>(std::min<size_t>(C, 64)), c, 0, cfg.heads, cfg.kv_heads) +
                  8 * attn_splitkv_scratch_floats_per_token(afmt, cfg.heads, cfg.head_dim, c)) * sizeof(float);
            x += static_cast<size_t>(C * std::max<size_t>(1, 1024 / C)) * 2 * static_cast<size_t>(cfg.hidden) * sizeof(float);   // one forward piece: two [T][hidden] buffers
        }
        return x;
    };

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
    const size_t ram_margin = 1536 * kMiB;
    const char* ce = std::getenv("DARA_CPU_DECODE");
    const bool cpu_mode = !(ce && ce[0] == '0') && cpu_decode_available() && cpu_decode_supports(opt.kv_format);
    struct Planned { dara::core::PlacementPlan plan; size_t staging = 0; bool head_cpu = false; };
    auto make_plan = [&](size_t ctx) {
        Planned r;
        dara::core::PlacementRequest req;
        req.model = man;
        req.kv_mode = opt.kv_format == KvFormat::TQ4 ? dara::core::KvMode::TQ4 : kv_q8_values(opt.kv_format) ? dara::core::KvMode::Q8 : dara::core::KvMode::F32;
        req.tq_overhead = opt.kv_format == KvFormat::K16V8 ? 0.5 + 2.0 / 32 : opt.kv_format == KvFormat::Q8 ? 4.0 / 32 : 4.0 / cfg.head_dim;   // K16V8: avg 1.5 B + V scales  // Q8: one float scale per 32 values; TQ4: the per-vector float norm
        req.context_tokens = ctx;
        req.vision_resident = false;
        req.budget.vram_total = total_b;
        req.budget.vram_free = free_b;
        req.budget.runtime_reserve = 0;  // the CUDA context already exists, so free_b is net of it
        req.budget.safety_reserve = safety;
        const size_t ws = workspace + ctx_extra(ctx);
        req.budget.workspace_bytes = ws;
        req.budget.ram_free = sys.available_bytes > ram_margin ? sys.available_bytes - ram_margin : 0;
        r.plan = dara::core::plan_placement(req);
        if (r.plan.host_layers > 0) {  // hosted layers need a double-buffered staging arena in VRAM: plan again with it counted
            r.staging = 2 * align_up(max_layer);
            req.budget.workspace_bytes = ws + r.staging;
            r.plan = dara::core::plan_placement(req);
        }
        // CPU decode of hosted layers: re-plan with the head in host RAM and a single staging arena (more layers fit in VRAM)
        size_t hosted_bytes = 0;
        for (int i = static_cast<int>(r.plan.gpu_layers); i < L; ++i) hosted_bytes += layer_bytes[i];
        if (cpu_mode && r.plan.host_layers > 0 && opt.force_resident_layers < 0 && sys.available_bytes > hosted_bytes + head_bytes + ram_margin) {
            dara::core::PlacementRequest r2 = req;
            r2.model.embed_output_bytes = 0;
            r2.budget.workspace_bytes = ws + align_up(max_layer);
            const dara::core::PlacementPlan p2 = dara::core::plan_placement(r2);
            if (p2.kind != dara::core::Placement::DoesNotFit && p2.kind != dara::core::Placement::StorageBacked && p2.gpu_layers > r.plan.gpu_layers) {
                r.plan = p2;
                r.head_cpu = true;
                r.staging = align_up(max_layer);
            }
        }
        return r;
    };
    if (auto_ctx) {   // largest context with every layer and the head in VRAM; otherwise the partial-model default
        const size_t native = cfg.context_length ? cfg.context_length : 32768, step = 1024;
        auto fully_resident = [&](size_t ctx) {
            const Planned p = make_plan(ctx);
            return p.plan.kind != dara::core::Placement::DoesNotFit && p.plan.kind != dara::core::Placement::StorageBacked && p.plan.host_layers == 0 && !p.head_cpu;
        };
        const size_t lo = std::min<size_t>(4096, native);
        if (fully_resident(lo)) {
            size_t a = lo / step, b = native / step;
            while (a < b) { const size_t m = (a + b + 1) / 2; if (fully_resident(m * step)) a = m; else b = m - 1; }
            kv_capacity = a * step;
            if (kv_capacity < opt.auto_min_context) kv_capacity = std::min(native, opt.auto_min_context);   // minimum context over full residency
        } else {
            kv_capacity = std::min(native, opt.auto_partial_context);
        }
    }
    kv_capacity_ = kv_capacity;
    const size_t kv_bytes = kv_per_token * kv_capacity;
    const Planned planned = make_plan(kv_capacity);
    dara::core::PlacementPlan plan = planned.plan;
    size_t staging = planned.staging;
    if (planned.head_cpu) { head_on_cpu_ = true; nslots_ = 1; }
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
    if (head_on_cpu_) {
        // CPU head: read from the memory-mapped model file (hot pages every token; no private copy). DARA_EMBD_RAM=1 keeps a private copy.
        dara::kernels::QuantType hq;
        static const bool ram = [] { const char* e = std::getenv("DARA_EMBD_RAM"); return e && e[0] == '1'; }();
        size_t hb = 0;
        const uint8_t* mp = ram ? nullptr : weights.map_output_head(&hq, &hb);
        if (!mp) { head_host_ = weights.read_output_head(&hq); mp = head_host_.data(); hb = head_host_.size(); }
        const size_t rb = hb / static_cast<size_t>(cfg_.vocab);
        head_cpu_ = HostMatrix{mp, hq, static_cast<size_t>(cfg_.vocab), static_cast<size_t>(cfg_.hidden), rb};
    } else {
        // Q6_K head (the largest single matrix: 521 MB per token on the 4B): planar copy for the aligned int8 kernel. DARA_Q6_PLANAR=0 or the FP32
        // reference path (DARA_INT8=0) keep the GGUF layout.
        static const bool planar = [] { const char* e = std::getenv("DARA_Q6_PLANAR"); return !(e && e[0] == '0'); }();
        dara::kernels::QuantType hq = dara::kernels::QuantType::Q4_K;
        std::vector<uint8_t> raw;
        if (planar && int8_enabled()) raw = weights.read_output_head(&hq);
        if (!raw.empty() && hq == dara::kernels::QuantType::Q6_K) head_ = upload_q6k_planar(raw.data(), static_cast<size_t>(cfg_.vocab), static_cast<size_t>(cfg_.hidden));
        else if (!raw.empty()) head_ = upload_quant_raw(hq, raw.data(), static_cast<size_t>(cfg_.vocab), static_cast<size_t>(cfg_.hidden));
        else head_ = weights.load_output_head();
    }
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
    report_.head_bytes = head_on_cpu_ ? 0 : head_.bytes;
    weights.embed_row(0);  // read the embedding table into host RAM now (cold load), not on the first token
    report_.embedding_host_bytes = weights.embedding_bytes();
    report_.state_bytes = state_bytes + kv_bytes;
    report_.workspace_bytes = workspace;
    report_.vram_free_at_start = free_b;
    report_.vram_total = total_b;

    if (resident_ < L) {  // staging arenas + copy stream for hosted layers
        size_t hosted_max = 0;
        for (int i = resident_; i < L; ++i) hosted_max = std::max(hosted_max, layers_[i].matrix_bytes);
        report_.staging_bytes = static_cast<size_t>(nslots_) * hosted_max;
        for (int k = 0; k < nslots_; ++k) {
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
    if (const char* e = std::getenv("DARA_LAYER_PROF")) if (e[0] == '1' || e[0] == '2') use_graphs_ = false;   // profiling times every operation: eager steps
    reserve_pool();  // size every scratch slot for the largest chunk now, so a captured graph never sees an allocation or a moved buffer
    {   // CPU decode of hosted layers (DARA_CPU_DECODE=0 disables; DARA_CPU_THREADS overrides the thread count)
        const char* e = std::getenv("DARA_CPU_DECODE");
        const bool want = !(e && e[0] == '0');
        if (want && (resident_ < cfg_.layers || head_on_cpu_) && cpu_decode_available() && cpu_decode_supports(opt.kv_format)) {
            const char* th = std::getenv("DARA_CPU_THREADS");
            cpu_ = std::make_unique<CpuDecoder>(th ? std::atoi(th) : 0);
            for (int l = resident_; l < cfg_.layers; ++l) {
                cpu_w_.push_back(CpuDecoder::make_weights(layers_[l]));
                cpu_s_.push_back(CpuDecoder::make_state(cpu_w_.back(), kv_capacity, opt.kv_format));
            }
            report_.plan += " + CPU decode of the " + std::to_string(cfg_.layers - resident_) + " hosted layers (" + std::to_string(cpu_->threads()) + " threads)";
            if (head_on_cpu_) report_.plan += ", output head in host RAM, 1 staging arena";
        }
    }
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

void QwenModel::sync_hosted_to_cpu() {
    for (size_t i = 0; i < cpu_s_.size(); ++i) {
        const int l = resident_ + static_cast<int>(i);
        CpuDecoder::download(states_[l], cpu_s_[i], states_[l].pos);
    }
    cpu_current_ = true;
}

void QwenModel::sync_hosted_to_gpu() {
    for (size_t i = 0; i < cpu_s_.size(); ++i) {
        const int l = resident_ + static_cast<int>(i);
        CpuDecoder::upload(cpu_s_[i], states_[l], states_[l].pos);
    }
    cpu_current_ = false;
}

QwenModel::Checkpoint QwenModel::checkpoint() const {
    if (cpu_current_) const_cast<QwenModel*>(this)->sync_hosted_to_gpu();   // checkpoints copy the GPU states
    Checkpoint c;
    c.pos = pos_;
    for (const auto& s : states_) c.layers.push_back(gpu_layer_checkpoint(s));
    return c;
}

void QwenModel::restore(const Checkpoint& c) {
    if (c.layers.size() != states_.size()) throw std::invalid_argument("checkpoint does not match this model");
    for (size_t i = 0; i < states_.size(); ++i) gpu_layer_restore(states_[i], c.layers[i]);
    pos_ = c.pos;
    cpu_current_ = false;   // the GPU copy is the restored one
}

void QwenModel::snapshot() {
    // CPU decode mode: the verification step runs the hosted layers on the CPU, so their snapshot lives in host RAM
    const bool cpu_side = cpu_ != nullptr && !cpu_s_.empty();
    const size_t n_gpu = cpu_side ? static_cast<size_t>(resident_) : states_.size();
    if (cpu_side) { if (!cpu_current_) sync_hosted_to_cpu(); }
    else if (cpu_current_) sync_hosted_to_gpu();   // the snapshot shadows the GPU states
    if (shadow_conv_.empty()) {  // first use: allocate the in-VRAM shadow of every GPU-side DeltaNet layer's state
        shadow_conv_.resize(states_.size());
        shadow_S_.resize(states_.size());
        shadow_len_.assign(states_.size(), 0);
        for (size_t i = 0; i < n_gpu; ++i)
            if (states_[i].kind == LayerKind::DeltaNet) { shadow_conv_[i] = DevBuf<float>(states_[i].conv_state.n); shadow_S_[i] = DevBuf<float>(states_[i].S.n); }
    }
    cpu_snap_ = cpu_side;
    if (cpu_side) {
        cpu_shadow_conv_.resize(cpu_s_.size()); cpu_shadow_S_.resize(cpu_s_.size());
        if (cpu_tape_.size() != cpu_s_.size()) {
            cpu_tape_.resize(cpu_s_.size());
            for (size_t i = 0; i < cpu_s_.size(); ++i) cpu_tape_[i].resize(CpuDecoder::tape_floats(cpu_w_[i], kMaxSpecTokens));
        }
        for (size_t i = 0; i < cpu_s_.size(); ++i) { cpu_shadow_conv_[i] = cpu_s_[i].conv; cpu_shadow_S_[i] = cpu_s_[i].S; }
    }
    for (size_t i = 0; i < states_.size(); ++i) {
        shadow_len_[i] = states_[i].pos;
        if (states_[i].kind != LayerKind::DeltaNet || i >= n_gpu) continue;  // KV: rewinding the length is enough, entries past it are overwritten
        cuda_check(cudaMemcpyAsync(shadow_conv_[i].p, states_[i].conv_state.p, states_[i].conv_state.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "snapshot conv");
        cuda_check(cudaMemcpyAsync(shadow_S_[i].p, states_[i].S.p, states_[i].S.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "snapshot S");
    }
    shadow_pos_ = pos_;
    if (tape_.empty()) {
        tape_.resize(states_.size());
        for (size_t i = 0; i < n_gpu; ++i)
            if (states_[i].kind == LayerKind::DeltaNet) tape_[i] = DevBuf<float>(deltanet_tape_floats(layers_[i].weights, kMaxSpecTokens));
    }
    record_next_ = true;
    tape_tokens_ = 0;
}

void QwenModel::rollback(int keep) {
    if (shadow_conv_.empty()) throw std::logic_error("rollback without a snapshot");
    if (keep < 0 || (keep > 0 && keep > tape_tokens_)) throw std::invalid_argument("rollback: cannot keep more tokens than the recorded forward");
    const size_t n_gpu = cpu_snap_ ? static_cast<size_t>(resident_) : states_.size();
    if (cpu_snap_) {   // hosted layers: restore the host shadow, then replay the kept tokens from the host tape (no weight read)
        if (!cpu_current_) throw std::logic_error("rollback: CPU-side snapshot but the hosted states moved to the GPU");
        for (size_t i = 0; i < cpu_s_.size(); ++i) {
            if (cpu_w_[i].kind != LayerKind::DeltaNet) continue;
            cpu_s_[i].conv = cpu_shadow_conv_[i];
            cpu_s_[i].S = cpu_shadow_S_[i];
            if (keep > 0) cpu_->replay(cpu_w_[i], cpu_s_[i], cpu_tape_[i].data(), tape_tokens_, keep);
        }
    }
    for (size_t i = 0; i < states_.size(); ++i) {
        states_[i].pos = shadow_len_[i] + static_cast<size_t>(keep);
        if (states_[i].kind != LayerKind::DeltaNet || i >= n_gpu) continue;
        cuda_check(cudaMemcpyAsync(states_[i].conv_state.p, shadow_conv_[i].p, states_[i].conv_state.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "rollback conv");
        cuda_check(cudaMemcpyAsync(states_[i].S.p, shadow_S_[i].p, states_[i].S.n * sizeof(float), cudaMemcpyDeviceToDevice, 0), "rollback S");
        if (keep > 0) gpu_deltanet_replay(layers_[i].weights, states_[i], tape_[i].p, tape_tokens_, keep, pool_);
    }
    pos_ = shadow_pos_ + static_cast<size_t>(keep);
    record_next_ = false;
}

void QwenModel::start_copy(int layer) {
    const int slot = (layer - resident_) % nslots_;
    PackedLayer& p = layers_[layer];
    // The host runs ahead of the GPU now: do not overwrite this staging slot until compute that read it has finished.
    if (slot_used_[slot]) cuda_check(cudaStreamWaitEvent(static_cast<cudaStream_t>(copy_stream_), static_cast<cudaEvent_t>(compute_done_[slot]), 0), "copy waits for compute");
    cuda_check(cudaMemcpyAsync(arena_[slot], p.host.data(), p.matrix_bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(copy_stream_)), "stage layer");
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(copy_done_[slot]), static_cast<cudaStream_t>(copy_stream_)), "record copy");
    streamed_bytes_ += p.matrix_bytes;
}

void QwenModel::add_image(size_t start, int grid_h, int grid_w, std::vector<float> rows) {
    if (grid_h <= 0 || grid_w <= 0 || rows.size() != static_cast<size_t>(grid_h) * grid_w * cfg_.hidden)
        throw std::invalid_argument("image soft tokens must be [grid_h * grid_w][hidden]");
    if (!images_.empty() && start < images_.back().start + static_cast<size_t>(images_.back().gh) * images_.back().gw)
        throw std::invalid_argument("images must be added in order and must not overlap");
    images_.push_back({start, grid_h, grid_w, std::move(rows)});
}

void QwenModel::rope_position(size_t slot, int64_t out[3]) const {
    int64_t delta = 0;   // rotary position minus token position for text after the images seen so far
    for (const ImageSpan& sp : images_) {
        if (slot < sp.start) break;
        const int64_t c = static_cast<int64_t>(sp.start) + delta;
        const size_t n = static_cast<size_t>(sp.gh) * sp.gw;
        if (slot < sp.start + n) {
            const int64_t i = static_cast<int64_t>(slot - sp.start);
            out[0] = c; out[1] = c + i / sp.gw; out[2] = c + i % sp.gw;
            return;
        }
        delta = c + std::max(sp.gh, sp.gw) - static_cast<int64_t>(sp.start + n);
    }
    const int64_t p = static_cast<int64_t>(slot) + delta;
    out[0] = out[1] = out[2] = p;
}

std::vector<float> QwenModel::forward(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats,
                                      std::vector<int32_t>* argmax_out, std::vector<float>* features_out, int feat_from) {
    // A forward call keeps two [T][hidden] device buffers (and host copies) for the whole call, so long prompts go through in pieces.
    // Pieces are multiples of the prefill chunk, so the chunking (and the arithmetic) is unchanged.
    const int C = std::max(1, opt_.max_chunk_tokens);
    const int piece = C * std::max(1, 1024 / C);
    if (T <= piece || layer_outputs) return forward_one(tokens, T, logits_from, layer_outputs, stats, argmax_out, features_out, feat_from);
    if (logits_from < 0 || logits_from > T) throw std::invalid_argument("bad forward arguments");
    const int f0 = feat_from < 0 ? logits_from : std::min(feat_from, T);
    std::vector<float> logits, f_all;
    std::vector<int32_t> ids_all;
    RunStats acc;
    acc.device_free_min_bytes = ~size_t(0);
    for (int off = 0; off < T; off += piece) {
        const int n = std::min(piece, T - off);
        const int lf = std::clamp(logits_from - off, 0, n), ff = std::clamp(f0 - off, 0, n);
        std::vector<float> f;
        std::vector<int32_t> ids;
        RunStats rs;
        std::vector<float> lg = forward_one(tokens + off, n, lf, nullptr, &rs, argmax_out ? &ids : nullptr, features_out ? &f : nullptr, ff);
        logits.insert(logits.end(), lg.begin(), lg.end());
        if (argmax_out) ids_all.insert(ids_all.end(), ids.begin(), ids.end());
        if (features_out) f_all.insert(f_all.end(), f.begin(), f.end());
        acc.seconds += rs.seconds; acc.embed_seconds += rs.embed_seconds; acc.compute_seconds += rs.compute_seconds;
        acc.copy_wait_seconds += rs.copy_wait_seconds; acc.head_seconds += rs.head_seconds; acc.streamed_bytes += rs.streamed_bytes;
        acc.device_peak_bytes = std::max(acc.device_peak_bytes, rs.device_peak_bytes);
        acc.device_free_min_bytes = std::min(acc.device_free_min_bytes, rs.device_free_min_bytes);
    }
    if (argmax_out) *argmax_out = std::move(ids_all);
    if (features_out) *features_out = std::move(f_all);
    if (stats) *stats = acc;
    return logits;
}

std::vector<float> QwenModel::forward_one(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats,
                                      std::vector<int32_t>* argmax_out, std::vector<float>* features_out, int feat_from) {
    if (T <= 0 || logits_from < 0 || logits_from > T) throw std::invalid_argument("bad forward arguments");
    const auto t_call = Clock::now();
    const int H = cfg_.hidden, L = cfg_.layers;
    RunStats rs;
    device_stats().reset_peak();
    size_t free_min = ~size_t(0);
    auto sample_free = [&] { size_t fr = 0, tot = 0; cudaMemGetInfo(&fr, &tot); free_min = std::min(free_min, fr); };
    sample_free();
    streamed_bytes_ = 0;

    // single-token decode with hosted layers: those layers run on the CPU (no weight streaming); everything else keeps the GPU path
    // ... and so do short verification steps (<= kMaxSpecTokens tokens): every hosted weight is read from RAM once for all of them (forward_multi)
    static const bool cpu_multi = [] { const char* e = std::getenv("DARA_CPU_MULTI"); return !(e && e[0] == '0'); }();
    const bool cpu_dec = cpu_ && !layer_outputs && (T == 1 ? !record_next_ || !cpu_s_.empty() : (cpu_multi && T <= kMaxSpecTokens));
    if (cpu_dec) { if (!cpu_current_) sync_hosted_to_cpu(); }
    else if (cpu_current_) sync_hosted_to_gpu();
    const int gpu_layers = cpu_dec ? resident_ : L;
    if (cpu_dec) cpu_->set_hot(true);   // wake the CPU workers now: they will be spinning when the GPU layers finish
    if (resident_ < L && !cpu_dec) start_copy(resident_);  // prefetch the first hosted layer while the host gathers embeddings

    auto t0 = Clock::now();
    std::vector<float> hidden(static_cast<size_t>(T) * H);
    for (int t = 0; t < T; ++t) {
        if (tokens[t] < 0 || tokens[t] >= cfg_.vocab) throw std::out_of_range("token id outside the vocabulary");
        const auto row = weights_.embed_row(tokens[t]);
        std::copy(row.begin(), row.end(), hidden.begin() + static_cast<long>(t) * H);
    }
    // image prompt: soft tokens replace the placeholder embeddings; 3D rotary positions for every token of this call
    std::vector<int64_t> rope3_host;
    if (!images_.empty()) {
        for (const ImageSpan& sp : images_) {
            const size_t n = static_cast<size_t>(sp.gh) * sp.gw;
            const size_t a = std::max(sp.start, pos_), b = std::min(sp.start + n, pos_ + static_cast<size_t>(T));
            if (a < b) std::copy(sp.rows.begin() + static_cast<long>((a - sp.start) * H), sp.rows.begin() + static_cast<long>((b - sp.start) * H),
                                 hidden.begin() + static_cast<long>((a - pos_) * H));
        }
        rope3_host.resize(static_cast<size_t>(T) * 3);
        for (int t = 0; t < T; ++t) rope_position(pos_ + static_cast<size_t>(t), &rope3_host[static_cast<size_t>(t) * 3]);
        if (d_rope3_.n < rope3_host.size()) d_rope3_ = DevBuf<int64_t>(std::max<size_t>(rope3_host.size(), 3 * 1024));
        cuda_check(cudaMemcpy(d_rope3_.p, rope3_host.data(), rope3_host.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "rope positions");
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
    // With CPU decode of hosted layers, the graph covers the VRAM-resident prefix [0, gpu_layers) and the CPU runs the rest.
    const bool use_graph = use_graphs_ && images_.empty() && T == 1 && (resident_ == L || cpu_dec) && !layer_outputs && !record_next_ &&
                           pos_ + 1 <= static_cast<size_t>(kAttentionMaxKvLen);
    bool graph_done = false;
    if (use_graph) {
        for (const GpuLayerState& st : states_)
            if (st.kind == LayerKind::Attention && st.pos + 1 > st.kv_cap) throw std::length_error("KV cache capacity exceeded");
        try {
            *h_pos_ = static_cast<int64_t>(pos_);
            cuda_check(cudaMemcpyAsync(d_pos_.p, h_pos_, sizeof(int64_t), cudaMemcpyHostToDevice, graph_.stream()), "upload position");
            const std::vector<void*> key = {hid_dev[0], hid_dev[1], pool_.get<float>(1, 1), pool_.get<float>(10, 1), pool_.get<float>(17, 1),
                                            reinterpret_cast<void*>(static_cast<uintptr_t>(gpu_layers))};
            if (!graph_.ready_for(key)) {
                graph_.capture(key, [&] {
                    int c = 0;
                    for (int l = 0; l < gpu_layers; ++l) {
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
        for (int l = 0; l < gpu_layers; ++l) states_[l].pos += 1;  // host bookkeeping the replayed graph cannot do (CPU layers count their own)
        cur = gpu_layers % 2;                                      // each layer flips the ping-pong buffers
    } else {
    for (int l = 0; l < gpu_layers; ++l) {
        PackedLayer& p = layers_[l];
        if (l >= resident_) {  // hosted layer: wait for its copy, bind to the arena, and start the next layer's copy
            const int slot = (l - resident_) % nslots_;
            const auto tw = Clock::now();
            cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(copy_done_[slot])), "wait for staged layer");
            rs.copy_wait_seconds += since(tw);
            bind_matrices(p, arena_owner_[slot], arena_[slot]);
            if (l + 1 < L && nslots_ == 2) start_copy(l + 1);   // double buffer: prefetch the next layer now
        }
        float* tape = (record_next_ && T <= kMaxSpecTokens && T <= C && states_[l].kind == LayerKind::DeltaNet) ? tape_[l].p : nullptr;
        for (int c0 = 0; c0 < T; c0 += C) {
            const int n = std::min(C, T - c0);
            if (!rope3_host.empty()) { states_[l].rope3 = d_rope3_.p + static_cast<size_t>(c0) * 3; states_[l].mrope_h = cfg_.mrope_h; states_[l].mrope_w = cfg_.mrope_w; }
            gpu_layer_forward_dev(p.weights, states_[l], hid_dev[cur] + static_cast<size_t>(c0) * H, hid_dev[1 - cur] + static_cast<size_t>(c0) * H, n, pool_, tape);
            states_[l].rope3 = nullptr;
        }
        if (l >= resident_) {  // this layer's weights stay in use until its compute drains; tell the copy stream when that happens
            const int slot = (l - resident_) % nslots_;
            cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(compute_done_[slot]), 0), "record compute done");
            slot_used_[slot] = true;
            if (nslots_ == 1 && l + 1 < gpu_layers) start_copy(l + 1);   // single arena: the next copy waits for this layer's compute
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
    if (cpu_dec) {   // the hosted layers, on the CPU, in place on the downloaded hidden state
        for (int l = resident_; l < L; ++l) {
            const size_t i = static_cast<size_t>(l - resident_);
            float* tape = (record_next_ && cpu_snap_ && cpu_w_[i].kind == LayerKind::DeltaNet) ? cpu_tape_[i].data() : nullptr;
            cpu_s_[i].rope3 = rope3_host.empty() ? nullptr : rope3_host.data();
            cpu_s_[i].mrope_h = cfg_.mrope_h; cpu_s_[i].mrope_w = cfg_.mrope_w;
            if (T == 1 && !tape) cpu_->forward(cpu_w_[i], cpu_s_[i], hidden.data(), hidden.data(), static_cast<int64_t>(states_[l].pos));
            else cpu_->forward_multi(cpu_w_[i], cpu_s_[i], hidden.data(), hidden.data(), static_cast<int64_t>(states_[l].pos), T, tape);
            states_[l].pos += static_cast<size_t>(T);
            cpu_s_[i].rope3 = nullptr;
        }
    }
    layer_prof_flush();  // no-op unless DARA_LAYER_PROF=1; the copy above synchronised the stream
    rs.compute_seconds = since(t0) - rs.copy_wait_seconds;
    if (record_next_) { tape_tokens_ = (T <= kMaxSpecTokens && T <= C) ? T : 0; record_next_ = false; }
    sample_free();
    pos_ += T;

    t0 = Clock::now();
    const std::vector<float> final_scale = weights_.final_norm_scale();
    const int n_out = T - logits_from;
    const bool ids_only = argmax_out != nullptr && !head_on_cpu_;
    std::vector<float> logits(ids_only ? 0 : static_cast<size_t>(n_out) * cfg_.vocab);
    if (argmax_out) argmax_out->assign(static_cast<size_t>(std::max(0, n_out)), 0);
    if (features_out) {   // final normed hidden states for the drafter (host-side final norm, as for the head below)
        const int f0 = feat_from < 0 ? logits_from : std::min(feat_from, T);
        features_out->resize(static_cast<size_t>(T - f0) * H);
        for (int t = f0; t < T; ++t) rmsnorm(&hidden[static_cast<size_t>(t) * H], final_scale.data(), H, cfg_.eps, &(*features_out)[static_cast<size_t>(t - f0) * H]);
    }
    if (n_out > 0) {
        std::vector<float> normed(static_cast<size_t>(n_out) * H);
        for (int t = 0; t < n_out; ++t)
            rmsnorm(&hidden[static_cast<size_t>(t + logits_from) * H], final_scale.data(), H, cfg_.eps, &normed[static_cast<size_t>(t) * H]);
        // One batched projection: the output head is read once per group of 8 rows instead of once per row.
        if (head_on_cpu_) {
            if (n_out == 1) cpu_->matvec(head_cpu_, normed.data(), logits.data());
            else cpu_->matvec_multi(head_cpu_, normed.data(), n_out, logits.data());   // the head is read once for all rows
        } else {
        float* dx = pool_.get<float>(30, static_cast<size_t>(n_out) * H);
        float* dy = pool_.get<float>(31, static_cast<size_t>(n_out) * cfg_.vocab);
        cuda_check(cudaMemcpy(dx, normed.data(), static_cast<size_t>(n_out) * H * sizeof(float), cudaMemcpyHostToDevice), "head input");
        cuda_check(matvec_multi_cuda(head_, dx, static_cast<size_t>(H), dy, static_cast<size_t>(cfg_.vocab), n_out), "lm_head");
        if (ids_only) {   // greedy token of every row on the device; only n_out ints come back
            const size_t scr = argmax_scratch_bytes();
            if (d_argmax_.n < scr + static_cast<size_t>(n_out) * sizeof(int)) d_argmax_ = DevBuf<unsigned char>(scr + static_cast<size_t>(std::max(n_out, 16)) * sizeof(int));
            int* d_ids = reinterpret_cast<int*>(d_argmax_.p + scr);
            for (int t = 0; t < n_out; ++t)
                cuda_check(argmax_cuda(dy + static_cast<size_t>(t) * cfg_.vocab, cfg_.vocab, d_argmax_.p, d_ids + t), "argmax");
            cuda_check(cudaMemcpy(argmax_out->data(), d_ids, static_cast<size_t>(n_out) * sizeof(int), cudaMemcpyDeviceToHost), "ids copy");
        } else {
            cuda_check(cudaMemcpy(logits.data(), dy, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "logits copy");
        }
        }
        if (argmax_out && head_on_cpu_)   // CPU head: logits are already on the host
            for (int t = 0; t < n_out; ++t) {
                const float* r = &logits[static_cast<size_t>(t) * cfg_.vocab];
                (*argmax_out)[t] = static_cast<int32_t>(std::max_element(r, r + cfg_.vocab) - r);
            }
    }
    rs.head_seconds = since(t0);
    if (cpu_dec) cpu_->set_hot(false);
    sample_free();
    rs.seconds = since(t_call);
    rs.streamed_bytes = streamed_bytes_;
    rs.device_peak_bytes = device_stats().peak.load();
    rs.device_free_min_bytes = free_min;
    if (stats) *stats = rs;
    return logits;
}

}  // namespace dara::model
