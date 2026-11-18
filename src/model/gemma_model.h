#pragma once
// Nano tier: Gemma 4 E2B on the GPU, GGUF only. Weights, dimensions and layer layout all come from the .gguf file.
//
// Architecture (verified against the Hugging Face Gemma 4 reference, see tools/ref_forward_gemma.py):
//   * 35 attention layers, no recurrence. Layer pattern from `gemma4.attention.sliding_window_pattern`: sliding-window layers
//     (window 512, head_dim 256, rope theta 1e4) and global layers (head_dim 512, "proportional" partial rope, theta 1e6).
//   * attention scaling is 1.0 (q_norm / k_norm carry the scale); V is RMS-normalised without a learned weight.
//   * the last `shared_kv_layers` (20) layers own no K/V: they read the K/V of the last own-KV layer of the same kind.
//   * four norms per layer (input, post-attention, pre-feed-forward, post-feed-forward); GELU(tanh)-gated MLP whose width is
//     doubled on the KV-sharing layers.
//   * per-layer input embeddings: a 262144 x (35*256) table (kept in host RAM) plus a projection of the token embedding feed a
//     gate/projection/norm block at the end of every layer.
//   * every layer output is multiplied by a learned scalar; final logits are soft-capped at 30.
// Residency follows the Qwen backend: VRAM-resident layers in permanent slabs, the rest in (pinned) host RAM streamed through a
// double-buffered staging arena; the planner is fed the real free VRAM and the constructor throws if the model cannot be placed.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "model/decode_graph.h"
#include "model/layer_cuda.h"
#include "model/packed_layer.h"
#include "model/qwen_model.h"  // ResidencyOptions, PlacementReport, RunStats are shared by both backends

namespace dara::model {

struct GemmaConfig {
    int layers = 0, hidden = 0, vocab = 0;
    int heads = 0, kv_heads = 0;
    int head_dim_swa = 0, head_dim_full = 0;   // per-head width of sliding / global layers
    int window = 0;                            // sliding-window size
    int per_layer_dim = 0;                     // width of the per-layer input embedding
    int shared_kv_layers = 0;
    size_t context_length = 0;                 // native maximum context (GGUF gemma4.context_length; 0 = not stated)
    float eps = 1e-6f, softcap = 0.0f;
    double theta_swa = 1e4, theta_full = 1e6;
    int rotated_pairs_full = 0;                // global layers rotate this many (leading) frequency pairs; the rest are position-independent
    std::vector<uint8_t> sliding;              // per layer: 1 = sliding window, 0 = global
    std::vector<int> inter;                    // per layer feed-forward width
    std::vector<int> kv_src;                   // per layer: itself if it owns K/V, else the layer whose K/V it reads
    int head_dim(int l) const { return sliding[l] ? head_dim_swa : head_dim_full; }
    bool owns_kv(int l) const { return kv_src[l] == l; }
    static GemmaConfig from_gguf(const dara::io::GgufCheckpoint& g);  // throws on anything unsupported or inconsistent
};

class GemmaModel {
public:
    // kv_capacity: max positions the K/V caches hold (<= kGemmaMaxContext). `g` must outlive the model.
    GemmaModel(const dara::io::GgufCheckpoint& g, size_t kv_capacity, const ResidencyOptions& opt = {});
    ~GemmaModel();
    GemmaModel(const GemmaModel&) = delete;
    GemmaModel& operator=(const GemmaModel&) = delete;

    // Runs T tokens from the current position; returns logits (soft-capped) for positions [logits_from, T) as [(T-logits_from)][vocab].
    // layer_outputs (optional) receives each layer's output [T][hidden] for validation.
    // features (optional): the final normed hidden state of every position [T][hidden] (Dara Flash drafter input).
    std::vector<float> forward(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs = nullptr,
                               RunStats* stats = nullptr, std::vector<float>* features = nullptr);
    // One greedy decode step: runs `token`, picks the next token on the GPU (argmax of the soft-capped logits, same choice as a host argmax) and copies back
    // only its id. The logits are copied to the host only when `logits` is non-null.
    int forward_greedy(int32_t token, RunStats* stats = nullptr, std::vector<float>* logits = nullptr);
    // T tokens (a speculative verification step): the greedy token after EVERY position, chosen on the GPU (only T ids come back).
    std::vector<int32_t> forward_argmax(const int32_t* tokens, int T, RunStats* stats = nullptr, std::vector<float>* features = nullptr);
    // Dequantised token_embd row (unscaled): the drafter's token input (same values as tools/gguf_dump_tensor).
    std::vector<float> embed_row_raw(int32_t token) const;

    size_t position() const { return pos_; }
    void rewind(size_t pos);  // K/V caches only need their length reset: entries past it are overwritten
    const GemmaConfig& config() const { return cfg_; }
    const PlacementReport& placement() const { return report_; }
    double load_seconds() const { return load_seconds_; }
    // Multimodal soft tokens (image encoder output): positions [start, start + n) take `rows` ([n][hidden]) as their input embedding. The
    // token ids there must be the pad id (the per-layer token table reads it); the soft tokens also feed the per-layer projection.
    void add_soft_tokens(size_t start, int n, std::vector<float> rows);
    void clear_soft_tokens() { soft_.clear(); }
    size_t kv_capacity() const { return capacity_; }   // positions the caches hold (the auto choice when constructed with 0)
    // long forward calls run in pieces of this many tokens (multiple of the prefill chunk): bounds the per-call device buffers
    int forward_piece_tokens() const { const int c = opt_.max_chunk_tokens > 0 ? opt_.max_chunk_tokens : 1; return c * (256 / c > 0 ? 256 / c : 1); }

    // Built-in profiler (no Nsight needed): when enabled, GPU timestamps are recorded between operations of every single-token forward
    // and summed per operation. profile_report() prints time per operation class, kernel-launch counts and the host enqueue time.
    void profile_enable(bool on) { prof_on_ = on; }
    // CUDA Graphs: a resident single-token decode step is captured once and replayed with one launch (the position lives in device memory,
    // so the same graph serves every position). Results are bitwise identical to the eager path; DARA_NO_GRAPH=1 or set_graphs(false) disables.
    void set_graphs(bool on) { use_graphs_ = on; }
    bool graphs_in_use() const { return graph_.valid(); }
    long graph_captures() const { return graph_.captures(); }
    long graph_replays() const { return graph_.replays(); }
    long graph_failures() const { return graph_.failures(); }
    bool graphs_enabled() const { return use_graphs_; }
    void invalidate_graph() { graph_.invalidate(); }  // forces a re-capture on the next decode step (also used by tests)
    std::string profile_report() const;

private:
    void prof_mark(const char* label);
    struct Small {                     // always-resident parameters of one layer
        DevBuf<float> input_norm, post_attn_norm, pre_ff_norm, post_ff_norm, post_pli_norm, q_norm, k_norm, inv_freq;
        DeviceMatrix inp_gate, proj;   // per-layer-input gate [256][hidden] and projection [hidden][256] (BF16, exact)
        float layer_scalar = 1.0f;
    };
    struct KV {                        // only for layers that own K/V
        DevBuf<float> K, V;            // KvFormat::F32:  [capacity][kv_heads*head_dim]
        DevBuf<uint8_t> Kq, Vq;        // KvFormat::TQ4:  packed codes [capacity][kv_heads][head_dim/2]
        DevBuf<float> Kn, Vn;          //                 and norms [capacity][kv_heads]
        DevBuf<uint8_t> Kr;            // KvFormat::Q4:   key ring (kv4.h)
        DevBuf<float> Krs;
        KvBuffers bufs() { return {&K, &V, &Kq, &Vq, &Kn, &Vn, &Kr, &Krs}; }
    };

    void start_copy(int layer);
    void run_layer(int l, int T, int c0, int n, const float* x, float* out, const float* pli_l, size_t pos0, bool use_dev_pos);
    void reserve_pool();
    void release_raw();

    const dara::io::GgufCheckpoint& g_;
    GemmaConfig cfg_;
    ResidencyOptions opt_;
    size_t capacity_;
    struct SoftSpan { size_t start; int n; std::vector<float> rows; };
    std::vector<SoftSpan> soft_;
    std::vector<PackedLayer> layers_;
    std::vector<Small> small_;
    std::vector<KV> kv_;
    DeviceMatrix head_, pli_proj_;     // tied output head (token_embd) and the per-layer-input projection of the embedding
    DevBuf<float> out_norm_, pli_proj_norm_, ones_;
    ScratchPool pool_;
    std::vector<uint8_t> embd_;        // token_embd (Q4_K etc.) in host RAM for row gathers
    std::vector<uint8_t> pli_table_;   // per_layer_token_embd in host RAM (empty if it did not fit: rows are then read on demand)
    const uint8_t* embd_p_ = nullptr;  // token_embd rows: the memory-mapped GGUF (or embd_ with DARA_EMBD_RAM=1)
    const uint8_t* pli_p_ = nullptr;   // per_layer_token_embd rows: mapped (or pli_table_); null: read_rows from the file
    uint32_t embd_type_ = 0, pli_type_ = 0;
    int resident_ = 0;
    size_t pos_ = 0;
    std::shared_ptr<void> arena_owner_[2];
    uint8_t* arena_[2] = {nullptr, nullptr};
    void* copy_stream_ = nullptr;
    void* copy_done_[2] = {nullptr, nullptr};
    void* compute_done_[2] = {nullptr, nullptr};
    bool slot_used_[2] = {false, false};
    size_t streamed_bytes_ = 0;
    // work stream (blocking, so it orders against the legacy default stream used for uploads) and CUDA-graph state
    static constexpr int kMaxGraphVerify = 8;   // verify steps of 2..8 tokens replay their own graphs (DARA_NO_VGRAPH=1: eager)
    DecodeGraph graphs_v_[kMaxGraphVerify + 1];
    int64_t* h_posv_ = nullptr;
    bool vgraphs_on_ = [] { const char* e = std::getenv("DARA_NO_VGRAPH"); return !(e && e[0] == '1'); }();
    int vgraph_failures_ = 0;
    DecodeGraph graph_;                // shared graph-decode engine (decode_graph.h); also owns the work stream
    std::vector<float> forward_impl(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs, RunStats* stats,
                                    bool copy_logits, int* next_token, std::vector<int32_t>* ids_out = nullptr, std::vector<float>* feats_out = nullptr);
    DevBuf<unsigned char> d_argmax_;   // argmax scratch + result (greedy decode)
    int* h_next_ = nullptr;            // pinned host copy of the chosen token
    DevBuf<int64_t> d_pos_;            // position of the single token being decoded, read by RoPE / KV append / attention
    int64_t* h_pos_ = nullptr;         // pinned host copy uploaded before each decode step
    bool use_graphs_ = true;
    int graph_failures_ = 0;           // consecutive graph failures (3 disable graphs)
    // profiler state
    bool prof_on_ = false, prof_active_ = false;
    std::vector<void*> prof_events_;                 // cudaEvent_t pool
    std::vector<const char*> prof_labels_;
    size_t prof_used_ = 0;
    struct ProfAgg { double ms = 0; long count = 0; };
    std::vector<std::pair<std::string, ProfAgg>> prof_agg_;
    double prof_gpu_ms_ = 0, prof_host_enqueue_ms_ = 0, prof_wall_ms_ = 0;
    long prof_tokens_ = 0;
    PlacementReport report_;
    double load_seconds_ = 0;
};

}  // namespace dara::model
