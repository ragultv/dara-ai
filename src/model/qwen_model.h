#pragma once
// Full Qwen3.5-family model on the GPU with PERSISTENT weights. Dara supports GGUF models only: weights, dimensions and
// tokenizer all come from the .gguf file.
//
// Three stages, kept separate:
//   1. LOADING   (constructor, once): every layer is read from the GGUF into a contiguous host image; the placement planner
//                (fed with the real free VRAM, state, workspace and safety reserves) decides how many layers live in VRAM.
//   2. RESIDENCY (for the model's lifetime): VRAM-resident layers sit in permanent device slabs; the rest stay in host RAM
//                (pinned when the OS allows). The embedding table lives in host RAM, the output head in VRAM, and every
//                layer's sequence state (DeltaNet state / KV cache) stays on the GPU.
//   3. DECODE    (forward, many times): no disk reads and no allocation in steady state. Host-resident layers are copied into
//                a double-buffered staging arena, the copy of the next layer overlapping the compute of the current one.
// If the configured placement cannot fit, the constructor throws with the plan and the numbers instead of failing later.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "model/cpu_decode.h"
#include "model/decode_graph.h"
#include "model/layer_cuda.h"
#include "model/packed_layer.h"
#include "model/qwen_weights.h"

namespace dara::model {

struct QwenConfig {
    int hidden = 0, layers = 0, vocab = 0, inter = 0;
    int heads = 0, kv_heads = 0, head_dim = 256;
    int k_heads = 0, v_heads = 0, k_dim = 128, v_dim = 128, conv_kernel = 4;
    double rope_theta = 1e7;
    double partial_rotary = 0.25;
    float eps = 1e-6f;
    bool tied_embeddings = true;
    std::vector<LayerKind> kinds;
    int rot_dim() const { return static_cast<int>(head_dim * partial_rotary); }
    // Derives the configuration from the GGUF metadata and tensor shapes. Throws on an unsupported architecture or any
    // missing / inconsistent field (never guesses).
    static QwenConfig from_gguf(const dara::io::GgufCheckpoint& g);
};

struct ResidencyOptions {
    int max_chunk_tokens = 64;      // prefill is processed in chunks of at most this many tokens (bounds the workspace)
    int force_resident_layers = -1; // -1: let the planner decide; otherwise exactly this many layers (prefix) live in VRAM
    size_t safety_bytes = 0;        // extra VRAM kept free; 0 = max(256 MiB, 1/16 of total)
    KvFormat kv_format = KvFormat::F32;  // KV cache format of the attention layers (F32 reference or 4-bit TurboQuant)
};

struct PlacementReport {
    std::string plan;                 // human-readable planner outcome
    std::string gpu;                  // the GPU in use: name, architecture, SM count, memory
    int resident_layers = 0, hosted_layers = 0;
    size_t resident_weight_bytes = 0, hosted_weight_bytes = 0, head_bytes = 0, embedding_host_bytes = 0;
    size_t staging_bytes = 0, state_bytes = 0, workspace_bytes = 0;
    size_t vram_free_at_start = 0, vram_total = 0;
    bool hosted_pinned = true;        // false if any hosted layer fell back to pageable memory
};

struct RunStats {
    double seconds = 0;                 // whole call
    double embed_seconds = 0;           // token embedding gather (host RAM)
    double compute_seconds = 0;         // all layers' compute
    double copy_wait_seconds = 0;       // time the compute was BLOCKED waiting for a hosted layer's copy (non-overlapped transfer)
    double head_seconds = 0;            // final norm + output head + logits download
    size_t streamed_bytes = 0;          // host -> device weight bytes copied during the call
    size_t device_peak_bytes = 0;       // peak counted device allocations during the call
    size_t device_free_min_bytes = 0;   // lowest cudaMemGetInfo free during the call
};

class QwenModel {
public:
    // kv_capacity: max positions the attention layers can hold. `weights` must outlive the model.
    // Throws std::runtime_error if the model cannot be placed within the available memory.
    QwenModel(const QwenConfig& cfg, const IModelWeights& weights, size_t kv_capacity, const ResidencyOptions& opt = {});
    ~QwenModel();
    QwenModel(const QwenModel&) = delete;
    QwenModel& operator=(const QwenModel&) = delete;

    // Runs `tokens` (T ids) through all layers starting at the current position, advancing every layer's state.
    // Returns logits for positions [logits_from, T) of this call as [(T - logits_from)][vocab] floats.
    // If layer_outputs is non-null it receives each layer's output [T][hidden] (host copies, for validation).
    std::vector<float> forward(const int32_t* tokens, int T, int logits_from, std::vector<std::vector<float>>* layer_outputs = nullptr,
                               RunStats* stats = nullptr);

    size_t position() const { return pos_; }
    const QwenConfig& config() const { return cfg_; }
    const PlacementReport& placement() const { return report_; }
    double load_seconds() const { return load_seconds_; }   // cold load: disk -> host images -> VRAM slabs
    // CUDA Graphs (see decode_graph.h): a resident single-token decode step is captured once and replayed with one launch. Results are
    // bitwise identical to the eager path. Not used while layers are streamed from host RAM, when per-layer outputs are requested, or
    // right after snapshot() (the verification forward records recurrence inputs). DARA_NO_GRAPH=1 or set_graphs(false) disables.
    void set_graphs(bool on) { use_graphs_ = on; if (!on) graph_.invalidate(); }
    long graph_captures() const { return graph_.captures(); }
    long graph_replays() const { return graph_.replays(); }
    long graph_failures() const { return graph_.failures(); }
    bool graphs_enabled() const { return use_graphs_; }
    void invalidate_graph() { graph_.invalidate(); }  // forces a re-capture on the next decode step (also used by tests)

    struct Checkpoint {
        std::vector<GpuLayerCheckpoint> layers;
        size_t pos = 0;
    };
    Checkpoint checkpoint() const;           // copies DeltaNet states, records KV lengths
    void restore(const Checkpoint& c);       // rewinds the whole model to the checkpoint
    // Fast in-VRAM variant for speculative decoding: snapshot() copies DeltaNet states device-to-device (no host traffic) and
    // records KV lengths; rollback() returns the model to exactly that point. One snapshot is kept; allocated on first use.
    // The forward() right after snapshot() (at most kMaxSpecTokens tokens) also records each DeltaNet layer's recurrence inputs, so
    // rollback(keep) can rebuild the state after only the first `keep` of those tokens WITHOUT touching any weights (keep = 0:
    // plain return to the snapshot). KV caches just rewind their length; entries past it are overwritten later.
    static constexpr int kMaxSpecTokens = 16;
    void snapshot();
    void rollback(int keep = 0);

private:
    void start_copy(int layer);
    // CPU decode of the hosted layers (cpu_decode.h): single-token steps run layers [resident_, L) on the CPU from their host images.
    // The hosted layers' state is current on exactly one side; cpu_current_ says which (true: CpuLayerState is newer than the GPU copy).
    void sync_hosted_to_cpu();
    void sync_hosted_to_gpu();
    std::unique_ptr<CpuDecoder> cpu_;
    std::vector<CpuLayerWeights> cpu_w_;   // index l - resident_
    std::vector<CpuLayerState> cpu_s_;
    bool cpu_current_ = false;
    // CPU-decode placement: the output head stays in host RAM (its VRAM goes to layers, as llama.cpp does for partially offloaded models)
    // and hosted layers stream through ONE staging arena during prompt chunks.
    bool head_on_cpu_ = false;
    std::vector<uint8_t> head_host_;
    HostMatrix head_cpu_;
    int nslots_ = 2;   // staging arenas for hosted layers during prompt chunks (1 in CPU-decode placement)

    QwenConfig cfg_;
    const IModelWeights& weights_;
    ResidencyOptions opt_;
    std::vector<GpuLayerState> states_;
    std::vector<PackedLayer> layers_;
    DeviceMatrix head_;
    ScratchPool pool_;
    int resident_ = 0;                       // layers [0, resident_) are VRAM-resident; the rest are hosted
    size_t pos_ = 0;
    DecodeGraph graph_;
    DevBuf<int64_t> d_pos_;          // position of the single decode token, read by RoPE / KV append / attention inside the graph
    int64_t* h_pos_ = nullptr;       // pinned host copy, uploaded before each replay
    bool use_graphs_ = true;
    int graph_failures_ = 0;         // consecutive graph failures (3 disable graphs)
    size_t kv_capacity_ = 0;
    void reserve_pool();
    void release_raw();
    // staging arenas for hosted layers (double buffered)
    std::shared_ptr<void> arena_owner_[2];
    uint8_t* arena_[2] = {nullptr, nullptr};
    void* copy_stream_ = nullptr;            // cudaStream_t
    void* copy_done_[2] = {nullptr, nullptr};  // cudaEvent_t: staged layer's copy finished
    void* compute_done_[2] = {nullptr, nullptr};  // cudaEvent_t: compute that read the staging slot has drained
    bool slot_used_[2] = {false, false};
    std::vector<DevBuf<float>> shadow_conv_, shadow_S_;
    std::vector<size_t> shadow_len_;
    size_t shadow_pos_ = 0;
    std::vector<DevBuf<float>> tape_;   // per layer (DeltaNet): recorded recurrence inputs of the last snapshot-following forward
    bool record_next_ = false;
    int tape_tokens_ = 0;               // tokens recorded (0 = none)
    size_t streamed_bytes_ = 0;              // since the last forward() began
    PlacementReport report_;
    double load_seconds_ = 0;
};

}  // namespace dara::model
