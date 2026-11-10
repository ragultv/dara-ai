#pragma once
// One Qwen3.5-family decoder layer on the GPU (DeltaNet or full attention + MLP). Weights and per-sequence state are
// SEPARATE objects: weights can be streamed in and out (the full model does not fit in VRAM on small GPUs) while the
// layer's recurrent state / KV cache stays resident. Projection matrices are DeviceMatrix objects that hide their
// storage format (BF16 or the quantized GEMV formats), so layer logic does not depend on any bit layout. Dimensions come
// from the weights, so one implementation serves the 4B, 9B and 27B tiers.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels/quant.h"
#include "model/attention.h"
#include "model/deltanet.h"
#include "model/device_buffer.h"
#include "model/layer.h"

namespace dara::model {

struct DeviceMatrix {
    bool quantized = false;
    dara::kernels::QuantType qtype = dara::kernels::QuantType::Q4_K;  // when quantized
    std::shared_ptr<void> data;  // device memory, freed with the last copy
    size_t rows = 0, cols = 0;
    size_t ld = 0;               // row stride: elements (BF16) or blocks (quantized)
    size_t bytes = 0;
};

// Exact when the values are BF16-representable (checkpoint-derived weights); otherwise rounds to nearest-even.
DeviceMatrix upload_bf16_matrix(const std::vector<float>& host, size_t rows, size_t cols);
// Uploads raw BF16 bits as stored in the checkpoint (no float round trip).
DeviceMatrix upload_bf16_raw(const uint16_t* bits, size_t rows, size_t cols);
// Uploads raw quantized blocks (GGUF layout) of the given format; `row_blocks` blocks per row, rows tightly packed.
DeviceMatrix upload_quant_raw(dara::kernels::QuantType t, const void* blocks, size_t rows, size_t cols);
// y = M x (FP32 accumulation). Returns cudaError_t as int.
int matvec_cuda(const DeviceMatrix& m, const float* d_x, float* d_y, void* stream = nullptr);
// y[t*ldy ..] = M x[t*ldx ..] for t < nt: quantized weights are read once per group of up to 8 vectors, and every output is bitwise
// identical to matvec_cuda on that vector. ldx must be a multiple of 4.
int int8_init();   // allocates the int8-activation scratch when DARA_INT8=1 (call outside graph capture)
bool int8_enabled();
// Declares that d_x stays unchanged for the scope's lifetime, so the int8 path quantises it once for all the matrices that read it (Q/K/V, gate/up, ...).
// No effect on the FP32 paths. Scopes do not nest.
struct Int8SharedInput {
    explicit Int8SharedInput(const float* x);
    ~Int8SharedInput();
    Int8SharedInput(const Int8SharedInput&) = delete;
    Int8SharedInput& operator=(const Int8SharedInput&) = delete;
};
int matvec_multi_cuda(const DeviceMatrix& m, const float* d_x, size_t ldx, float* d_y, size_t ldy, int nt, void* stream = nullptr);

// Device-resident weights of one layer.
struct GpuLayerWeights {
    LayerKind kind = LayerKind::DeltaNet;
    int hidden = 0, inter = 0;
    float eps = 1e-6f;
    DevBuf<float> input_norm, post_norm;  // effective scales
    DeviceMatrix gate, up, down;
    // DeltaNet
    DeltaNetDims dn;
    VHeadOrder order = VHeadOrder::Checkpoint;
    DeviceMatrix in_qkv, in_z, in_b, in_a, dn_out;
    DevBuf<float> conv_w, ssm_a, dt_bias, norm_w;
    // Attention
    AttentionDims at;
    DeviceMatrix q_proj, k_proj, v_proj, o_proj;
    DevBuf<float> q_norm, k_norm, inv_freq;
    size_t total_bytes() const;  // device bytes held by this layer's weights
};

// Per-sequence state of one layer (stays resident while weights stream).
// KV cache storage format (per model, chosen at construction). F32 is the validation reference; TQ4 is the 4-bit TurboQuant-style cache
// (tq_kv.h): ~4.1 bits per value, lossy, so it is gated on quality (logit error / agreement) instead of bitwise identity.
enum class KvFormat { F32, TQ4, Q8, K16V8 };   // K16V8: FP16 keys + Q8 values   // Q8: int8 codes + a float scale per 32 values (attn_splitkv.h)
inline const char* to_string(KvFormat f) { return f == KvFormat::F32 ? "f32" : f == KvFormat::TQ4 ? "tq4" : f == KvFormat::Q8 ? "q8" : "k16v8"; }
inline bool kv_q8_values(KvFormat f) { return f == KvFormat::Q8 || f == KvFormat::K16V8; }   // values stored as int8 + per-32 scales
// Bytes of KV cache per position per kv head for a given format and head dim.
inline size_t kv_bytes_per_head_vector(KvFormat f, int head_dim) {
    // per head vector, averaged over K and V (callers count 2 vectors per position)
    if (f == KvFormat::K16V8) return (static_cast<size_t>(head_dim) * 3 + static_cast<size_t>(head_dim) / 32 * 4 + 1) / 2;
    return f == KvFormat::F32 ? static_cast<size_t>(head_dim) * 4 : f == KvFormat::TQ4 ? static_cast<size_t>(head_dim) / 2 + 4 : static_cast<size_t>(head_dim) + static_cast<size_t>(head_dim) / 32 * 4;
}

struct GpuLayerState {
    LayerKind kind = LayerKind::DeltaNet;
    size_t pos = 0;             // tokens processed (KV length for attention layers)
    DevBuf<float> conv_state, S;  // DeltaNet
    DevBuf<float> Kc, Vc;         // attention (KvFormat::F32), [position][kv_heads][head_dim]
    KvFormat kv_format = KvFormat::F32;
    DevBuf<uint8_t> Kq, Vq;       // attention (KvFormat::TQ4): packed 4-bit codes [position][kv_heads][head_dim/2]; (Q8): int8 codes [position][kv_heads][head_dim]
    DevBuf<float> Kn, Vn;         //                             and per-head-vector norms [position][kv_heads]; (Q8): scales [position][kv_heads][head_dim/32]
    size_t kv_cap = 0;
};
struct GpuLayerCheckpoint {
    std::vector<float> conv, S;  // DeltaNet state copy (empty for attention layers)
    size_t length = 0;           // tokens processed (attention: the KV length to truncate back to)
};

// Per-operation GPU timing of the Qwen layer path (set DARA_LAYER_PROF=1). Events are recorded between operations of every gpu_layer_forward_dev call;
// layer_prof_flush() (after the work was synchronised) folds them into per-label totals; layer_prof_report() formats them.
void layer_prof_flush();
std::string layer_prof_report();

GpuLayerWeights upload_layer_weights(const DecoderLayerWeights& host);  // dense floats -> BF16 device
// State sized for the layer described by `w`. kv_capacity applies to attention layers.
GpuLayerState make_gpu_layer_state(const GpuLayerWeights& w, size_t kv_capacity, KvFormat fmt = KvFormat::F32);
GpuLayerCheckpoint gpu_layer_checkpoint(const GpuLayerState& s);
void gpu_layer_restore(GpuLayerState& s, const GpuLayerCheckpoint& c);  // KV: can only rewind

// Processes T tokens (host x [T][hidden]) at the state's current position, advancing it; returns [T][hidden].
// `pool` supplies reusable scratch buffers (no allocation in steady state); if null, a temporary pool is used for this call.
// Device-resident variant: x and out are device pointers ([T][hidden], distinct buffers); queues all work on the default stream
// with no host transfer and no synchronization. Uses pool slots 1..7 and 10..16.
// If `tape` is non-null and the layer is DeltaNet, the recurrence inputs of the T tokens (qkv, z, b, a) are also copied there
// (deltanet_tape_floats(w, T) floats) so the state after any prefix of the T tokens can be rebuilt without re-reading any weights.
//   stream / dev_pos: CUDA-graph decode. With dev_pos non-null (a device int64 array holding each token's absolute position) the layer enqueues
//   everything on `stream`, takes positions from device memory, uses fixed shared-memory sizes, and does NOT advance s.pos (the caller owns host
//   position bookkeeping, because host code does not run again when a captured graph is replayed). dev_pos == nullptr: unchanged behaviour.
void gpu_layer_forward_dev(const GpuLayerWeights& w, GpuLayerState& s, const float* x, float* out, int T, ScratchPool& pool, float* tape = nullptr,
                           void* stream = nullptr, const int64_t* dev_pos = nullptr);
size_t deltanet_tape_floats(const GpuLayerWeights& w, int T);
// Rebuilds a DeltaNet layer's state after the first `keep` tokens of a T-token forward whose inputs are on `tape`: the caller has
// already restored the state to its value before that forward. Same kernels, same inputs => bitwise the state sequential decoding gives.
void gpu_deltanet_replay(const GpuLayerWeights& w, GpuLayerState& s, const float* tape, int T, int keep, ScratchPool& pool);
std::vector<float> gpu_layer_forward(const GpuLayerWeights& w, GpuLayerState& s, const float* x, int T, LayerTrace* trace = nullptr,
                                     ScratchPool* pool = nullptr);

// Convenience wrapper: weights + state together (used by the single-layer tests).
class GpuDecoderLayer {
public:
    GpuDecoderLayer(const DecoderLayerWeights& host, size_t kv_capacity)
        : w_(upload_layer_weights(host)), s_(make_gpu_layer_state(w_, kv_capacity)) {}
    LayerKind kind() const { return w_.kind; }
    int hidden() const { return w_.hidden; }
    std::vector<float> forward(const float* x, int T, LayerTrace* trace = nullptr) { return gpu_layer_forward(w_, s_, x, T, trace); }
    using Checkpoint = GpuLayerCheckpoint;
    std::shared_ptr<Checkpoint> checkpoint() const { return std::make_shared<Checkpoint>(gpu_layer_checkpoint(s_)); }
    void restore(const Checkpoint& c) { gpu_layer_restore(s_, c); }
    size_t position() const { return s_.pos; }

private:
    GpuLayerWeights w_;
    GpuLayerState s_;
};

}  // namespace dara::model
