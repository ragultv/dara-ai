#pragma once
// Full-attention token mixer of Qwen3.5 (CPU FP32 reference), following the pinned reference
// (transformers Qwen3_5Attention):
//
//   qg = W_q x   viewed as [heads][512] = per head [query(256) | gate(256)]        (GGUF row layout identical)
//   k  = W_k x   [kv_heads][256]            v = W_v x   [kv_heads][256]
//   q <- RMSNorm_head(query),  k <- RMSNorm_head(k)        (per head over head_dim, eps 1e-6)
//   q, k <- partial NeoX RoPE (first rot_dim = 64 of 256 dims, theta 1e7) at the token's position
//   append (k, v) to the KV cache
//   for each head h, kv head h / groups (GQA):  p = softmax_j( q . K_j / sqrt(head_dim) ) over j <= position
//   attn_h = sum_j p_j V_j
//   y = attn * sigmoid(gate)          (the output gate is applied BEFORE o_proj)
//   out = W_o y
// The norm scales are the EFFECTIVE multipliers (the GGUF tensors as-is; the checkpoint's (1 + w) is folded there).
#include <cstddef>
#include <cstdint>
#include <vector>

#include "model/kv_cache.h"

namespace dara::model {

struct AttentionDims {
    int hidden = 2560;
    int heads = 16, kv_heads = 4, head_dim = 256, rot_dim = 64;
    float eps = 1e-6f;
    double rope_theta = 1e7;
    int groups() const { return heads / kv_heads; }
    int q_rows() const { return heads * head_dim * 2; }   // query + gate
    int kv_width() const { return kv_heads * head_dim; }
    int out_width() const { return heads * head_dim; }
};

struct AttentionWeights {
    AttentionDims dims;
    std::vector<float> q_proj;      // [q_rows][hidden], rows per head: query(head_dim) then gate(head_dim)
    std::vector<float> k_proj;      // [kv_width][hidden]
    std::vector<float> v_proj;      // [kv_width][hidden]
    std::vector<float> o_proj;      // [hidden][out_width]
    std::vector<float> q_norm;      // [head_dim] effective scale
    std::vector<float> k_norm;      // [head_dim] effective scale
    std::vector<float> inv_freq;    // filled by finalize()
    void finalize();                // computes the RoPE table (reference float32 pipeline)
};

// KV cache, FP32, [position][kv_heads][head_dim]. Append-only: a checkpoint is just a length, and restoring is
// truncating to it (later entries are simply overwritten). Contrast with DeltaNet state, which must be copied.
class KvCache : public IAttentionCache {
public:
    KvCache(const AttentionDims& d, size_t capacity) : d_(d), cap_(capacity), k_(capacity * d.kv_width()), v_(capacity * d.kv_width()) {}
    size_t size() const override { return len_; }
    size_t capacity() const { return cap_; }
    void append(const float* k, const float* v) override;  // throws std::length_error when full
    void truncate(size_t n) override;                      // n <= size()
    std::vector<KvSpan> spans(size_t start, size_t length) const override;  // one contiguous span
    const float* k(size_t pos) const { return &k_[pos * d_.kv_width()]; }
    const float* v(size_t pos) const { return &v_[pos * d_.kv_width()]; }
    // Content equality of the live prefix (for tests).
    bool same_prefix(const KvCache& o) const;

private:
    AttentionDims d_;
    size_t cap_, len_ = 0;
    std::vector<float> k_, v_;
};

struct AttentionTrace {  // [T] leading
    std::vector<float> gate;      // [T][out_width]
    std::vector<float> q_normed;  // [T][heads][head_dim]   after q_norm, before RoPE
    std::vector<float> k_normed;  // [T][kv_heads][head_dim]
    std::vector<float> v;         // [T][kv_heads][head_dim]
    std::vector<float> q_rot, k_rot;
    std::vector<float> attn;      // [T][out_width]  before the gate
    std::vector<float> gated;     // [T][out_width]  after * sigmoid(gate), the input of o_proj
};

// Processes T tokens at positions cache.size() .. cache.size()+T-1 sequentially, appending to the cache.
// x: [T][hidden]; out: [T][hidden]. Prefill and single-token decode are the same code path by construction.
void attention_forward(const AttentionWeights& w, IAttentionCache& cache, const float* x, int T, float* out,
                       AttentionTrace* trace = nullptr);

}  // namespace dara::model
