#pragma once
// Gated DeltaNet token mixer of Qwen3.5 (CPU FP32 reference), following the pinned reference implementation
// (transformers Qwen3_5GatedDeltaNet, torch_recurrent_gated_delta_rule):
//
//   qkv = W_qkv x        (conv_dim = 2*key_dim + value_dim)      z = W_z x      b = W_b x      a = W_a x
//   qkv <- silu(causal depthwise conv1d(qkv))   (kernel 4, no bias, per-channel state of the last 3 inputs)
//   q,k,v = split(qkv)   q,k: k_heads x k_dim   v: v_heads x v_dim
//   beta = sigmoid(b)    g = ssm_a * softplus(a + dt_bias)       (ssm_a = -exp(A_log) < 0)
//   q,k <- l2norm(q), l2norm(k)  (eps 1e-6);   q <- q / sqrt(k_dim)
//   for each V head h with key head kh(h):
//       S <- S * exp(g_h);   kv = S^T k_kh;   delta = (v_h - kv) * beta_h;   S <- S + k_kh (x) delta;   o_h = S^T q_kh
//   y_h = RMSNorm_eps(o_h) * norm_w * silu(z_h)       (weight used as-is: NOT 1 + w)
//   out = W_out y
//
// V-HEAD ORDER IS MODEL METADATA. The checkpoint orders the 32 V heads as h = 2*kg + r (a V head's key head
// is h / ratio); the GGUF converter de-interleaves them as j = 16*r + kg (key head is j % k_heads). Every
// tensor with a V-head axis (V rows of in_qkv and conv channels, in_z rows, in_a/in_b rows, ssm_a, dt_bias,
// out columns) must be in the order the weights declare. Per-head arithmetic is order-independent; only the
// V-head -> key-head mapping differs.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/session.h"

namespace dara::model {

enum class VHeadOrder { Checkpoint, Gguf };

struct DeltaNetDims {
    int hidden = 2560;
    int k_heads = 16, v_heads = 32;
    int k_dim = 128, v_dim = 128;
    int conv_kernel = 4;
    float eps = 1e-6f;
    int key_dim() const { return k_heads * k_dim; }
    int value_dim() const { return v_heads * v_dim; }
    int conv_dim() const { return 2 * key_dim() + value_dim(); }
    int ratio() const { return v_heads / k_heads; }
};

// Key head used by V head `v` under `order`.
int key_head_for_v(VHeadOrder order, int v, const DeltaNetDims& d);
// GGUF V-head j <-> checkpoint V-head. Inverse of each other.
int checkpoint_head_of_gguf(int j, const DeltaNetDims& d);
int gguf_head_of_checkpoint(int h, const DeltaNetDims& d);

// Dense FP32 weights, all in `order`. Matrices are row-major [out][in].
struct DeltaNetWeights {
    DeltaNetDims dims;
    VHeadOrder order = VHeadOrder::Checkpoint;
    std::vector<float> in_qkv;   // [conv_dim][hidden]   rows: q (key_dim) | k (key_dim) | v (value_dim, in `order`)
    std::vector<float> in_z;     // [value_dim][hidden]
    std::vector<float> in_b;     // [v_heads][hidden]
    std::vector<float> in_a;     // [v_heads][hidden]
    std::vector<float> conv_w;   // [conv_dim][conv_kernel]
    std::vector<float> ssm_a;    // [v_heads]  = -exp(A_log)
    std::vector<float> dt_bias;  // [v_heads]
    std::vector<float> norm_w;   // [v_dim]
    std::vector<float> out;      // [hidden][value_dim]
};

// Returns the same weights re-expressed in the other V-head order (a pure permutation).
DeltaNetWeights reorder_v_heads(const DeltaNetWeights& w, VHeadOrder target);

// Per-session recurrent state. First-class session state: it clones, so it can be checkpointed and restored.
struct DeltaNetState : core::BackendState {
    DeltaNetDims dims;
    std::vector<float> conv;   // [conv_dim][conv_kernel - 1], oldest first
    std::vector<float> S;      // [v_heads][k_dim][v_dim]
    explicit DeltaNetState(const DeltaNetDims& d = {});
    std::unique_ptr<core::BackendState> clone() const override { return std::make_unique<DeltaNetState>(*this); }
    std::string owner() const override { return "deltanet-cpu"; }
    bool operator==(const DeltaNetState& o) const { return conv == o.conv && S == o.S; }
};

// Optional capture of intermediates, [T] leading, for validation against the reference.
struct DeltaNetTrace {
    std::vector<float> q_conv, k_conv;  // [T][v_heads][k_dim]  after conv+silu, repeated per V head, before l2norm
    std::vector<float> v_conv;          // [T][v_heads][v_dim]
    std::vector<float> beta, g;         // [T][v_heads]
    std::vector<float> core;            // [T][v_heads][v_dim]  recurrence output, before the gated norm
    std::vector<float> normed;          // [T][value_dim]       after gated RMSNorm
};

// The GEMV parts of the layer, exposed so other backends (GPU) can be validated stage by stage.
// qkv[conv_dim], z[value_dim], b[v_heads], a[v_heads] from one hidden vector; out[hidden] from the normed vector.
void deltanet_project(const DeltaNetWeights& w, const float* x, float* qkv, float* z, float* b, float* a);
void deltanet_out_project(const DeltaNetWeights& w, const float* normed, float* out);

// Runs T tokens sequentially (the recurrent form), advancing `state`. x: [T][hidden], out: [T][hidden].
void deltanet_forward(const DeltaNetWeights& w, DeltaNetState& state, const float* x, int T, float* out,
                      DeltaNetTrace* trace = nullptr);

}  // namespace dara::model
