#pragma once
// One Qwen3.5-family decoder layer (CPU FP32 reference), the same structure for the Core, Pro and Ultra tiers
// (4B, 9B, 27B; only the dimensions differ):
//
//   r1 = x
//   h1 = RMSNorm(x; input_norm)
//   m  = mixer(h1)                    DeltaNet (linear attention) or full attention, per layer kind
//   x1 = r1 + m
//   h2 = RMSNorm(x1; post_norm)
//   f  = W_down( silu(W_gate h2) * (W_up h2) )
//   out = x1 + f
// Norm scales are EFFECTIVE multipliers (GGUF convention, (1+w) folded). Dimensions come from the weights, never
// from constants, so a layer of any tier is described entirely by its tensors.
#include <memory>
#include <vector>

#include "model/attention.h"
#include "model/deltanet.h"

namespace dara::model {

enum class LayerKind { DeltaNet, Attention };

struct MlpWeights {
    int hidden = 0, inter = 0;
    std::vector<float> gate;  // [inter][hidden]
    std::vector<float> up;    // [inter][hidden]
    std::vector<float> down;  // [hidden][inter]
};

struct DecoderLayerWeights {
    LayerKind kind = LayerKind::DeltaNet;
    int hidden = 0;
    float eps = 1e-6f;
    std::vector<float> input_norm, post_norm;  // [hidden] effective scales
    DeltaNetWeights dn;                        // used when kind == DeltaNet
    AttentionWeights at;                       // used when kind == Attention
    MlpWeights mlp;
};

// Per-sequence state of one layer: recurrent state for DeltaNet layers, KV cache for attention layers.
struct LayerState {
    LayerKind kind = LayerKind::DeltaNet;
    std::unique_ptr<DeltaNetState> dn;
    std::unique_ptr<IAttentionCache> kv;
};
LayerState make_layer_state(const DecoderLayerWeights& w, size_t kv_capacity);

struct LayerTrace {  // [T][hidden]
    std::vector<float> normed1, mixer, resid1, normed2, mlp, out;
};

// Processes T tokens sequentially (prefill and decode are the same path), advancing `state`.
void decoder_layer_forward(const DecoderLayerWeights& w, LayerState& state, const float* x, int T, float* out,
                           LayerTrace* trace = nullptr);

}  // namespace dara::model
