#pragma once
// Builds a DecoderLayerWeights from a layer golden file (tools/make_layer_golden.py). Every dimension is derived
// from tensor shapes, never from constants, so the same code serves the 4B, 9B and 27B tiers.
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "model/layer.h"
#include "model/ops.h"
#include "support/golden.h"

namespace dara::test {

using dara::model::AttentionWeights;
using dara::model::DecoderLayerWeights;
using dara::model::LayerKind;

inline DecoderLayerWeights layer_from_golden(std::map<std::string, GoldenTensor>& g) {
    auto W = [&](const std::string& n) -> GoldenTensor& {
        auto it = g.find("w." + n);
        if (it == g.end()) throw std::runtime_error("golden file lacks tensor w." + n);
        return it->second;
    };
    DecoderLayerWeights w;
    w.kind = g["meta.kind"].f[0] == 0.0f ? LayerKind::DeltaNet : LayerKind::Attention;
    w.hidden = static_cast<int>(g["x"].dims[1]);
    w.eps = 1e-6f;
    w.input_norm = dara::model::fold_plus_one(W("input_layernorm.weight").f);       // GGUF convention: (1 + w) folded
    w.post_norm = dara::model::fold_plus_one(W("post_attention_layernorm.weight").f);
    w.mlp.hidden = w.hidden;
    w.mlp.inter = static_cast<int>(W("mlp.gate_proj.weight").dims[0]);
    w.mlp.gate = std::move(W("mlp.gate_proj.weight").f);
    w.mlp.up = std::move(W("mlp.up_proj.weight").f);
    w.mlp.down = std::move(W("mlp.down_proj.weight").f);

    if (w.kind == LayerKind::DeltaNet) {
        auto& d = w.dn.dims;
        d.hidden = w.hidden;
        const int conv_dim = static_cast<int>(W("linear_attn.in_proj_qkv.weight").dims[0]);
        const int value_dim = static_cast<int>(W("linear_attn.in_proj_z.weight").dims[0]);
        d.v_heads = static_cast<int>(W("linear_attn.in_proj_a.weight").dims[0]);
        d.k_dim = d.v_dim = 128;
        d.k_heads = (conv_dim - value_dim) / 2 / d.k_dim;
        d.conv_kernel = static_cast<int>(W("linear_attn.conv1d.weight").dims[2]);
        if (d.conv_dim() != conv_dim || d.value_dim() != value_dim) throw std::runtime_error("inconsistent DeltaNet shapes");
        w.dn.order = dara::model::VHeadOrder::Checkpoint;
        w.dn.in_qkv = std::move(W("linear_attn.in_proj_qkv.weight").f);
        w.dn.in_z = std::move(W("linear_attn.in_proj_z.weight").f);
        w.dn.in_b = std::move(W("linear_attn.in_proj_b.weight").f);
        w.dn.in_a = std::move(W("linear_attn.in_proj_a.weight").f);
        w.dn.conv_w = std::move(W("linear_attn.conv1d.weight").f);
        w.dn.dt_bias = std::move(W("linear_attn.dt_bias").f);
        w.dn.norm_w = std::move(W("linear_attn.norm.weight").f);
        w.dn.out = std::move(W("linear_attn.out_proj.weight").f);
        const auto& A = W("linear_attn.A_log").f;
        w.dn.ssm_a.resize(A.size());
        for (size_t i = 0; i < A.size(); ++i) w.dn.ssm_a[i] = -std::exp(A[i]);   // GGUF stores -exp(A_log)
    } else {
        auto& d = w.at.dims;
        d.hidden = w.hidden;
        d.head_dim = 256;
        d.heads = static_cast<int>(W("self_attn.q_proj.weight").dims[0]) / (2 * d.head_dim);
        d.kv_heads = static_cast<int>(W("self_attn.k_proj.weight").dims[0]) / d.head_dim;
        d.rot_dim = d.head_dim / 4;  // partial_rotary_factor 0.25
        w.at.q_proj = std::move(W("self_attn.q_proj.weight").f);
        w.at.k_proj = std::move(W("self_attn.k_proj.weight").f);
        w.at.v_proj = std::move(W("self_attn.v_proj.weight").f);
        w.at.o_proj = std::move(W("self_attn.o_proj.weight").f);
        w.at.q_norm = dara::model::fold_plus_one(W("self_attn.q_norm.weight").f);
        w.at.k_norm = dara::model::fold_plus_one(W("self_attn.k_norm.weight").f);
        w.at.finalize();
    }
    return w;
}

}  // namespace dara::test
