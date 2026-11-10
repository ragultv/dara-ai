#pragma once
// Gated DeltaNet decode step on the GPU: everything between the projections and the output projection.
// The projections (GEMVs) run through the quantized GEMV kernels; this covers
//   causal conv1d + silu (with its 3-step state) -> gates -> l2norm(q,k) -> delta-rule recurrence -> gated RMSNorm.
// State lives in device memory and is advanced in place, one token per call:
//   conv_state [conv_dim][conv_kernel-1] (oldest first),  S [v_heads][k_dim][v_dim] (FP32).
// A checkpoint of the layer is simply a copy of those two buffers; restore copies them back.
#include <cstddef>

#include "model/deltanet.h"

namespace dara::model {

struct DeltaNetDeviceParams {  // small per-layer tensors, device pointers, in the V-head order of `order`
    const float* conv_w;   // [conv_dim][conv_kernel]
    const float* ssm_a;    // [v_heads]  (= -exp(A_log))
    const float* dt_bias;  // [v_heads]
    const float* norm_w;   // [v_dim]
};

size_t deltanet_conv_state_floats(const DeltaNetDims& d);  // conv_dim * (conv_kernel - 1)
size_t deltanet_S_floats(const DeltaNetDims& d);           // v_heads * k_dim * v_dim

// One token. d_qkv[conv_dim], d_z[value_dim], d_b/d_a[v_heads] are the projection outputs; d_normed[value_dim]
// receives the gated-RMSNorm output (the input of out_proj). d_conv_scratch holds conv_dim floats.
// Returns a cudaError_t as int. Requires k_dim == v_dim == 128 and conv_kernel <= 8.
int deltanet_step_cuda(const DeltaNetDims& d, VHeadOrder order, const DeltaNetDeviceParams& p, float* d_conv_state,
                       float* d_S, const float* d_qkv, const float* d_z, const float* d_b, const float* d_a,
                       float* d_conv_scratch, float* d_normed, void* stream = nullptr);

}  // namespace dara::model
