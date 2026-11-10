#pragma once
// CUDA pieces of the full-attention layer (correctness-first; see attention.h for the equations).
// The projections (GEMVs) use the quantized/BF16 GEMV kernels, the per-head norms and RoPE use ops_cuda.h; this
// header covers the rest: splitting q|gate, KV append, and causal GQA attention with the fused output gate.
// KV cache layout (device, FP32): [position][kv_heads][head_dim] for K and V.
#include <cstdint>

#include "model/attention.h"

namespace dara::model {

// Max number of cached positions the single-block-per-(head, query) kernel supports (scores live in shared memory).
// Longer contexts need split-KV attention (not implemented yet): the call returns an error instead.
constexpr int kAttentionMaxKvLen = 12000;  // 48 KB default dynamic smem minus the kernel's static buffers

// qg: [T][heads][2*head_dim] (per head [query | gate])  ->  q: [T][heads][head_dim], gate: [T][heads*head_dim].
int attn_split_qgate_cuda(const AttentionDims& d, const float* d_qg, int T, float* d_q, float* d_gate, void* stream = nullptr);

// Writes n tokens of k, v ([n][kv_heads*head_dim]) into the cache at positions pos0 .. pos0+n-1.
int attn_kv_append_cuda(const AttentionDims& d, float* d_Kcache, float* d_Vcache, const float* d_k, const float* d_v, int pos0, int n,
                        void* stream = nullptr);

// Causal attention for nq query tokens at absolute positions pos0 .. pos0+nq-1 over cache positions 0..pos,
// GQA (head h uses kv head h / groups), scores scaled by 1/sqrt(head_dim), softmax in FP32, then multiplied by
// sigmoid(gate) (the layer's output gate). d_q is [nq][heads][head_dim] AFTER norm and RoPE; the cache must already
// contain positions up to pos0+nq-1. d_out is [nq][heads*head_dim], the input of o_proj.
int attention_core_cuda(const AttentionDims& d, const float* d_q, const float* d_gate, const float* d_Kcache, const float* d_Vcache,
                        int pos0, int nq, float* d_out, void* stream = nullptr);

// Device-position variants (CUDA-graph decode): token t sits at position d_pos[t] (device array). The attention kernel reads d_pos[0] as pos0
// (nq tokens at consecutive positions); smem_positions fixes the shared-memory score slots (>= the longest context it will see, <= kAttentionMaxKvLen).
int attention_core_dev_cuda(const AttentionDims& d, const float* d_q, const float* d_gate, const float* d_Kcache, const float* d_Vcache,
                            const int64_t* d_pos, int nq, int smem_positions, float* d_out, void* stream = nullptr);
int attn_kv_append_dev_cuda(const AttentionDims& d, float* d_Kcache, float* d_Vcache, const float* d_k, const float* d_v, const int64_t* d_pos, int n,
                            void* stream = nullptr);

}  // namespace dara::model
