#pragma once
// Prefill attention for FP32 K/V caches as GEMM -> masked softmax -> GEMM (attn_prefill.cu). For chunks of many tokens; decode keeps split-KV attention.
#include <cstddef>

#include "model/attn_splitkv.h"   // AttnFmt

namespace dara::model {

constexpr int kPrefillAttnMinTokens = 48;

// Scratch floats needed for a chunk of n tokens starting at absolute position pos0 (window 0 = full causal attention).
size_t attn_prefill_scratch_floats(int n, int pos0, int window, int heads);

// out[t][h][:] = softmax(scale * q_t,h . K[k] over keys k in [max(0, p - window + 1), p], p = pos0 + t) V   (no gate)
// q: [n][heads][hd]; K, V: caches [position][kv_heads][hd] already holding positions < pos0 + n; out: [n][heads][hd]. hd % 128 == 0.
// General form: q8 = INT8 cache (K/V int8 codes, Ksc/Vsc per-32 scales as in attn_splitkv.h), otherwise FP32 (Ksc/Vsc ignored). gate: [n][heads*hd] or
// null; when given, out *= sigmoid(gate) (Qwen). Queries are processed in sub-chunks so the scratch stays bounded.
int attn_prefill_cuda(AttnFmt fmt, const float* q, const float* gate, const void* K, const float* Ksc, const void* V, const float* Vsc, int n, int pos0, int heads,
                      int kv_heads, int hd, int window, float scale, float* scratch, size_t scratch_floats, float* out, void* stream = nullptr);

int attn_prefill_f32_cuda(const float* q, const float* K, const float* V, int n, int pos0, int heads, int kv_heads, int hd, int window, float scale, float* scratch,
                          size_t scratch_floats, float* out, void* stream = nullptr);

}  // namespace dara::model
