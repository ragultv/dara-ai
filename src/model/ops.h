#pragma once
// Elementwise / normalization operators of Qwen3.5, CPU FP32 references (see the pinned reference implementation,
// transformers Qwen3_5RMSNorm and Qwen3_5TextRotaryEmbedding / apply_rotary_pos_emb).
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dara::model {

// ---- RMSNorm ----------------------------------------------------------------------------------------
// y = x * rsqrt(mean(x^2) + eps) * scale        (scale is the EFFECTIVE per-element multiplier)
//
// WEIGHT CONVENTION. The checkpoint stores the four main norms (input, post-attention, q_norm, k_norm, final) as
// w with y = x_hat * (1 + w). The GGUF has the +1 folded in (audited on all layers), so a GGUF-fed runtime passes
// the GGUF tensor here directly as `scale`. The DeltaNet gated norm is stored WITHOUT the +1 in both formats and is
// also passed as-is. Never apply (1 + w) to a GGUF tensor. fold_plus_one() exists only to convert checkpoint
// weights (e.g. for tests against the reference).
void rmsnorm(const float* x, const float* scale, int n, float eps, float* y);
float rmsnorm_inv_rms(const float* x, int n, float eps);  // rsqrt(mean(x^2) + eps), the intermediate statistic
std::vector<float> fold_plus_one(const std::vector<float>& w_checkpoint);

// ---- RoPE (NeoX / rotate_half style, partial) --------------------------------------------------------
// Rotates the first rot_dim dimensions of each head: for i < rot_dim/2, with angle = pos * inv_freq[i],
//   x[i]            <- x[i] * cos - x[i + rot_dim/2] * sin
//   x[i + rot_dim/2] <- x[i + rot_dim/2] * cos + x[i] * sin
// The remaining head_dim - rot_dim dimensions pass through unchanged.
// Qwen3.5-4B: head_dim 256, partial_rotary_factor 0.25 -> rot_dim 64, theta 1e7, rope_type "default" (no
// scaling). M-RoPE position streams (T, H, W) are identical for text, so this is plain 1-D RoPE.
//
// PRECISION. The reference computes angle = float32(pos) * float32(inv_freq) and cos/sin in float32. At long
// positions that fp32 angle carries a phase error of order pos * 6e-8 * freq. To match the reference, this
// implementation reproduces that arithmetic exactly; rope_angle_fp32 and rope_angle_exact expose both so the
// divergence can be measured.
std::vector<float> rope_inv_freq(int rot_dim, double theta);        // reference pipeline: every step in float32
std::vector<float> rope_inv_freq_exact(int rot_dim, double theta);  // correctly rounded; NOT what the reference uses
inline float rope_angle_fp32(int64_t pos, float inv_freq) { return static_cast<float>(pos) * inv_freq; }
inline double rope_angle_exact(int64_t pos, double inv_freq) { return static_cast<double>(pos) * inv_freq; }
void rope_cos_sin(int64_t pos, const std::vector<float>& inv_freq, float* cos_out, float* sin_out);  // [rot_dim/2] each
// x: [heads][head_dim]; cos/sin: [rot_dim/2] for this position.
void rope_apply(float* x, int heads, int head_dim, int rot_dim, const float* cos, const float* sin);

}  // namespace dara::model
