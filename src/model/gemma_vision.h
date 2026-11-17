// Gemma 4 vision tower (Nano image input), from a llama.cpp-format mmproj GGUF ("clip", projector gemma4v). Same arithmetic as the
// reference (transformers Gemma4VisionModel + Gemma4MultimodalEmbedder): aspect-preserving bicubic resize to at most 280 soft tokens,
// 16x16 patches, learned x/y position table, 16 encoder layers (2D RoPE, bidirectional attention with scale 1, clamped linears, GeGLU),
// 3x3 average pooling, x sqrt(hidden), unscaled RMSNorm and the projection into the text model's embedding space.
// Matrices run as BF16 (exact for the BF16-origin weights the F16 file carries) with FP32 activations.
#pragma once

#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "io/image.h"
#include "model/device_buffer.h"
#include "model/layer_cuda.h"

namespace dara::model {

struct VisionInput {
    int n = 0;                    // patches
    int grid_w = 0, grid_h = 0;   // patch grid
    std::vector<float> patches;   // [n][patch*patch*3], pixel values in [0, 1], patch-major (y, x, channel) as the reference processor
    std::vector<int> px, py;      // patch grid position of each patch
    int soft_tokens() const { return n / 9; }
};

class GemmaVision {
public:
    explicit GemmaVision(const dara::io::GgufCheckpoint& mmproj, int max_soft_tokens = 280);
    VisionInput preprocess(const dara::io::RgbImage& img) const;
    // Soft tokens for the text model: [soft_tokens][proj_dim] (row-major, pooled grid order).
    std::vector<float> encode(const VisionInput& in);
    int proj_dim() const { return proj_dim_; }
    double last_seconds() const { return last_seconds_; }

private:
    struct Linear { DeviceMatrix w; float in_lo, in_hi, out_lo, out_hi; };
    struct Layer {
        Linear q, k, v, o, gate, up, down;
        DevBuf<float> ln1, attn_post, ln2, ffn_post, q_norm, k_norm;
    };
    void linear(const Linear& l, const float* x, float* tmp, float* y, int n);
    int hidden_ = 768, heads_ = 12, hd_ = 64, inter_ = 3072, layers_n_ = 16, patch_ = 16, pool_ = 3, max_soft_ = 280, proj_dim_ = 1536, pos_size_ = 10240;
    float eps_ = 1e-6f, theta_ = 100.0f;
    DeviceMatrix patch_w_, proj_w_;
    std::vector<float> pos_table_;   // [2][pos_size][hidden]
    std::vector<Layer> layers_;
    DevBuf<float> ones_;
    double last_seconds_ = 0;
};

}  // namespace dara::model
