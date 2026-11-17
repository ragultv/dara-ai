// Qwen3.5 vision tower (Core 4B / Pro 9B image input), from a llama.cpp-format mmproj GGUF ("clip", projector qwen3vl_merger). Same
// arithmetic as the reference (transformers Qwen2-VL image processor + Qwen3_5VisionModel):
//   preprocessing: smart_resize to multiples of 32 px within [min, max] pixels (max set by the image token budget), Pillow bicubic, rescale,
//                  normalize (mean 0.5, std 0.5), 16x16 patches with the frame duplicated over the 2-frame temporal patch, 2x2 merge-block order;
//   encoder:       Conv3d patch embedding (+ bias), learned 48x48 position grid bilinearly resampled to the patch grid, depth x [LayerNorm,
//                  fused qkv (+ bias), 2D rotate-half RoPE (theta 10000), bidirectional attention (1/sqrt(hd)), proj; LayerNorm, MLP with
//                  gelu-tanh], merger: LayerNorm, 2x2 patches concatenated, fc1 + exact GELU + fc2 to the text width.
// Matrices run as BF16 (exact for the BF16-origin weights the F16 file carries) with FP32 activations.
#pragma once

#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "io/image.h"
#include "model/device_buffer.h"
#include "model/layer_cuda.h"

namespace dara::model {

struct QwenVisionInput {
    int n = 0;                    // patches (merge-block order)
    int grid_h = 0, grid_w = 0;   // patch grid (multiples of the merge size)
    std::vector<float> patches;   // [n][3 * 2 * 16 * 16] normalized pixels, (channel, frame, y, x)
    std::vector<int> row, col;    // patch grid position of each patch
    int tokens() const { return n / 4; }
    int merged_h() const { return grid_h / 2; }
    int merged_w() const { return grid_w / 2; }
};

class QwenVision {
public:
    // max_tokens: image token budget (merged tokens); images are resized to at most max_tokens * 32 * 32 pixels (and at least 64 tokens).
    explicit QwenVision(const dara::io::GgufCheckpoint& mmproj, int max_tokens = 1024);
    QwenVisionInput preprocess(const dara::io::RgbImage& img) const;
    std::vector<float> encode(const QwenVisionInput& in);   // [tokens][out_dim], merged-grid row-major
    int out_dim() const { return out_dim_; }
    double last_seconds() const { return last_seconds_; }

private:
    struct Lin { DeviceMatrix w; DevBuf<float> b; };
    struct Block { Lin qkv, proj, fc1, fc2; DevBuf<float> ln1_w, ln1_b, ln2_w, ln2_b; };
    void linear(const Lin& l, const float* x, size_t ldx, float* y, int n);
    int hidden_ = 1024, heads_ = 16, hd_ = 64, inter_ = 4096, inter_pad_ = 4096, depth_ = 24, patch_ = 16, merge_ = 2, temporal_ = 2, out_dim_ = 2560;
    int grid_side_ = 48, max_tokens_ = 1024;
    float eps_ = 1e-6f;
    Lin patch_lin_, mm0_, mm2_;
    DevBuf<float> post_w_, post_b_;
    std::vector<float> pos_table_;   // [grid_side^2][hidden]
    std::vector<Block> blocks_;
    double last_seconds_ = 0;
};

}  // namespace dara::model
