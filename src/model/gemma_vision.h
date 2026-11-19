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
#include "model/vision_stream.h"

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
    // stream_weights: 1 = the matrices are streamed from the (memory-mapped) mmproj file while an image is encoded (a few MB of VRAM between images),
    // 0 = all resident, -1 = DARA_VISION_STREAM (default 1). The output is bit-identical. `mmproj` must outlive the encoder.
    explicit GemmaVision(const dara::io::GgufCheckpoint& mmproj, int max_soft_tokens = 280, int stream_weights = -1);
    VisionInput preprocess(const dara::io::RgbImage& img) const;
    // Soft tokens for the text model: [soft_tokens][proj_dim] (row-major, pooled grid order).
    std::vector<float> encode(const VisionInput& in);
    int proj_dim() const { return proj_dim_; }
    double last_seconds() const { return last_seconds_; }
    bool streaming() const { return streamed_; }
    // Device memory encode() allocates for an image of n_patches patches (each buffer rounded up to 256 bytes), including the weight-stream buffers
    // when streaming. Pinned host memory of the stream: WeightStream::host_bytes().
    size_t scratch_bytes(int n_patches) const;
    size_t resident_bytes() const { return resident_bytes_; }   // VRAM kept between images

private:
    struct Linear { DeviceMatrix w; StreamRef s; float in_lo, in_hi, out_lo, out_hi; };
    struct Layer {
        Linear q, k, v, o, gate, up, down;
        DevBuf<float> ln1, attn_post, ln2, ffn_post, q_norm, k_norm;
    };
    void linear(WeightStream* ws, const Linear& l, const float* x, float* tmp, float* y, int n);
    int hidden_ = 768, heads_ = 12, hd_ = 64, inter_ = 3072, layers_n_ = 16, patch_ = 16, pool_ = 3, max_soft_ = 280, proj_dim_ = 1536, pos_size_ = 10240;
    float eps_ = 1e-6f, theta_ = 100.0f;
    DeviceMatrix patch_w_;
    Linear proj_;                    // multimodal embedder projection (streamed or resident)
    const uint8_t* pos_map_ = nullptr;   // [2][pos_size][hidden] in the mapped file (F16 or F32): rows are read as needed
    uint32_t pos_type_ = 0;
    std::vector<Layer> layers_;
    bool streamed_ = false;
    std::vector<ChunkReq> plan_;     // every chunk of one image in compute order (streaming)
    size_t resident_bytes_ = 0;
    void build_plan();
    void matmul(WeightStream* ws, const Linear& l, const float* x, size_t ldx, float* y, int n);
    DevBuf<float> ones_;
    double last_seconds_ = 0;
};

}  // namespace dara::model
