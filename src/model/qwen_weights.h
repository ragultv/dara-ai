#pragma once
// Where a model's weights come from. The runtime (QwenModel) asks for one layer at a time -- as a contiguous host image it can
// keep in RAM or upload to VRAM -- plus the embedding rows, final norm scale and output head. Dara supports GGUF models only:
//   GgufWeights : a quantized GGUF (Q4_K_M etc.): GGUF V-head order (de-interleaved), norm tensors with +1 already folded
//                 in, ssm_a = -exp(A_log), Q4_K/Q5_K/Q6_K/Q8_0 matrices consumed by the fused dequantize+GEMV kernels.
// The interface remains so a test double can stand in for a file.
#include <cstdint>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "model/layer_cuda.h"
#include "model/packed_layer.h"

namespace dara::model {

struct QwenConfig;

class IModelWeights {
public:
    virtual ~IModelWeights() = default;
    virtual const char* format() const = 0;
    virtual PackedLayer load_layer_packed(int layer) const = 0;    // one layer as a contiguous host image (read from disk once)
    virtual size_t layer_matrix_bytes(int layer) const = 0;        // size of that image, known without reading it
    virtual size_t head_bytes() const = 0;                         // size of the output head matrix
    virtual size_t embedding_bytes() const = 0;                    // size of the embedding table kept in host RAM
    virtual std::vector<float> embed_row(int32_t token) const = 0; // [hidden]; served from host RAM after the first call
    virtual std::vector<float> final_norm_scale() const = 0;       // effective multiplier (+1 convention resolved)
    virtual DeviceMatrix load_output_head() const = 0;             // [vocab][hidden], uploaded by the caller's request
};

class GgufWeights : public IModelWeights {
public:
    GgufWeights(const dara::io::GgufCheckpoint& g, const QwenConfig& cfg);
    const char* format() const override { return "gguf"; }
    PackedLayer load_layer_packed(int layer) const override;
    size_t layer_matrix_bytes(int layer) const override;
    size_t head_bytes() const override;
    size_t embedding_bytes() const override;
    std::vector<float> embed_row(int32_t token) const override;
    std::vector<float> final_norm_scale() const override;
    DeviceMatrix load_output_head() const override;

private:
    const dara::io::GgufCheckpoint& g_;
    const QwenConfig& cfg_;
    mutable std::vector<uint8_t> embd_;      // token_embd, read once and kept in host RAM for row gathers
    mutable std::vector<float> final_norm_;
};

}  // namespace dara::model
