#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dara::core {

enum class LayerKind {
    Attention,       // full attention: keeps KV for the whole context
    LocalAttention,  // sliding window: bounded KV
    DeltaNet         // recurrent: fixed-size state, no KV
};

struct LayerSpec {
    LayerKind kind = LayerKind::Attention;
    uint64_t stored_bytes = 0;  // bytes in the checkpoint as published
    uint64_t weight_bytes = 0;  // bytes planned for the runtime format (after quantization)
};

// Machine-readable description of one exact checkpoint. Backends configure themselves from this
// and reject anything that does not match.
struct ModelManifest {
    std::string name;
    std::string family;     // "qwen35", "gemma4"
    std::string revision;   // exact source commit
    uint64_t native_context = 0;
    std::vector<LayerSpec> layers;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint64_t recurrent_state_bytes = 0;  // whole model, per session
    uint64_t embed_output_bytes = 0;     // kept on GPU
    uint64_t vision_bytes = 0;
    uint64_t host_resident_bytes = 0;    // e.g. Gemma per-layer embeddings
    uint64_t mtp_bytes = 0;              // draft head weights, 0 if absent

    uint32_t n_layers() const { return static_cast<uint32_t>(layers.size()); }
    uint32_t kv_layers() const;          // full-attention layers only
    uint64_t total_layer_bytes() const;
};
using ModelSpec = ModelManifest;

// Planned bytes = stored bytes * scale. Scales are estimates until a GGUF header gives exact sizes.
struct QuantProfile {
    double layer_scale = 1.0;   // e.g. ~4.85 bits/weight over 16-bit storage = 0.303
    double embed_scale = 1.0;
    double vision_scale = 1.0;  // vision towers are usually kept at 16 or 8 bits
};

// A real 2-D weight [rows = outputs, cols = inputs] taken from the checkpoint's tensor headers.
struct MatrixShape {
    std::string name;
    uint64_t rows = 0, cols = 0;
    std::string dtype;  // as stored, e.g. "BF16"
};
// Throws std::runtime_error if the manifest has no "matrices" section.
std::vector<MatrixShape> load_matrix_shapes(const std::string& path);

// Throws std::runtime_error on missing or inconsistent data.
ModelManifest load_manifest(const std::string& path, const QuantProfile& q);

}  // namespace dara::core
