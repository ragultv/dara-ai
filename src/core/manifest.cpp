#include "core/manifest.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace dara::core {

uint32_t ModelManifest::kv_layers() const {
    uint32_t n = 0;
    for (auto& l : layers) n += l.kind == LayerKind::Attention;
    return n;
}

uint64_t ModelManifest::total_layer_bytes() const {
    uint64_t t = 0;
    for (auto& l : layers) t += l.weight_bytes;
    return t;
}

static uint64_t scaled(uint64_t v, double s) { return static_cast<uint64_t>(static_cast<double>(v) * s); }

std::vector<MatrixShape> load_matrix_shapes(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("manifest: cannot open " + path);
    try {
        nlohmann::json j;
        f >> j;
        std::vector<MatrixShape> out;
        for (auto& m : j.at("matrices")) {
            MatrixShape s;
            s.name = m.at("name").get<std::string>();
            s.rows = m.at("rows").get<uint64_t>();
            s.cols = m.at("cols").get<uint64_t>();
            s.dtype = m.at("dtype").get<std::string>();
            if (!s.rows || !s.cols) throw std::runtime_error("empty matrix " + s.name);
            out.push_back(std::move(s));
        }
        return out;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("manifest: ") + e.what());
    }
}

ModelManifest load_manifest(const std::string& path, const QuantProfile& q) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("manifest: cannot open " + path);
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("manifest: bad json: ") + e.what());
    }
    try {
        ModelManifest m;
        m.name = j.at("name").get<std::string>();
        m.family = j.at("family").get<std::string>();
        m.revision = j.at("source").at("revision").get<std::string>();
        m.native_context = j.at("native_context").get<uint64_t>();
        for (auto& l : j.at("layers")) {
            LayerSpec s;
            auto kind = l.at("kind").get<std::string>();
            if (kind == "attention") s.kind = LayerKind::Attention;
            else if (kind == "local_attention") s.kind = LayerKind::LocalAttention;
            else if (kind == "deltanet") s.kind = LayerKind::DeltaNet;
            else throw std::runtime_error("unknown layer kind " + kind);
            s.stored_bytes = l.at("stored_bytes").get<uint64_t>();
            if (s.stored_bytes == 0) throw std::runtime_error("layer with zero bytes");
            s.weight_bytes = scaled(s.stored_bytes, q.layer_scale);
            m.layers.push_back(s);
        }
        if (m.layers.empty()) throw std::runtime_error("no layers");
        m.kv_heads = j.at("kv_heads").get<uint32_t>();
        m.head_dim = j.at("head_dim").get<uint32_t>();
        m.recurrent_state_bytes = j.at("recurrent_state_bytes").get<uint64_t>();
        m.embed_output_bytes = scaled(j.at("embed_output_bytes").get<uint64_t>(), q.embed_scale);
        m.vision_bytes = scaled(j.at("vision_bytes").get<uint64_t>(), q.vision_scale);
        m.host_resident_bytes = j.at("host_resident_bytes").get<uint64_t>();
        m.mtp_bytes = j.at("mtp").at("stored_bytes").get<uint64_t>();
        return m;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(std::string("manifest: ") + e.what());
    }
}

}  // namespace dara::core
