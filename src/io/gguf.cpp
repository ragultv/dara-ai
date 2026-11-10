#include "io/gguf.h"

#include <fstream>
#include <stdexcept>

namespace dara::io {
namespace {

struct Reader {
    std::ifstream f;
    uint64_t size = 0;
    template <class T> T get() {
        T v{};
        f.read(reinterpret_cast<char*>(&v), sizeof(T));
        if (!f) throw std::runtime_error("gguf: unexpected end of file");
        return v;
    }
    std::string str() {
        auto n = get<uint64_t>();
        if (n > size) throw std::runtime_error("gguf: string length exceeds file size");
        std::string s(n, '\0');
        f.read(s.data(), static_cast<std::streamsize>(n));
        if (!f) throw std::runtime_error("gguf: unexpected end of file");
        return s;
    }
};

size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: return 0;
    }
}

std::string read_value(Reader& r, uint32_t t) {
    switch (t) {
        case 0: return std::to_string(r.get<uint8_t>());
        case 1: return std::to_string(r.get<int8_t>());
        case 2: return std::to_string(r.get<uint16_t>());
        case 3: return std::to_string(r.get<int16_t>());
        case 4: return std::to_string(r.get<uint32_t>());
        case 5: return std::to_string(r.get<int32_t>());
        case 6: return std::to_string(r.get<float>());
        case 7: return r.get<uint8_t>() ? "true" : "false";
        case 8: return r.str();
        case 10: return std::to_string(r.get<uint64_t>());
        case 11: return std::to_string(r.get<int64_t>());
        case 12: return std::to_string(r.get<double>());
        case 9: {
            auto et = r.get<uint32_t>();
            auto n = r.get<uint64_t>();
            if (et == 8) {
                for (uint64_t i = 0; i < n; ++i) r.str();
            } else if (et == 9) {
                throw std::runtime_error("gguf: nested arrays unsupported");
            } else {
                auto sz = scalar_size(et);
                if (!sz) throw std::runtime_error("gguf: bad array element type");
                if (n > r.size / sz) throw std::runtime_error("gguf: array exceeds file size");
                r.f.seekg(static_cast<std::streamoff>(n * sz), std::ios::cur);
                if (!r.f) throw std::runtime_error("gguf: unexpected end of file");
            }
            return "[array of " + std::to_string(n) + "]";
        }
        default: throw std::runtime_error("gguf: unknown value type");
    }
}

// Reads an array under "tokenizer.*" and keeps its contents (strings, or integers of any width).
std::string collect_array(Reader& r, GgufFile& g, const std::string& key) {
    const auto et = r.get<uint32_t>();
    const auto n = r.get<uint64_t>();
    if (n > r.size) throw std::runtime_error("gguf: implausible array length for " + key);
    if (et == 8) {
        auto& v = g.string_arrays[key];
        v.reserve(n);
        for (uint64_t i = 0; i < n; ++i) v.push_back(r.str());
    } else if (et == 9) {
        throw std::runtime_error("gguf: nested arrays unsupported");
    } else {
        auto& v = g.int_arrays[key];
        v.reserve(n);
        for (uint64_t i = 0; i < n; ++i) {
            switch (et) {
                case 0: v.push_back(r.get<uint8_t>()); break;
                case 1: v.push_back(r.get<int8_t>()); break;
                case 2: v.push_back(r.get<uint16_t>()); break;
                case 3: v.push_back(r.get<int16_t>()); break;
                case 4: v.push_back(r.get<uint32_t>()); break;
                case 5: v.push_back(r.get<int32_t>()); break;
                case 7: v.push_back(r.get<uint8_t>() ? 1 : 0); break;  // bool
                case 6: r.get<float>(); break;  // e.g. token scores: read and drop
                case 10: v.push_back(static_cast<int64_t>(r.get<uint64_t>())); break;
                case 11: v.push_back(r.get<int64_t>()); break;
                case 12: r.get<double>(); break;
                default: throw std::runtime_error("gguf: bad array element type");
            }
        }
    }
    return "[array of " + std::to_string(n) + "]";
}

}  // namespace

GgufFile read_gguf_header(const std::string& path) {
    Reader r;
    r.f.open(path, std::ios::binary);
    if (!r.f) throw std::runtime_error("gguf: cannot open " + path);
    r.f.seekg(0, std::ios::end);
    r.size = static_cast<uint64_t>(r.f.tellg());
    r.f.seekg(0);

    char magic[4];
    r.f.read(magic, 4);
    if (!r.f || std::string(magic, 4) != "GGUF") throw std::runtime_error("gguf: bad magic");

    GgufFile g;
    g.version = r.get<uint32_t>();
    if (g.version < 2 || g.version > 3) throw std::runtime_error("gguf: unsupported version");
    g.tensor_count = r.get<uint64_t>();
    auto kv_count = r.get<uint64_t>();
    if (g.tensor_count > r.size || kv_count > r.size) throw std::runtime_error("gguf: implausible counts");

    for (uint64_t i = 0; i < kv_count; ++i) {
        auto key = r.str();
        auto type = r.get<uint32_t>();
        bool keep_array = type == 9 && key.rfind("tokenizer.", 0) == 0;
        if (type == 9 && !keep_array) {
            // Small numeric / bool arrays (per-layer feed-forward sizes, sliding-window patterns, ...) are kept in full too.
            const auto here = r.f.tellg();
            const auto et = r.get<uint32_t>();
            const auto n = r.get<uint64_t>();
            r.f.seekg(here);
            keep_array = et != 8 && et != 9 && n <= 4096;
        }
        if (keep_array) {
            auto desc = collect_array(r, g, key);
            g.metadata.emplace_back(std::move(key), std::move(desc));
            continue;
        }
        auto val = read_value(r, type);
        if (key == "general.alignment" && type == 4) g.alignment = std::stoull(val);
        g.metadata.emplace_back(std::move(key), std::move(val));
    }
    for (uint64_t i = 0; i < g.tensor_count; ++i) {
        GgufTensor t;
        t.name = r.str();
        auto nd = r.get<uint32_t>();
        if (nd > 8) throw std::runtime_error("gguf: too many dims");
        for (uint32_t d = 0; d < nd; ++d) t.dims.push_back(r.get<uint64_t>());
        t.type = r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        g.tensors.push_back(std::move(t));
    }
    if (g.alignment == 0) throw std::runtime_error("gguf: zero alignment");
    auto pos = static_cast<uint64_t>(r.f.tellg());
    g.data_offset = (pos + g.alignment - 1) / g.alignment * g.alignment;
    return g;
}

}  // namespace dara::io
