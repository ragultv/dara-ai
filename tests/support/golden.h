#pragma once
// Reader for the golden-tensor container written by tools/make_deltanet_golden.py (and later golden generators).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dara::test {

struct GoldenTensor {
    int dtype = 0;  // 0 = f32, 1 = bf16 raw
    std::vector<uint64_t> dims;
    std::vector<float> f;  // always decoded to float
    size_t numel() const { size_t n = 1; for (auto d : dims) n *= d; return n; }
};

inline std::map<std::string, GoldenTensor> load_golden(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open golden file " + path);
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "DGLD1\0\0\0", 8) != 0) throw std::runtime_error("bad golden magic");
    uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), 4);
    std::map<std::string, GoldenTensor> out;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t nl = 0, dtype = 0, nd = 0;
        in.read(reinterpret_cast<char*>(&nl), 4);
        std::string name(nl, '\0');
        in.read(name.data(), nl);
        in.read(reinterpret_cast<char*>(&dtype), 4);
        in.read(reinterpret_cast<char*>(&nd), 4);
        GoldenTensor t;
        t.dtype = static_cast<int>(dtype);
        t.dims.resize(nd);
        in.read(reinterpret_cast<char*>(t.dims.data()), static_cast<std::streamsize>(nd * 8));
        const size_t n = t.numel();
        t.f.resize(n);
        if (dtype == 0) {
            in.read(reinterpret_cast<char*>(t.f.data()), static_cast<std::streamsize>(n * 4));
        } else {
            std::vector<uint16_t> raw(n);
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(n * 2));
            for (size_t k = 0; k < n; ++k) {
                uint32_t u = static_cast<uint32_t>(raw[k]) << 16;
                std::memcpy(&t.f[k], &u, 4);
            }
        }
        if (!in) throw std::runtime_error("truncated golden file at tensor " + name);
        out.emplace(std::move(name), std::move(t));
    }
    return out;
}

}  // namespace dara::test
