#include <cstdio>
#include <string>

#include "io/gguf.h"
#include "io/safetensors.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: dara_inspect <file.gguf|file.safetensors>\n");
        return 2;
    }
    std::string p = argv[1];
    try {
        if (p.size() > 5 && p.substr(p.size() - 5) == ".gguf") {
            auto g = dara::io::read_gguf_header(p);
            std::printf("GGUF v%u, %llu tensors, data at %llu\n", g.version,
                        (unsigned long long)g.tensor_count, (unsigned long long)g.data_offset);
            for (auto& kv : g.metadata) std::printf("  %s = %.80s\n", kv.first.c_str(), kv.second.c_str());
        } else {
            auto s = dara::io::read_safetensors_header(p);
            std::printf("safetensors, %zu tensors\n", s.tensors.size());
            for (auto& t : s.tensors)
                std::printf("  %s %s [%llu bytes]\n", t.name.c_str(), t.dtype.c_str(),
                            (unsigned long long)(t.end - t.begin));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
