// Dumps one GGUF tensor, dequantised, as fp16 rows (for Dara Flash training: the target's token embedding / output head).
// Usage: gguf_dump_tensor --gguf <model.gguf> --tensor <name> --out <file.f16>     (writes [rows][cols] fp16; prints rows and cols)
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "kernels/half.h"
#include "kernels/quant.h"
#include "model/qwen_weights.h"

namespace dara::model {
static dara::kernels::QuantType gguf_qtype(uint32_t type, const std::string& name) {   // GGUF tensor type ids (as qwen_weights.cu)
    switch (type) {
        case 8: return dara::kernels::QuantType::Q8_0;
        case 12: return dara::kernels::QuantType::Q4_K;
        case 13: return dara::kernels::QuantType::Q5_K;
        case 14: return dara::kernels::QuantType::Q6_K;
    }
    throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(type) + " for " + name);
}
}

int main(int argc, char** argv) {
    std::string gguf_path, name, out;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gguf") gguf_path = argv[i + 1];
        else if (a == "--tensor") name = argv[i + 1];
        else if (a == "--out") out = argv[i + 1];
    }
    if (gguf_path.empty() || name.empty() || out.empty()) { std::fprintf(stderr, "usage: gguf_dump_tensor --gguf <model.gguf> --tensor <name> --out <file>\n"); return 2; }
    try {
        dara::io::GgufCheckpoint g(gguf_path);
        const auto& in = g.info(name);
        const size_t rows = in.rows(), cols = in.cols();
        const auto qt = dara::model::gguf_qtype(in.type, name);
        const auto& f = dara::kernels::quant_format(qt);
        const size_t row_bytes = cols / f.block_weights * f.block_bytes;
        std::ofstream o(out, std::ios::binary);
        std::vector<float> row(cols);
        std::vector<uint16_t> h(cols);
        const size_t chunk = 4096;
        for (size_t r0 = 0; r0 < rows; r0 += chunk) {
            const size_t n = std::min(chunk, rows - r0);
            const auto bytes = g.read_rows(name, r0, n);
            for (size_t r = 0; r < n; ++r) {
                f.dequant_f32(bytes.data() + r * row_bytes, cols / f.block_weights, row.data());
                for (size_t i = 0; i < cols; ++i) h[i] = dara::kernels::float_to_half(row[i]);
                o.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(cols * 2));
            }
        }
        std::printf("%s: %zu rows x %zu cols -> %s\n", name.c_str(), rows, cols, out.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
