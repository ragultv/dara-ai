// Dara Flash runtime parity: runs the C++ drafter over the first n positions of one training sequence (true target features + next tokens) and
// writes its final predicted feature (float32 [hidden]) plus the drafted token, for tools/flash/parity_check.py to compare with PyTorch.
// Usage: flash_parity_test --gguf <model.gguf> --drafter <file.dflash> --data <dir> --seq I --n N --out <file.bin>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf_checkpoint.h"
#include "kernels/half.h"
#include "model/flash_drafter.h"
#include "model/qwen_model.h"

using namespace dara::model;

int main(int argc, char** argv) {
    std::string gguf_path, drafter, data, out;
    int seq_i = 0, n = 64;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--gguf") gguf_path = argv[i + 1];
        else if (a == "--drafter") drafter = argv[i + 1];
        else if (a == "--data") data = argv[i + 1];
        else if (a == "--seq") seq_i = std::atoi(argv[i + 1]);
        else if (a == "--n") n = std::atoi(argv[i + 1]);
        else if (a == "--out") out = argv[i + 1];
    }
    try {
        dara::io::GgufCheckpoint g(gguf_path);
        const QwenConfig cfg = QwenConfig::from_gguf(g);
        GgufWeights w(g, cfg);
        const int H = cfg.hidden;
        std::ifstream sf(data + "/seqs.bin", std::ios::binary);
        std::vector<int32_t> toks;
        for (int s = 0; s <= seq_i; ++s) {
            int32_t hdr[2];
            sf.read(reinterpret_cast<char*>(hdr), 8);
            toks.resize(hdr[0]);
            sf.read(reinterpret_cast<char*>(toks.data()), hdr[0] * 4);
        }
        std::ifstream ix(data + "/index.jsonl");
        std::string line;
        for (int s = 0; s <= seq_i; ++s) std::getline(ix, line);
        const size_t off = nlohmann::json::parse(line)["feat_offset"].get<size_t>();
        if (n + 1 > static_cast<int>(toks.size())) n = static_cast<int>(toks.size()) - 1;
        std::ifstream ff(data + "/feats.bin", std::ios::binary);
        ff.seekg(static_cast<std::streamoff>(off * 2));
        std::vector<uint16_t> h(static_cast<size_t>(n) * H);
        ff.read(reinterpret_cast<char*>(h.data()), static_cast<std::streamsize>(h.size() * 2));
        std::vector<float> f(h.size());
        for (size_t i = 0; i < h.size(); ++i) f[i] = dara::kernels::half_to_float(h[i]);
        FlashDrafter d(drafter, g, [&](int32_t t) { return w.embed_row(t); }, 4096);
        std::vector<float> y; int32_t tok = 0;
        d.extend(f.data(), toks.data() + 1, n, y, tok);
        std::ofstream o(out, std::ios::binary);
        o.write(reinterpret_cast<const char*>(&tok), 4);
        o.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * 4));
        // also drafting continuity: extend one more position, split into two calls (n-1 then 1) must give the same result
        d.rewind(0);
        d.extend(f.data(), toks.data() + 1, n - 1, y, tok);
        d.extend(f.data() + static_cast<size_t>(n - 1) * H, toks.data() + n, 1, y, tok);
        o.write(reinterpret_cast<const char*>(&tok), 4);
        o.write(reinterpret_cast<const char*>(y.data()), static_cast<std::streamsize>(y.size() * 4));
        for (int nn : {1, 3, 8}) {   // timing: extend of nn positions, repeated
            cudaDeviceSynchronize();
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 50; ++r) { d.rewind(static_cast<size_t>(n - nn)); d.extend(f.data(), toks.data() + 1, nn, y, tok); }
            std::printf("extend n=%d: %.3f ms\n", nn, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50);
        }
        std::printf("n=%d token %d\n", n, tok);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
