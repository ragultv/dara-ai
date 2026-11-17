// Dara's Qwen3.5 vision path vs the transformers reference (tools/vision/qwen35v_ref.py): preprocessing (normalized patches in merge-block
// order) and the image tokens that go into the text model.
// Usage: qwen_vision_parity --mmproj <mmproj.gguf> --image <file> --ref <prefix> [--max-tokens 1024]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "io/image.h"
#include "model/qwen_vision.h"

template <class T> std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + path);
    const auto n = static_cast<size_t>(f.tellg());
    std::vector<T> v(n / sizeof(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
    return v;
}

int main(int argc, char** argv) {
    std::string mmproj, image, ref;
    int max_tokens = 1024;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--mmproj") mmproj = argv[i + 1];
        else if (a == "--image") image = argv[i + 1];
        else if (a == "--ref") ref = argv[i + 1];
        else if (a == "--max-tokens") max_tokens = std::atoi(argv[i + 1]);
    }
    if (mmproj.empty() || image.empty() || ref.empty()) { std::fprintf(stderr, "usage: qwen_vision_parity --mmproj <gguf> --image <file> --ref <prefix>\n"); return 2; }
    try {
        dara::io::GgufCheckpoint g(mmproj);
        dara::model::QwenVision vis(g, max_tokens);
        const auto img = dara::io::load_image_rgb(image);
        const auto in = vis.preprocess(img);
        std::printf("image %dx%d -> patch grid %dx%d (%d patches, %d tokens, merged %dx%d)\n", img.width, img.height, in.grid_w, in.grid_h, in.n, in.tokens(),
                    in.merged_w(), in.merged_h());
        const auto rp = read_bin<float>(ref + ".patches.bin");
        if (rp.size() != in.patches.size()) { std::printf("PATCHES: size %zu vs reference %zu\n", in.patches.size(), rp.size()); return 1; }
        double pmax = 0; size_t pdiff = 0;
        for (size_t i = 0; i < rp.size(); ++i) { const double d = std::fabs(rp[i] - in.patches[i]); pmax = std::max(pmax, d); pdiff += d > 0; }
        std::printf("preprocessing: %zu of %zu values differ (max |diff| %.7f)\n", pdiff, rp.size(), pmax);
        const auto f = vis.encode(in);
        const double first = vis.last_seconds();
        const auto f2 = vis.encode(in);
        const auto rf = read_bin<float>(ref + ".feats.bin");
        if (rf.size() != f.size()) { std::printf("FEATURES: size %zu vs reference %zu\n", f.size(), rf.size()); return 1; }
        const int D = vis.out_dim(), M = static_cast<int>(f.size() / D);
        double dot = 0, na = 0, nb = 0, maxd = 0, maxr = 0, worst = 1;
        for (int r = 0; r < M; ++r) {
            double d1 = 0, a1 = 0, b1 = 0;
            for (int c = 0; c < D; ++c) {
                const double x = f[static_cast<size_t>(r) * D + c], y = rf[static_cast<size_t>(r) * D + c];
                d1 += x * y; a1 += x * x; b1 += y * y;
                maxd = std::max(maxd, std::fabs(x - y)); maxr = std::max(maxr, std::fabs(y));
            }
            dot += d1; na += a1; nb += b1;
            worst = std::min(worst, d1 / std::sqrt(a1 * b1));
        }
        std::printf("image tokens [%d][%d]: cosine %.6f overall, worst row %.6f | max |diff| %.5f (max |ref| %.3f) | repeat run %s\n", M, D, dot / std::sqrt(na * nb),
                    worst, maxd, maxr, f == f2 ? "bitwise identical" : "DIFFERS");
        std::printf("encode: %.3f s first, %.3f s second\n", first, vis.last_seconds());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
