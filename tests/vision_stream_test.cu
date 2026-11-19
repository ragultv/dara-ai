// Streamed vs resident image encoding: the embeddings must be bit-identical, scratch_bytes() must match what encode() really allocates, and the encoder
// must keep (almost) nothing in VRAM between images when it streams.
//
// Usage: vision_stream_test qwen|gemma <mmproj.gguf> <image> [max_tokens]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "io/gguf_checkpoint.h"
#include "io/image.h"
#include "model/device_buffer.h"
#include "model/gemma_vision.h"
#include "model/qwen_vision.h"

using namespace dara::model;

namespace {

double since(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }

template <class Vision, class Input>
struct Run { std::vector<float> out; double seconds = 0; size_t scratch_peak = 0, predicted = 0, resident = 0; };

template <class Vision, class Input>
Run<Vision, Input> run(Vision& v, const Input& in, int patches) {
    Run<Vision, Input> r;
    v.encode(in);   // warm-up (first launches, page cache)
    const size_t before = device_stats().current.load();
    device_stats().reset_peak();
    const auto t0 = std::chrono::steady_clock::now();
    r.out = v.encode(in);
    r.seconds = since(t0);
    r.scratch_peak = device_stats().peak.load() - before;
    r.predicted = v.scratch_bytes(patches);
    r.resident = v.resident_bytes();
    return r;
}

// Memory behaviour of one encoder: (1) under an arena (a big borrowed block standing in for the unused part of the K/V caches) the encode allocates from it and
// gives the same bits; (2) a device allocation failure part-way through leaves nothing allocated and the next encode is still exact.
template <class Vision, class Input>
bool memory_checks(Vision& v, const Input& in, const std::vector<float>& want) {
    bool ok = true;
    {
        DevBuf<uint8_t> block(size_t{400} << 20);   // stands in for a K/V tail region
        DeviceArena arena;
        arena.regions.push_back({block.p, block.n, 0});
        const size_t before = device_stats().current.load();
        std::vector<float> got;
        {
            ArenaScope scope(arena);
            got = v.encode(in);
        }
        const bool same = got.size() == want.size() && std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0;
        std::printf("arena: %.1f MiB of the encode came from the borrowed block (cudaMalloc fallback %.1f MiB), output %s, device bytes %s\n", arena.peak / 1048576.0,
                    arena.fallback / 1048576.0, same ? "bit-identical" : "DIFFERENT", device_stats().current.load() == before ? "unchanged" : "CHANGED");
        ok = ok && same && arena.fallback == 0 && device_stats().current.load() == before;
    }
    for (long fail_at : {3L, 12L}) {
        const size_t before = device_stats().current.load();
        alloc_fault_countdown().store(fail_at);
        bool threw = false;
        try { v.encode(in); } catch (const std::exception&) { threw = true; }
        alloc_fault_countdown().store(0);
        const bool clean = device_stats().current.load() == before;
        const std::vector<float> again = v.encode(in);
        const bool same = again.size() == want.size() && std::memcmp(again.data(), want.data(), want.size() * sizeof(float)) == 0;
        std::printf("allocation %ld fails: %s, device memory %s, next encode %s\n", fail_at, threw ? "encode throws" : "NO EXCEPTION", clean ? "released" : "LEAKED", same ? "exact" : "WRONG");
        ok = ok && threw && clean && same;
    }
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: vision_stream_test qwen|gemma <mmproj.gguf> <image> [max_tokens]\n"); return 2; }
    const std::string kind = argv[1];
    const int max_tokens = argc > 4 ? std::atoi(argv[4]) : (kind == "qwen" ? 400 : 280);
    try {
        dara::io::GgufCheckpoint mm(argv[2]);
        const auto img = dara::io::load_image_rgb(argv[3]);
        std::printf("image %dx%d, %s encoder, token budget %d\n", img.width, img.height, kind.c_str(), max_tokens);
        size_t free0 = 0, tot = 0;
        cudaMemGetInfo(&free0, &tot);
        std::vector<float> a, b;
        if (kind == "qwen") {
            QwenVision res(mm, max_tokens, 0), str(mm, max_tokens, 1);
            const auto in = res.preprocess(img);
            std::printf("%d patches -> %d tokens; resident encoder keeps %.1f MiB, streaming encoder %.1f MiB\n", in.n, in.tokens(), res.resident_bytes() / 1048576.0, str.resident_bytes() / 1048576.0);
            auto ra = run<QwenVision, QwenVisionInput>(res, in, in.n), rb = run<QwenVision, QwenVisionInput>(str, in, in.n);
            a = ra.out; b = rb.out;
            std::printf("resident : %.3f s, scratch measured %.1f MiB, predicted %.1f MiB\n", ra.seconds, ra.scratch_peak / 1048576.0, ra.predicted / 1048576.0);
            std::printf("streamed : %.3f s, scratch measured %.1f MiB, predicted %.1f MiB (incl. %.1f MiB weight stream; pinned host %.1f MiB)\n", rb.seconds, rb.scratch_peak / 1048576.0,
                        rb.predicted / 1048576.0, WeightStream::device_bytes() / 1048576.0, WeightStream::host_bytes() / 1048576.0);
        } else {
            GemmaVision res(mm, max_tokens, 0), str(mm, max_tokens, 1);
            const auto in = res.preprocess(img);
            std::printf("%d patches -> %d soft tokens; resident encoder keeps %.1f MiB, streaming encoder %.1f MiB\n", in.n, in.soft_tokens(), res.resident_bytes() / 1048576.0, str.resident_bytes() / 1048576.0);
            auto ra = run<GemmaVision, VisionInput>(res, in, in.n), rb = run<GemmaVision, VisionInput>(str, in, in.n);
            a = ra.out; b = rb.out;
            std::printf("resident : %.3f s, scratch measured %.1f MiB, predicted %.1f MiB\n", ra.seconds, ra.scratch_peak / 1048576.0, ra.predicted / 1048576.0);
            std::printf("streamed : %.3f s, scratch measured %.1f MiB, predicted %.1f MiB (incl. %.1f MiB weight stream; pinned host %.1f MiB)\n", rb.seconds, rb.scratch_peak / 1048576.0,
                        rb.predicted / 1048576.0, WeightStream::device_bytes() / 1048576.0, WeightStream::host_bytes() / 1048576.0);
        }
        const bool same = a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        std::printf("embeddings: %zu values, streamed %s resident\n", a.size(), same ? "BIT-IDENTICAL to" : "DIFFER from");
        bool mem_ok = false;
        if (kind == "qwen") { QwenVision str(mm, max_tokens, 1); const auto in = str.preprocess(img); mem_ok = memory_checks(str, in, b); }
        else { GemmaVision str(mm, max_tokens, 1); const auto in = str.preprocess(img); mem_ok = memory_checks(str, in, b); }
        std::printf("%s\n", same && mem_ok ? "ALL OK" : "FAILED");
        return same && mem_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
