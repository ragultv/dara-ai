// CUDA-graph decode tests for the Qwen backend (hybrid DeltaNet + attention): see tests/graph_test_common.h for the scenarios.
// Restore points copy the DeltaNet recurrent state AND the KV length, so a restored session replays the graph from the right state.
// Usage: qwen_graph_test <model.gguf>   (a fully VRAM-resident model, e.g. the 4B on a 4 GB GPU)
#include <cstdio>
#include <memory>

#include "graph_test_common.h"
#include "io/gguf_checkpoint.h"
#include "model/qwen_model.h"
#include "model/qwen_weights.h"

#include <cstdlib>
#include <string>

using namespace dara::model;

// DARA_TEST_KV=q8|tq4 runs every scenario with that K/V cache format (default f32); DARA_INT8=1 (read by the kernels) adds int8 activations.
inline dara::model::ResidencyOptions test_options() {
    dara::model::ResidencyOptions o;
    if (const char* e = std::getenv("DARA_TEST_KV")) {
        const std::string v = e;
        if (v == "q8") o.kv_format = dara::model::KvFormat::Q8;
        else if (v == "tq4") o.kv_format = dara::model::KvFormat::TQ4;
        else if (v == "k16v8") o.kv_format = dara::model::KvFormat::K16V8;
    }
    return o;
}


struct QwenAdapter {
    using Model = QwenModel;
    using Snap = QwenModel::Checkpoint;
    const dara::io::GgufCheckpoint& g;
    QwenConfig cfg;
    GgufWeights weights;
    QwenAdapter(const dara::io::GgufCheckpoint& gg) : g(gg), cfg(QwenConfig::from_gguf(gg)), weights(gg, cfg) {}
    std::unique_ptr<Model> make() { return std::make_unique<QwenModel>(cfg, weights, 1024, test_options()); }
    int vocab(Model&) { return cfg.vocab; }
    Snap snap(Model& m) { return m.checkpoint(); }
    void rewind(Model& m, const Snap& s) { m.restore(s); }
    std::vector<float> fwd(Model& m, const std::vector<int32_t>& t, int from) { return m.forward(t.data(), static_cast<int>(t.size()), from); }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::puts("usage: qwen_graph_test <model.gguf>"); return 2; }
    try {
        dara::io::GgufCheckpoint g(argv[1]);
        QwenAdapter a(g);
        graphtest::Tests<QwenAdapter> t(a);
        return t.run_all("Qwen") ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
