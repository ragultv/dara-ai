// CUDA-graph decode tests for the Nano model (Gemma 4 E2B): see tests/graph_test_common.h for the scenarios.
// Usage: nano_graph_test <model.gguf>
#include <cstdio>
#include <memory>

#include "graph_test_common.h"
#include "io/gguf_checkpoint.h"
#include "model/gemma_model.h"

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


struct NanoAdapter {
    using Model = GemmaModel;
    using Snap = size_t;
    const dara::io::GgufCheckpoint& g;
    std::unique_ptr<Model> make() { return std::make_unique<GemmaModel>(g, 1024, test_options()); }
    int vocab(Model& m) { return m.config().vocab; }
    Snap snap(Model& m) { return m.position(); }
    void rewind(Model& m, const Snap& s) { m.rewind(s); }  // Gemma has only K/V state: the length is the whole restore point
    std::vector<float> fwd(Model& m, const std::vector<int32_t>& t, int from) { return m.forward(t.data(), static_cast<int>(t.size()), from); }
};

int main(int argc, char** argv) {
    if (argc < 2) { std::puts("usage: nano_graph_test <model.gguf>"); return 2; }
    try {
        dara::io::GgufCheckpoint g(argv[1]);
        NanoAdapter a{g};
        graphtest::Tests<NanoAdapter> t(a);
        return t.run_all("Nano") ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
