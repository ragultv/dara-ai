#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "core/backend.h"
#include "core/budget.h"
#include "core/manifest.h"
#include "core/mock_backend.h"
#include "core/nvml_provider.h"
#include "core/placement.h"
#include "core/session.h"
#include "core/system.h"

using namespace dara::core;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)
template <class F> static bool throws(F f) { try { f(); } catch (const std::exception&) { return true; } return false; }

static constexpr uint64_t GiB = 1ull << 30;
static constexpr uint64_t MiB = 1ull << 20;

// Q4_K_M-class: roughly 4.85 bits/weight over 16-bit storage. Estimate until a GGUF header gives exact sizes.
static QuantProfile q4() { return {4.85 / 16.0, 4.85 / 16.0, 1.0}; }
static ModelManifest manifest(const char* file) {
    return load_manifest(std::string(DARA_SOURCE_DIR) + "/manifests/" + file, q4());
}

// ---- manifests (real checkpoint data) ------------------------------------------------------

static void test_real_manifest() {
    auto m = manifest("qwen3.5-4b.json");
    CHECK(m.family == "qwen35" && m.n_layers() == 32 && m.kv_layers() == 8);
    CHECK(m.kv_heads == 4 && m.head_dim == 256);
    CHECK(m.revision.size() == 40);              // pinned commit, not a branch
    CHECK(m.mtp_bytes > 0);                      // draft head weights are in the checkpoint
    // Layer kinds follow 3 DeltaNet then 1 attention.
    CHECK(m.layers[0].kind == LayerKind::DeltaNet && m.layers[3].kind == LayerKind::Attention);
    // Layers differ in size: attention and DeltaNet blocks are not interchangeable.
    CHECK(m.layers[0].weight_bytes != m.layers[3].weight_bytes);
    // Recurrent state: ~50 MB per the plan (24 layers, FP32).
    CHECK(m.recurrent_state_bytes > 40 * MiB && m.recurrent_state_bytes < 60 * MiB);
    CHECK(manifest("qwen3.5-9b.json").total_layer_bytes() > m.total_layer_bytes());
}

static void test_bad_manifest_rejected() {
    CHECK(throws([] { load_manifest("does-not-exist.json", q4()); }));
}

static void test_kv_formula() {
    auto m = manifest("qwen3.5-4b.json");
    // Plan: 4B needs 4 GiB FP16 KV at 128K and 1 GiB at ideal 4-bit.
    CHECK(kv_bytes_per_token(m, KvMode::BF16) * 131072.0 == 4.0 * GiB);
    CHECK(kv_bytes_per_token(m, KvMode::TQ4) * 131072.0 == 1.0 * GiB);
}

// ---- Windows RAM discovery -----------------------------------------------------------------

static void test_system_memory() {
    auto s = query_system_memory();
    CHECK(s.total_bytes >= 1 * GiB);
    CHECK(s.available_bytes > 0 && s.available_bytes <= s.total_bytes);
    CHECK(s.process_working_set_bytes > 0 && s.process_working_set_bytes < s.total_bytes);
    CHECK(s.commit_limit_bytes >= s.commit_used_bytes && s.commit_used_bytes > 0);
    std::printf("  RAM: %.1f GiB total, %.1f GiB available, working set %.0f MiB\n",
                s.total_bytes / double(GiB), s.available_bytes / double(GiB),
                s.process_working_set_bytes / double(MiB));
}

// ---- real GPU discovery (skips when no NVIDIA driver) --------------------------------------

static void test_nvml_provider() {
    auto gpu = make_nvml_provider();
    if (!gpu) { std::puts("  NVML unavailable: skipped"); return; }
    auto g = gpu->query();
    CHECK(!g.name.empty());
    CHECK(g.cc_major >= 5);
    CHECK(g.total_bytes >= 1 * GiB && g.free_bytes > 0 && g.free_bytes <= g.total_bytes);
    // Real free memory is below total (display + context); the planner must use it.
    CHECK(g.free_bytes < g.total_bytes);
    std::printf("  NVML: %s sm_%d%d, %.0f MiB free of %.0f MiB\n", g.name.c_str(), g.cc_major, g.cc_minor,
                g.free_bytes / double(MiB), g.total_bytes / double(MiB));
    // End to end: real RAM + real VRAM -> budget -> plan for the 4B tier.
    PlacementRequest r;
    r.model = manifest("qwen3.5-4b.json");
    r.budget = make_budget(query_system_memory(), g, {300 * MiB, 256 * MiB, 256 * MiB});
    r.context_tokens = 8192;
    auto p = plan_placement(r);
    std::printf("  live plan, 4B @ 8K: %s\n", p.describe().c_str());
    CHECK(p.gpu_bytes <= r.budget.usable_vram());
}

// ---- placement across laptop VRAM sizes (mock GPU provider) --------------------------------

static MemoryBudget budget_for(uint64_t vram_gib, uint64_t free_ram_gib = 24) {
    MockGpuProvider gpu({"mock", 8, 9, vram_gib * GiB, vram_gib * GiB - 400 * MiB});
    SystemMemoryInfo sys;
    sys.available_bytes = free_ram_gib * GiB;
    return make_budget(sys, gpu.query(), {300 * MiB, 256 * MiB, 256 * MiB});
}

static PlacementPlan plan_for(const char* file, uint64_t vram_gib, uint64_t ctx, KvMode kv = KvMode::FP8,
                              uint64_t ram_gib = 24) {
    PlacementRequest r;
    r.model = manifest(file);
    r.budget = budget_for(vram_gib, ram_gib);
    r.context_tokens = ctx;
    r.kv_mode = kv;
    return plan_placement(r);
}

static void test_core_4b_across_vram_sizes() {
    auto p4 = plan_for("qwen3.5-4b.json", 4, 8192);
    auto p8 = plan_for("qwen3.5-4b.json", 8, 8192);
    auto p12 = plan_for("qwen3.5-4b.json", 12, 8192);
    auto p16 = plan_for("qwen3.5-4b.json", 16, 8192);
    std::printf("  4B @ 4/8/12/16 GB: %s | %s | %s | %s\n", p4.describe().c_str(), p8.describe().c_str(),
                p12.describe().c_str(), p16.describe().c_str());
    CHECK(p4.kind == Placement::Hybrid && !p4.performance_qualified && p4.loadable);
    CHECK(p8.kind == Placement::GpuOnly && p8.performance_qualified);
    CHECK(p12.kind == Placement::GpuOnly && p16.kind == Placement::GpuOnly);
    // More VRAM never means fewer GPU layers or less context headroom.
    CHECK(p4.gpu_layers <= p8.gpu_layers);
    CHECK(p8.max_context_gpu_only <= p12.max_context_gpu_only && p12.max_context_gpu_only <= p16.max_context_gpu_only);
}

static void test_pro_9b_across_vram_sizes() {
    auto p4 = plan_for("qwen3.5-9b.json", 4, 8192);
    auto p6 = plan_for("qwen3.5-9b.json", 6, 8192);
    auto p8 = plan_for("qwen3.5-9b.json", 8, 8192);
    auto p12 = plan_for("qwen3.5-9b.json", 12, 8192);
    std::printf("  9B @ 4/6/8/12 GB: %s | %s | %s | %s\n", p4.describe().c_str(), p6.describe().c_str(),
                p8.describe().c_str(), p12.describe().c_str());
    CHECK(p4.kind == Placement::Hybrid && p6.kind == Placement::Hybrid);
    // 8 GB is borderline at 8K context with FP8 KV (estimated Q4 sizes); assert only monotonicity.
    CHECK(p4.gpu_layers <= p6.gpu_layers && p6.gpu_layers <= p8.gpu_layers && p8.gpu_layers <= p12.gpu_layers);
    CHECK(p12.kind == Placement::GpuOnly);
    CHECK(p4.gpu_bytes <= budget_for(4).usable_vram());
    CHECK(p6.gpu_bytes <= budget_for(6).usable_vram());
}

static void test_reserves_shrink_usable_vram() {
    MockGpuProvider gpu({"mock", 8, 9, 8 * GiB, 8 * GiB});
    SystemMemoryInfo sys;
    sys.available_bytes = 16 * GiB;
    auto none = make_budget(sys, gpu.query(), {0, 0, 0});
    auto some = make_budget(sys, gpu.query(), {300 * MiB, 500 * MiB, 200 * MiB});
    CHECK(none.usable_vram() == 8 * GiB);
    CHECK(none.usable_vram() - some.usable_vram() == 1000 * MiB);
    // Reserves larger than free memory must clamp to zero, not wrap around.
    CHECK(make_budget(sys, gpu.query(), {9 * GiB, 0, 0}).usable_vram() == 0);
}

static void test_placement_follows_live_free_vram() {
    MockGpuProvider gpu({"mock", 8, 9, 8 * GiB, 7 * GiB});
    SystemMemoryInfo sys;
    sys.available_bytes = 24 * GiB;
    PlacementRequest r;
    r.model = manifest("qwen3.5-9b.json");
    r.context_tokens = 4096;
    r.budget = make_budget(sys, gpu.query(), {300 * MiB, 256 * MiB, 256 * MiB});
    auto roomy = plan_placement(r);
    gpu.set_free(3 * GiB);  // another application took VRAM
    r.budget = make_budget(sys, gpu.query(), {300 * MiB, 256 * MiB, 256 * MiB});
    auto tight = plan_placement(r);
    CHECK(tight.gpu_layers < roomy.gpu_layers);
}

static void test_storage_backed_is_loadable_but_not_qualified() {
    // 9B with little free RAM: host layers cannot stay resident.
    auto p = plan_for("qwen3.5-9b.json", 4, 4096, KvMode::FP8, 1);
    CHECK(p.kind == Placement::StorageBacked);
    CHECK(p.loadable && !p.performance_qualified);
    CHECK(!p.warnings.empty());
}

static void test_does_not_fit() {
    auto p = plan_for("qwen3.5-9b.json", 4, 262144, KvMode::BF16);
    CHECK(p.kind == Placement::DoesNotFit && !p.loadable && !p.performance_qualified);
}

static void test_heterogeneous_layers_use_own_sizes() {
    PlacementRequest r;
    r.model.family = "qwen35";
    r.model.kv_heads = 1; r.model.head_dim = 1;
    for (uint64_t b : {100, 10, 10, 500}) {
        LayerSpec l; l.weight_bytes = l.stored_bytes = b * MiB;
        r.model.layers.push_back(l);
    }
    r.budget = {600 * MiB, 600 * MiB, 0, 0, 0, 8 * GiB};
    r.context_tokens = 0;
    auto p = plan_placement(r);
    // 100+10+10 = 120 MiB fit; adding the 500 MiB layer would reach 620 MiB. A uniform average
    // (155 MiB each) would have wrongly said 3.9 layers fit.
    CHECK(p.gpu_layers == 3 && p.host_layers == 1);
    CHECK(p.gpu_bytes == 120 * MiB && p.host_bytes == 500 * MiB);
}

// ---- backends: contract and checkpoint/restore ---------------------------------------------

static std::vector<int32_t> seq(int32_t from, int n) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i) v.push_back(from + i);
    return v;
}

static PlacementPlan loadable_plan() {
    PlacementPlan p;
    p.loadable = true;
    return p;
}

template <class Backend>
static void run_contract(const char* family) {
    ModelManifest m;
    m.family = family;
    BlockPool pool(64, 32);
    Backend be;
    be.load(m, loadable_plan());

    {
        Session s(pool);
        be.prefill(s, {seq(10, 100)});
        auto cp = be.checkpoint(s);
        CHECK(cp.n_tokens == 100);
        const auto free_at_cp = pool.free_blocks();

        // Decode forward, then record what a clean continuation produces.
        std::vector<int32_t> first;
        for (int i = 0; i < 6; ++i) first.push_back(be.decode(s).token);
        CHECK(s.tokens().size() == 106);

        // Restore rewinds KV position and state: the same continuation must repeat exactly.
        be.restore(s, cp);
        CHECK(s.tokens().size() == 100);
        CHECK(pool.free_blocks() >= free_at_cp);
        std::vector<int32_t> again;
        for (int i = 0; i < 6; ++i) again.push_back(be.decode(s).token);
        CHECK(first == again);

        // Failed restores leave the session unchanged: wrong history, ahead of session, wrong owner.
        auto tokens_before = s.tokens();
        auto table_before = s.page_table();
        Session other(pool);
        be.prefill(other, {seq(500, 100)});
        auto foreign = be.checkpoint(other);
        CHECK(throws([&] { be.restore(s, foreign); }));          // diverged history
        Session shorter(pool);
        be.prefill(shorter, {std::vector<int32_t>(tokens_before.begin(), tokens_before.begin() + 50)});
        CHECK(throws([&] { be.restore(shorter, cp); }));         // checkpoint ahead of session
        CHECK(shorter.tokens().size() == 50);
        Checkpoint alien = cp;
        alien.state = std::make_shared<MockRecurrentState>();
        if (std::string(family) == "qwen35") alien.state = std::make_shared<MockWindowState>();
        CHECK(throws([&] { be.restore(s, alien); }));            // other backend's state
        Checkpoint stateless = cp;
        stateless.state.reset();
        CHECK(throws([&] { be.restore(s, stateless); }));        // no state at all

        // The session is bit-for-bit as before: tokens, page table, and state (continues like a replay).
        CHECK(s.tokens() == tokens_before && s.page_table() == table_before);
        Session replay(pool);
        be.prefill(replay, {tokens_before});
        CHECK(be.decode(s).token == be.decode(replay).token);

        // Rewinding by truncate alone is refused: state would be stale.
        s.truncate(50);
        CHECK(throws([&] { be.decode(s); }));
    }
    CHECK(pool.free_blocks() == 64);  // everything returned
}

static void test_backends() {
    run_contract<MockQwenBackend>("qwen35");
    run_contract<MockGemmaBackend>("gemma4");
}

static void test_backend_load_validation() {
    ModelManifest gemma; gemma.family = "gemma4";
    MockQwenBackend q;
    CHECK(throws([&] { q.load(gemma, loadable_plan()); }));      // wrong family
    ModelManifest qwen; qwen.family = "qwen35";
    PlacementPlan bad;                                            // not loadable
    CHECK(throws([&] { q.load(qwen, bad); }));
    BlockPool pool(4, 32);
    Session s(pool);
    CHECK(throws([&] { q.prefill(s, {seq(0, 4)}); }));            // not loaded
}

static void test_prefill_out_of_blocks_leaves_session_intact() {
    ModelManifest m; m.family = "qwen35";
    BlockPool pool(2, 32);
    MockQwenBackend be;
    be.load(m, loadable_plan());
    Session s(pool);
    be.prefill(s, {seq(0, 40)});
    auto cp = be.checkpoint(s);
    CHECK(throws([&] { be.prefill(s, {seq(0, 200)}); }));
    CHECK(s.tokens().size() == 40 && pool.free_blocks() == 0);
    // State was not advanced by the failed call: checkpoint still equals the earlier one.
    auto after = be.checkpoint(s);
    CHECK(after.prefix_hash == cp.prefix_hash);
    CHECK(static_cast<const MockRecurrentState*>(after.state.get())->h ==
          static_cast<const MockRecurrentState*>(cp.state.get())->h);
}

// ---- pool and session ----------------------------------------------------------------------

static void test_pool_returns_to_baseline() {
    BlockPool pool(10, 32);
    {
        Session s(pool);
        CHECK(s.append(seq(0, 100)));
        CHECK(pool.free_blocks() == 6);
        s.truncate(33);
        CHECK(pool.free_blocks() == 8);
        s.truncate(32);
        CHECK(pool.free_blocks() == 9);
    }
    CHECK(pool.free_blocks() == 10);
}

static void test_append_is_all_or_nothing() {
    BlockPool pool(2, 32);
    Session s(pool);
    CHECK(s.append(seq(0, 40)));
    CHECK(!s.append(seq(100, 100)));
    CHECK(s.tokens().size() == 40 && pool.free_blocks() == 0);
    Session t(pool);
    CHECK(!t.append(seq(0, 1)) && t.page_table().empty());
}

static void test_prefix_reuse() {
    BlockPool pool(16, 32);
    Session s(pool);
    s.append(seq(0, 100));
    auto next = seq(0, 80);
    for (int i = 0; i < 20; ++i) next.push_back(9000 + i);
    CHECK(s.reuse_prefix(next) == 80);
    CHECK(s.tokens().size() == 80 && pool.free_blocks() == 16 - 3);
    Session u(pool);
    u.append(seq(0, 50));
    CHECK(u.reuse_prefix(seq(0, 50)) == 49);
    CHECK(u.reuse_prefix(seq(500, 10)) == 0 && u.tokens().empty());
}

int main() {
    test_real_manifest();
    test_bad_manifest_rejected();
    test_kv_formula();
    test_system_memory();
    test_nvml_provider();
    test_core_4b_across_vram_sizes();
    test_pro_9b_across_vram_sizes();
    test_reserves_shrink_usable_vram();
    test_placement_follows_live_free_vram();
    test_storage_backed_is_loadable_but_not_qualified();
    test_does_not_fit();
    test_heterogeneous_layers_use_own_sizes();
    test_backends();
    test_backend_load_validation();
    test_prefill_out_of_blocks_leaves_session_intact();
    test_pool_returns_to_baseline();
    test_append_is_all_or_nothing();
    test_prefix_reuse();
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
