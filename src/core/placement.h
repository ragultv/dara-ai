#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "core/budget.h"

namespace dara::core {

// What the runtime may honestly claim about a configuration. Results for different classes
// must never be merged into one headline number.
enum class Placement {
    GpuOnly,        // all layers resident in VRAM
    Hybrid,         // some layers host-resident
    StorageBacked,  // host-resident weights exceed free RAM; memory-mapped
    DoesNotFit      // fixed GPU needs exceed VRAM even with every layer offloaded
};
const char* to_string(Placement p);

struct PlacementRequest {
    ModelManifest model;
    MemoryBudget budget;
    KvMode kv_mode = KvMode::FP8;
    double tq_overhead = 0.0;
    uint64_t context_tokens = 0;
    bool vision_resident = true;  // false: encoder staged on demand
};

struct PlacementPlan {
    Placement kind = Placement::DoesNotFit;
    // Technically loadable and runnable. StorageBacked is loadable but is NOT a claim of
    // interactive speed.
    bool loadable = false;
    // Only GpuOnly may be benchmarked as a performance configuration.
    bool performance_qualified = false;
    uint32_t gpu_layers = 0;   // layers [0, gpu_layers) are VRAM-resident
    uint32_t host_layers = 0;
    uint64_t gpu_bytes = 0;    // weights + session state planned in VRAM
    uint64_t host_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t max_context_gpu_only = 0;
    std::vector<std::string> warnings;

    std::string describe() const;
};

// Correctness-first planner. Reserves session state and fixed tensors first, then places layers
// from the front, using each layer's own size. KV sizes are ideal arithmetic until measured.
PlacementPlan plan_placement(const PlacementRequest& req);

}  // namespace dara::core
