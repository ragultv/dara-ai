#include "core/placement.h"

#include <cstdio>

namespace dara::core {

const char* to_string(Placement p) {
    switch (p) {
        case Placement::GpuOnly: return "gpu-only";
        case Placement::Hybrid: return "hybrid";
        case Placement::StorageBacked: return "storage-backed";
        case Placement::DoesNotFit: return "does-not-fit";
    }
    return "?";
}

std::string PlacementPlan::describe() const {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s (%u gpu / %u host layers, loadable=%s, performance-qualified=%s)",
                  to_string(kind), gpu_layers, host_layers, loadable ? "yes" : "no",
                  performance_qualified ? "yes" : "no");
    return buf;
}

PlacementPlan plan_placement(const PlacementRequest& req) {
    const ModelManifest& m = req.model;
    PlacementPlan plan;

    const uint64_t vision = req.vision_resident ? m.vision_bytes : 0;
    const uint64_t fixed = m.embed_output_bytes + vision + m.recurrent_state_bytes;
    const double per_token = kv_bytes_per_token(m, req.kv_mode, req.tq_overhead);
    plan.kv_bytes = static_cast<uint64_t>(per_token * static_cast<double>(req.context_tokens));

    const uint64_t usable = req.budget.usable_vram();
    const uint64_t all_layers = m.total_layer_bytes();

    const uint64_t weights_all = fixed + all_layers;
    if (usable > weights_all)
        plan.max_context_gpu_only = per_token > 0.0
            ? static_cast<uint64_t>(static_cast<double>(usable - weights_all) / per_token)
            : UINT64_MAX;

    const uint64_t state_need = fixed + plan.kv_bytes;
    const uint64_t staged_vision = req.vision_resident ? 0 : m.vision_bytes;
    if (state_need > usable) {
        plan.kind = Placement::DoesNotFit;
        plan.host_layers = m.n_layers();
        plan.host_bytes = all_layers + m.host_resident_bytes + staged_vision;
        plan.warnings.push_back("session state and fixed tensors exceed usable VRAM; reduce context or use a more compressed KV mode");
        return plan;
    }

    uint64_t room = usable - state_need;
    uint64_t used = 0;
    uint32_t n = 0;
    for (; n < m.n_layers(); ++n) {
        uint64_t b = m.layers[n].weight_bytes;
        if (b > room - used) break;
        used += b;
    }
    plan.gpu_layers = n;
    plan.host_layers = m.n_layers() - n;
    plan.gpu_bytes = state_need + used;
    plan.host_bytes = (all_layers - used) + m.host_resident_bytes + staged_vision;
    plan.loadable = true;

    if (plan.host_layers == 0 && plan.host_bytes <= req.budget.ram_free) {
        plan.kind = Placement::GpuOnly;
        plan.performance_qualified = true;
    } else if (plan.host_bytes <= req.budget.ram_free) {
        plan.kind = Placement::Hybrid;
        plan.warnings.push_back(std::to_string(plan.host_layers) + " of " + std::to_string(m.n_layers()) +
                                " layers are host-resident; not performance-qualified");
    } else {
        plan.kind = Placement::StorageBacked;
        plan.warnings.push_back("host-resident weights exceed available RAM; memory-mapped weights are loadable but may be too slow for interactive use");
    }
    return plan;
}

}  // namespace dara::core
