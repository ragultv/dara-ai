#pragma once
#include <cstdint>

#include "core/manifest.h"

namespace dara::core {

// Average bytes per stored KV value, including an allowance for metadata and residuals.
// TurboQuant figures are ideal payload sizes until real allocation sizes are measured.
enum class KvMode { F32, BF16, FP8, TQ4, TQ3, Q8 };   // Q8: int8 + per-group scale (overhead passed as tq_overhead)
double bytes_per_value(KvMode m, double tq_overhead = 0.0);
const char* to_string(KvMode m);

// B_kv = L_attn * 2 * H_kv * d_head * b_kv  (bytes per token)
double kv_bytes_per_token(const ModelManifest& m, KvMode mode, double tq_overhead = 0.0);

// M_usable = min(free, total) - workspace - runtime - safety
struct MemoryBudget {
    uint64_t vram_total = 0;
    uint64_t vram_free = 0;         // queried from the driver; never assume vram_total
    uint64_t runtime_reserve = 0;   // CUDA context, driver and Windows display headroom
    uint64_t workspace_bytes = 0;   // temporary compute workspace
    uint64_t safety_reserve = 0;    // margin against other processes growing
    uint64_t ram_free = 0;          // available (not installed) physical RAM

    uint64_t usable_vram() const {
        uint64_t avail = vram_free < vram_total ? vram_free : vram_total;
        uint64_t held = runtime_reserve + workspace_bytes + safety_reserve;
        return avail > held ? avail - held : 0;
    }
};

}  // namespace dara::core
