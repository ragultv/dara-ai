#include "core/budget.h"

namespace dara::core {

double bytes_per_value(KvMode m, double tq_overhead) {
    switch (m) {
        case KvMode::F32: return 4.0;
        case KvMode::BF16: return 2.0;
        case KvMode::FP8: return 1.0;
        case KvMode::TQ4: return 0.5 + tq_overhead;
        case KvMode::TQ3: return 0.375 + tq_overhead;
        case KvMode::Q8: return 1.0 + tq_overhead;
    }
    return 2.0;
}

const char* to_string(KvMode m) {
    switch (m) {
        case KvMode::F32: return "f32";
        case KvMode::BF16: return "bf16";
        case KvMode::FP8: return "fp8";
        case KvMode::TQ4: return "tq4";
        case KvMode::TQ3: return "tq3";
        case KvMode::Q8: return "q8";
    }
    return "?";
}

double kv_bytes_per_token(const ModelManifest& s, KvMode m, double tq_overhead) {
    return static_cast<double>(s.kv_layers()) * 2.0 * s.kv_heads * s.head_dim *
           bytes_per_value(m, tq_overhead);
}

}  // namespace dara::core
