#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace dara::core {

// Instantaneous GPU operating conditions. Benchmarks record these so a result can be judged against
// the clocks, temperature and power state it ran under.
struct GpuTelemetry {
    unsigned sm_mhz = 0, mem_mhz = 0, max_sm_mhz = 0;
    unsigned temp_c = 0;
    double power_w = 0;
    uint64_t throttle_mask = 0;  // NVML clocksThrottleReason* bits
    std::string throttle_text() const;  // e.g. "power-cap, thermal-slowdown" or "none"
};

// nullopt when NVML or the device is unavailable. Cheap enough to call between benchmark iterations.
std::optional<GpuTelemetry> query_gpu_telemetry(unsigned device_index = 0);

}  // namespace dara::core
