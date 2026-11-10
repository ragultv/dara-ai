#include "core/gpu_telemetry.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace dara::core {

std::string GpuTelemetry::throttle_text() const {
    struct Bit { uint64_t bit; const char* name; };
    static const Bit bits[] = {
        {0x1, "gpu-idle"}, {0x2, "app-clocks-setting"}, {0x4, "sw-power-cap"}, {0x8, "hw-slowdown"},
        {0x10, "sync-boost"}, {0x20, "sw-thermal-slowdown"}, {0x40, "hw-thermal-slowdown"},
        {0x80, "hw-power-brake"}, {0x100, "display-clocks-setting"}};
    std::string s;
    for (auto& b : bits)
        if (throttle_mask & b.bit) s += (s.empty() ? "" : ", ") + std::string(b.name);
    return s.empty() ? "none" : s;
}

#ifdef _WIN32
namespace {
using R = int;
using Dev = void*;

struct Api {
    HMODULE lib = nullptr;
    R (*init)() = nullptr;
    R (*handle)(unsigned, Dev*) = nullptr;
    R (*clock)(Dev, int, unsigned*) = nullptr;
    R (*max_clock)(Dev, int, unsigned*) = nullptr;
    R (*temp)(Dev, int, unsigned*) = nullptr;
    R (*power)(Dev, unsigned*) = nullptr;
    R (*throttle)(Dev, unsigned long long*) = nullptr;
    bool ok = false;

    template <class F> bool sym(F& f, const char* n) { f = reinterpret_cast<F>(GetProcAddress(lib, n)); return f; }
    Api() {
        lib = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!lib) return;
        ok = sym(init, "nvmlInit_v2") && sym(handle, "nvmlDeviceGetHandleByIndex_v2") &&
             sym(clock, "nvmlDeviceGetClockInfo") && sym(max_clock, "nvmlDeviceGetMaxClockInfo") &&
             sym(temp, "nvmlDeviceGetTemperature") && sym(power, "nvmlDeviceGetPowerUsage") &&
             sym(throttle, "nvmlDeviceGetCurrentClocksThrottleReasons") && init() == 0;
    }
    // The library stays loaded for the process lifetime on purpose (telemetry is polled repeatedly).
};
}  // namespace
#endif

std::optional<GpuTelemetry> query_gpu_telemetry(unsigned device_index) {
#ifdef _WIN32
    static Api api;
    if (!api.ok) return std::nullopt;
    Dev d = nullptr;
    if (api.handle(device_index, &d) != 0) return std::nullopt;
    GpuTelemetry t;
    unsigned v = 0;
    if (api.clock(d, 1, &v) == 0) t.sm_mhz = v;          // NVML_CLOCK_SM
    if (api.clock(d, 2, &v) == 0) t.mem_mhz = v;         // NVML_CLOCK_MEM
    if (api.max_clock(d, 1, &v) == 0) t.max_sm_mhz = v;
    if (api.temp(d, 0, &v) == 0) t.temp_c = v;           // NVML_TEMPERATURE_GPU
    if (api.power(d, &v) == 0) t.power_w = v / 1000.0;   // milliwatts
    unsigned long long m = 0;
    if (api.throttle(d, &m) == 0) t.throttle_mask = m;
    return t;
#else
    (void)device_index;
    return std::nullopt;
#endif
}

}  // namespace dara::core
