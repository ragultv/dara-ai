#include "serve/metrics.h"

#include <cuda_runtime.h>

#include <filesystem>
#include <mutex>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace dara::serve {

ProcessMemory process_memory() {
    ProcessMemory m;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX2 pmc{};
    pmc.cb = sizeof pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc)) {
        m.private_working_set = pmc.PrivateWorkingSetSize;
        m.working_set = pmc.WorkingSetSize;
        m.peak_working_set = pmc.PeakWorkingSetSize;
        m.committed = pmc.PrivateUsage;
    }
#endif
    return m;
}

SystemMemory system_memory() {
    SystemMemory s;
#ifdef _WIN32
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) { s.total = ms.ullTotalPhys; s.available = ms.ullAvailPhys; }
#endif
    return s;
}

DiskSpace disk_space(const std::string& path_on_disk) {
    DiskSpace d;
    std::error_code ec;
    std::filesystem::path p = std::filesystem::absolute(std::filesystem::u8path(path_on_disk), ec);
    if (ec) return d;
    const auto info = std::filesystem::space(p.has_filename() && !std::filesystem::is_directory(p, ec) ? p.parent_path() : p, ec);
    if (!ec) { d.total = info.capacity; d.free = info.available; }
    return d;
}

#ifdef _WIN32
namespace {
// The few NVML entry points used, declared locally (no nvml.h / nvml.lib dependency).
using nvmlReturn_t = int;
using nvmlDevice_t = void*;
struct nvmlUtilization_t { unsigned gpu, memory; };
struct nvmlMemory_t { unsigned long long total, free, used; };
struct Nvml {
    bool ok = false;
    nvmlDevice_t dev = nullptr;
    nvmlReturn_t (*GetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*GetTemperature)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*GetPowerUsage)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*GetEnforcedPowerLimit)(nvmlDevice_t, unsigned*) = nullptr;
    nvmlReturn_t (*GetClockInfo)(nvmlDevice_t, int, unsigned*) = nullptr;
    nvmlReturn_t (*GetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*GetPerformanceState)(nvmlDevice_t, int*) = nullptr;
};

Nvml& nvml() {
    static Nvml n;
    static std::once_flag once;
    std::call_once(once, [] {
        HMODULE h = LoadLibraryA("nvml.dll");
        if (!h) {
            char dir[MAX_PATH];
            if (GetEnvironmentVariableA("ProgramW6432", dir, MAX_PATH)) h = LoadLibraryA((std::string(dir) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll").c_str());
        }
        if (!h) return;
        auto sym = [&](const char* s) { return reinterpret_cast<void*>(GetProcAddress(h, s)); };
        auto init = reinterpret_cast<nvmlReturn_t (*)()>(sym("nvmlInit_v2"));
        auto by_pci = reinterpret_cast<nvmlReturn_t (*)(const char*, nvmlDevice_t*)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
        if (!init || !by_pci || init() != 0) return;
        int dev = 0;
        cudaGetDevice(&dev);
        char bus[64] = {};
        if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess || by_pci(bus, &n.dev) != 0) return;
        n.GetUtilizationRates = reinterpret_cast<decltype(n.GetUtilizationRates)>(sym("nvmlDeviceGetUtilizationRates"));
        n.GetTemperature = reinterpret_cast<decltype(n.GetTemperature)>(sym("nvmlDeviceGetTemperature"));
        n.GetPowerUsage = reinterpret_cast<decltype(n.GetPowerUsage)>(sym("nvmlDeviceGetPowerUsage"));
        n.GetEnforcedPowerLimit = reinterpret_cast<decltype(n.GetEnforcedPowerLimit)>(sym("nvmlDeviceGetEnforcedPowerLimit"));
        n.GetClockInfo = reinterpret_cast<decltype(n.GetClockInfo)>(sym("nvmlDeviceGetClockInfo"));
        n.GetMemoryInfo = reinterpret_cast<decltype(n.GetMemoryInfo)>(sym("nvmlDeviceGetMemoryInfo"));
        n.GetPerformanceState = reinterpret_cast<decltype(n.GetPerformanceState)>(sym("nvmlDeviceGetPerformanceState"));
        n.ok = true;
    });
    return n;
}
}  // namespace
#endif

GpuTelemetry gpu_telemetry() {
    GpuTelemetry g;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) { g.name = p.name; g.cc_major = p.major; g.cc_minor = p.minor; }
        size_t fr = 0, tot = 0;
        if (cudaMemGetInfo(&fr, &tot) == cudaSuccess) { g.vram_free = fr; g.vram_total = tot; }
    }
#ifdef _WIN32
    Nvml& n = nvml();
    if (n.ok) {
        g.nvml = true;
        nvmlUtilization_t u{};
        if (n.GetUtilizationRates && n.GetUtilizationRates(n.dev, &u) == 0) { g.util_gpu = u.gpu; g.util_mem = u.memory; }
        unsigned v = 0;
        if (n.GetTemperature && n.GetTemperature(n.dev, 0, &v) == 0) g.temp_c = v;   // NVML_TEMPERATURE_GPU
        if (n.GetPowerUsage && n.GetPowerUsage(n.dev, &v) == 0) g.power_w = v / 1000.0;
        if (n.GetEnforcedPowerLimit && n.GetEnforcedPowerLimit(n.dev, &v) == 0) g.power_limit_w = v / 1000.0;
        if (n.GetClockInfo && n.GetClockInfo(n.dev, 1, &v) == 0) g.clock_sm_mhz = v;   // NVML_CLOCK_SM
        if (n.GetClockInfo && n.GetClockInfo(n.dev, 2, &v) == 0) g.clock_mem_mhz = v;  // NVML_CLOCK_MEM
        nvmlMemory_t m{};
        if (n.GetMemoryInfo && n.GetMemoryInfo(n.dev, &m) == 0) g.vram_used_all = m.used;
        int ps = -1;
        if (n.GetPerformanceState && n.GetPerformanceState(n.dev, &ps) == 0) g.pstate = ps;
    }
#endif
    return g;
}

}  // namespace dara::serve
