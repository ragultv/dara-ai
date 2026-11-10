#include "core/system.h"

#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif

namespace dara::core {

SystemMemoryInfo query_system_memory() {
#ifdef _WIN32
    SystemMemoryInfo s;

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (!GlobalMemoryStatusEx(&ms)) throw std::runtime_error("GlobalMemoryStatusEx failed");
    s.total_bytes = ms.ullTotalPhys;
    s.available_bytes = ms.ullAvailPhys;

    PERFORMANCE_INFORMATION pi{};
    pi.cb = sizeof pi;
    if (!GetPerformanceInfo(&pi, sizeof pi)) throw std::runtime_error("GetPerformanceInfo failed");
    s.commit_limit_bytes = static_cast<uint64_t>(pi.CommitLimit) * pi.PageSize;
    s.commit_used_bytes = static_cast<uint64_t>(pi.CommitTotal) * pi.PageSize;

    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    s.process_working_set_bytes = pmc.WorkingSetSize;
    s.process_peak_working_set_bytes = pmc.PeakWorkingSetSize;
    return s;
#else
    throw std::runtime_error("query_system_memory: Windows only");
#endif
}

MemoryBudget make_budget(const SystemMemoryInfo& sys, const GpuMemoryInfo& gpu, const Reserves& r) {
    MemoryBudget b;
    b.vram_total = gpu.total_bytes;
    b.vram_free = gpu.free_bytes;
    b.runtime_reserve = r.runtime;
    b.workspace_bytes = r.workspace;
    b.safety_reserve = r.safety;
    b.ram_free = sys.available_bytes;
    return b;
}

}  // namespace dara::core
