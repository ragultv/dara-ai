#include "core/system.h"

#include <cstdio>
#include <cstdlib>
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

void memtrace(const char* stage) {
#ifdef _WIN32
    static const bool on = [] { const char* e = std::getenv("DARA_MEMTRACE"); return e && e[0] == '1'; }();
    if (!on) return;
    // PrivateWorkingSetSize = Task Manager's "Memory" column (file-backed pages such as the mapped GGUF are excluded); PrivateUsage = committed
    PROCESS_MEMORY_COUNTERS_EX2 pmc{};
    pmc.cb = sizeof pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc)) return;
    std::fprintf(stderr, "MEMTRACE %-34s private working set %7.1f MB | working set %7.1f MB | committed %7.1f MB\n", stage, pmc.PrivateWorkingSetSize / 1048576.0,
                 pmc.WorkingSetSize / 1048576.0, pmc.PrivateUsage / 1048576.0);
#else
    (void)stage;
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
