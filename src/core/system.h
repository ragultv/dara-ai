#pragma once
#include <cstdint>
#include <string>

#include "core/budget.h"

namespace dara::core {

struct SystemMemoryInfo {
    uint64_t total_bytes = 0;                 // installed physical RAM
    uint64_t available_bytes = 0;             // plan host allocations against this
    uint64_t process_working_set_bytes = 0;   // this process
    uint64_t process_peak_working_set_bytes = 0;  // highest working set of this process so far
    uint64_t commit_limit_bytes = 0;          // system-wide
    uint64_t commit_used_bytes = 0;
};

// Throws std::runtime_error if the OS query fails. Windows only.
SystemMemoryInfo query_system_memory();

// DARA_MEMTRACE=1: prints this process's working set and private bytes with a stage label (memory budget work). No-op otherwise.
void memtrace(const char* stage);

struct GpuMemoryInfo {
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;  // real free VRAM right now, not total
};

// Keeps the planner independent of CUDA/NVML. A real NVML provider comes with CUDA bring-up.
class IGpuMemoryProvider {
public:
    virtual ~IGpuMemoryProvider() = default;
    virtual GpuMemoryInfo query() = 0;
};

class MockGpuProvider : public IGpuMemoryProvider {
public:
    explicit MockGpuProvider(GpuMemoryInfo info) : info_(std::move(info)) {}
    GpuMemoryInfo query() override { return info_; }
    void set_free(uint64_t f) { info_.free_bytes = f; }

private:
    GpuMemoryInfo info_;
};

struct Reserves {
    uint64_t runtime = 0;
    uint64_t workspace = 0;
    uint64_t safety = 0;
};

// Combines live GPU and RAM readings with explicit reserves. Uses available RAM, not installed RAM.
MemoryBudget make_budget(const SystemMemoryInfo& sys, const GpuMemoryInfo& gpu, const Reserves& r);

}  // namespace dara::core
