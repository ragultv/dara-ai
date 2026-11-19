// System / process / GPU telemetry for `dara serve` (/dara/stats, /metrics). Windows; NVML is loaded at run time from the driver (nvml.dll), so the
// server still starts without it (the GPU fields then come from the CUDA runtime only).
#pragma once

#include <cstdint>
#include <string>

namespace dara::serve {

struct ProcessMemory {
    uint64_t private_working_set = 0;   // Task Manager "Memory" (excludes file-backed pages such as the mapped GGUF)
    uint64_t working_set = 0;
    uint64_t peak_working_set = 0;
    uint64_t committed = 0;             // private commit charge
};

struct SystemMemory {
    uint64_t total = 0, available = 0;
};

struct DiskSpace {
    uint64_t total = 0, free = 0;
};

struct GpuTelemetry {
    bool nvml = false;                  // false: only the CUDA fields below are filled
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t vram_total = 0, vram_free = 0;          // cudaMemGetInfo (this process' view)
    uint64_t vram_used_all = 0;                      // NVML: used by every process on this GPU
    unsigned util_gpu = 0, util_mem = 0;             // %, last sample period
    unsigned temp_c = 0;
    double power_w = 0, power_limit_w = 0;
    unsigned clock_sm_mhz = 0, clock_mem_mhz = 0;
    int pstate = -1;
};

ProcessMemory process_memory();
SystemMemory system_memory();
DiskSpace disk_space(const std::string& path_on_disk);   // the volume holding this path
GpuTelemetry gpu_telemetry();                              // the CUDA device in use (matched to NVML by PCI bus id)

}  // namespace dara::serve
