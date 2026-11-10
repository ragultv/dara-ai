#pragma once
#include <memory>

#include "core/system.h"

namespace dara::core {

// Real free-VRAM and compute-capability queries through the driver's nvml.dll, loaded at runtime
// so the core library has no CUDA link dependency. Returns nullptr when NVML or a device is
// unavailable (e.g. CI without a GPU); callers decide how to degrade.
std::unique_ptr<IGpuMemoryProvider> make_nvml_provider(unsigned device_index = 0);

}  // namespace dara::core
