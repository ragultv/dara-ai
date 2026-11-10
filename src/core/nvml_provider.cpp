#include "core/nvml_provider.h"

#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace dara::core {
namespace {

#ifdef _WIN32
// Minimal ABI subset of nvml.h (stable C API).
using nvmlReturn_t = int;
using nvmlDevice_t = void*;
struct nvmlMemory_t { unsigned long long total, free, used; };

struct Nvml {
    HMODULE lib = nullptr;
    nvmlReturn_t (*init)() = nullptr;
    nvmlReturn_t (*shutdown)() = nullptr;
    nvmlReturn_t (*handle)(unsigned, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*name)(nvmlDevice_t, char*, unsigned) = nullptr;
    nvmlReturn_t (*mem)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*cc)(nvmlDevice_t, int*, int*) = nullptr;
    bool inited = false;

    ~Nvml() {
        if (inited && shutdown) shutdown();
        if (lib) FreeLibrary(lib);
    }
    template <class F> bool sym(F& f, const char* n) {
        f = reinterpret_cast<F>(GetProcAddress(lib, n));
        return f != nullptr;
    }
    bool load() {
        // System32 holds the driver's copy; never search the current directory.
        lib = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!lib) return false;
        if (!(sym(init, "nvmlInit_v2") && sym(shutdown, "nvmlShutdown") &&
              sym(handle, "nvmlDeviceGetHandleByIndex_v2") && sym(name, "nvmlDeviceGetName") &&
              sym(mem, "nvmlDeviceGetMemoryInfo") && sym(cc, "nvmlDeviceGetCudaComputeCapability")))
            return false;
        inited = init() == 0;
        return inited;
    }
};

class NvmlProvider : public IGpuMemoryProvider {
public:
    NvmlProvider(std::unique_ptr<Nvml> n, nvmlDevice_t d) : nvml_(std::move(n)), dev_(d) {}
    GpuMemoryInfo query() override {
        GpuMemoryInfo g;
        char buf[96] = {};
        if (nvml_->name(dev_, buf, sizeof buf) == 0) g.name = buf;
        nvmlMemory_t m{};
        if (nvml_->mem(dev_, &m) != 0) throw std::runtime_error("nvmlDeviceGetMemoryInfo failed");
        g.total_bytes = m.total;
        g.free_bytes = m.free;
        nvml_->cc(dev_, &g.cc_major, &g.cc_minor);
        return g;
    }

private:
    std::unique_ptr<Nvml> nvml_;
    nvmlDevice_t dev_;
};
#endif

}  // namespace

std::unique_ptr<IGpuMemoryProvider> make_nvml_provider(unsigned device_index) {
#ifdef _WIN32
    auto n = std::make_unique<Nvml>();
    if (!n->load()) return nullptr;
    nvmlDevice_t d = nullptr;
    if (n->handle(device_index, &d) != 0) return nullptr;
    return std::make_unique<NvmlProvider>(std::move(n), d);
#else
    (void)device_index;
    return nullptr;
#endif
}

}  // namespace dara::core
