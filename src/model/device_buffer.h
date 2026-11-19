#pragma once
// RAII device buffers with allocation accounting. Every device allocation made through these types is counted, so a
// run can report its peak device memory by category (the memory-accounting gate), not just what nvidia-smi shows.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace dara::model {

struct DeviceMemoryStats {
    std::atomic<size_t> current{0}, peak{0};
    void add(size_t n) {
        const size_t c = current.fetch_add(n) + n;
        size_t p = peak.load();
        while (c > p && !peak.compare_exchange_weak(p, c)) {}
    }
    void sub(size_t n) { current.fetch_sub(n); }
    void reset_peak() { peak.store(current.load()); }
};
inline DeviceMemoryStats& device_stats() { static DeviceMemoryStats s; return s; }

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
inline void cuda_check(int e, const char* what) { cuda_check(static_cast<cudaError_t>(e), what); }

// Dara's kernels are built for Turing (sm_75) and newer (CUDA 13 dropped Pascal). Fails early with a clear message instead of a
// later "no kernel image is available" error; returns a one-line description of the GPU the runtime will use.
inline std::string require_supported_gpu() {
    int count = 0;
    const cudaError_t ce = cudaGetDeviceCount(&count);
    if (ce == cudaErrorInsufficientDriver)
        throw std::runtime_error("the installed NVIDIA driver is too old for this build (CUDA 13 needs driver 580 or newer): update the GeForce driver");
    if (ce == cudaErrorNoDevice || (ce == cudaSuccess && count == 0))
        throw std::runtime_error("no NVIDIA GPU with a working driver was found: Dara needs an NVIDIA GPU (GTX 16xx or RTX 20/30/40/50 series) and the GeForce driver");
    cuda_check(ce, "cudaGetDeviceCount");
    int dev = 0;
    cuda_check(cudaGetDevice(&dev), "cudaGetDevice");
    cudaDeviceProp p{};
    cuda_check(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
    const int cc = p.major * 10 + p.minor;
    if (cc < 75)
        throw std::runtime_error(std::string("unsupported GPU: ") + p.name + " (compute capability " + std::to_string(p.major) + "." + std::to_string(p.minor) +
                                 "); Dara needs a Turing-or-newer NVIDIA GPU (GTX 16xx, RTX 20/30/40/50 series)");
    return std::string(p.name) + " (sm_" + std::to_string(cc) + ", " + std::to_string(p.multiProcessorCount) + " SMs, " +
           std::to_string(p.totalGlobalMem >> 20) + " MiB)";
}

// Test hook: make the Nth device allocation from now fail (N = 1: the very next one), exactly as cudaMalloc running out of memory would.
// 0 (the default) disables it. Used by the allocation-failure tests; costs one relaxed atomic load per allocation.
inline std::atomic<long>& alloc_fault_countdown() { static std::atomic<long> c{0}; return c; }

// Borrowed device memory: while an arena is active, allocations are carved from its regions (memory that is allocated but holds nothing
// needed right now -- e.g. the unused tail of the K/V caches while an image is encoded) before falling back to cudaMalloc. Freeing an
// arena allocation is a no-op; everything allocated from an arena must be released before the arena is deactivated.
struct DeviceArena {
    struct Region { uint8_t* p; size_t size; size_t used; };
    std::vector<Region> regions;
    size_t borrowed = 0, peak = 0, fallback = 0;   // bytes taken from regions now / at most; bytes that had to come from cudaMalloc
    void* alloc(size_t bytes) {
        bytes = (bytes + 255) & ~size_t(255);
        Region* best = nullptr;   // best fit: the region with the least room left that still holds the request
        size_t best_room = ~size_t(0), best_off = 0;
        for (Region& r : regions) {
            const size_t off = (r.used + 255) & ~size_t(255);
            if (off + bytes > r.size) continue;
            const size_t room = r.size - off - bytes;
            if (room < best_room) { best = &r; best_room = room; best_off = off; }
        }
        if (!best) return nullptr;
        best->used = best_off + bytes;
        borrowed += bytes;
        peak = std::max(peak, borrowed);
        return best->p + best_off;
    }
    bool owns(const void* p) const {
        for (const Region& r : regions) if (p >= r.p && p < r.p + r.size) return true;
        return false;
    }
    size_t capacity() const { size_t s = 0; for (const Region& r : regions) s += r.size; return s; }
};
inline DeviceArena*& active_arena() { static DeviceArena* a = nullptr; return a; }
// Activates `a` for the lifetime of the guard (not nestable).
struct ArenaScope {
    explicit ArenaScope(DeviceArena& a) { active_arena() = &a; }
    ~ArenaScope() { active_arena() = nullptr; }
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;
};

// Raw counted allocation, freed by the deleter; used for matrices whose element type varies.
inline void* counted_malloc(size_t bytes) {
    {
        long c = alloc_fault_countdown().load(std::memory_order_relaxed);
        if (c > 0 && alloc_fault_countdown().fetch_sub(1) == 1) throw std::runtime_error("cudaMalloc: injected device allocation failure (out of memory)");
    }
    if (DeviceArena* a = active_arena()) {
        if (void* q = a->alloc(bytes)) return q;
        a->fallback += bytes;
    }
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
    device_stats().add(bytes);
    return p;
}
inline void counted_free(void* p, size_t bytes) {
    if (!p) return;
    if (DeviceArena* a = active_arena(); a && a->owns(p)) return;   // borrowed: released with the arena
    cudaFree(p);
    device_stats().sub(bytes);
}

template <class T> struct DevBuf {
    T* p = nullptr;
    size_t n = 0;
    DevBuf() = default;
    explicit DevBuf(size_t count) : n(count) {
        p = static_cast<T*>(counted_malloc(count * sizeof(T)));
        const cudaError_t e = cudaMemset(p, 0, count * sizeof(T));
        if (e != cudaSuccess) { counted_free(p, count * sizeof(T)); p = nullptr; n = 0; cuda_check(e, "cudaMemset"); }
    }
    explicit DevBuf(const std::vector<T>& v) : n(v.size()) {
        p = static_cast<T*>(counted_malloc(n * sizeof(T)));
        const cudaError_t e = cudaMemcpy(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice);
        if (e != cudaSuccess) { counted_free(p, n * sizeof(T)); p = nullptr; n = 0; cuda_check(e, "cudaMemcpy"); }
    }
    DevBuf(DevBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) { counted_free(p, n * sizeof(T)); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
        return *this;
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { counted_free(p, n * sizeof(T)); }
    std::vector<T> get() const {
        std::vector<T> v(n);
        cuda_check(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
        return v;
    }
    void put(const T* src, size_t count) { cuda_check(cudaMemcpy(p, src, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy"); }
};

// Reusable device scratch buffers, indexed by slot. A slot grows when asked for more than it holds and is otherwise reused,
// so a decode step does no cudaMalloc/cudaFree. Contents are NOT preserved across get() calls on the same slot.
class ScratchPool {
public:
    template <class T> T* get(size_t slot, size_t count) {
        if (slot >= slots_.size()) slots_.resize(slot + 1);
        const size_t bytes = count * sizeof(T);
        if (slots_[slot].n < bytes) slots_[slot] = DevBuf<uint8_t>(bytes);  // the old buffer is freed by the move
        return reinterpret_cast<T*>(slots_[slot].p);
    }
    size_t bytes() const { size_t b = 0; for (const auto& s : slots_) b += s.n; return b; }

private:
    std::vector<DevBuf<uint8_t>> slots_;
};

}  // namespace dara::model
