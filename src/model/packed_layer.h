#pragma once
// A layer's weights as one contiguous host image, so a layer can live in host RAM and be moved to the GPU with a single copy.
//
//   PackedLayer = GpuLayerWeights (the small parameters -- norms, conv, gates, rope table -- always resident on the device)
//               + a host buffer holding every projection matrix back to back (256-byte aligned), described by `slots`.
// The matrices are NOT device-resident until bound: bind_matrices() points the layer's DeviceMatrix members at device memory
// (a permanent slab for VRAM-resident layers, a reusable staging arena for streamed ones). The same PackedLayer therefore
// serves both placements, and no disk is touched after it was read once.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/quant.h"
#include "model/layer_cuda.h"

namespace dara::model {

// Host memory, page-locked (pinned) when the OS allows it so host-to-device copies are fast and can overlap with compute;
// ordinary pageable memory otherwise (reported via pinned()).
class HostBuffer {
public:
    HostBuffer() = default;
    explicit HostBuffer(size_t bytes);
    ~HostBuffer();
    HostBuffer(HostBuffer&& o) noexcept;
    HostBuffer& operator=(HostBuffer&& o) noexcept;
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    uint8_t* data() const { return p_; }
    size_t size() const { return n_; }
    bool pinned() const { return pinned_; }
    void release();

private:
    uint8_t* p_ = nullptr;
    size_t n_ = 0;
    bool pinned_ = false;
};

struct MatrixSlot {
    DeviceMatrix GpuLayerWeights::* member;  // which matrix of the layer this is (Qwen layers); null for indexed slots
    dara::kernels::QuantType qtype;
    size_t rows = 0, cols = 0;
    size_t offset = 0, bytes = 0;            // position inside the packed image
    int index = -1;                          // >= 0: bind into PackedLayer::mats[index] instead (other architectures, e.g. Gemma)
};

struct PackedLayer {
    GpuLayerWeights weights;   // small parameters resident on the device; matrices bound by bind_matrices()
    std::vector<MatrixSlot> slots;
    std::vector<DeviceMatrix> mats;  // targets of indexed slots (sized by the architecture's loader)
    HostBuffer host;           // packed matrices; released once uploaded to a resident slab
    size_t matrix_bytes = 0;   // size of the packed image
    bool resident = false;
    // Device bytes of the always-resident small parameters (norms, conv, gates, rope table); excludes the matrices.
    size_t small_param_bytes() const {
        return (weights.input_norm.n + weights.post_norm.n + weights.conv_w.n + weights.ssm_a.n + weights.dt_bias.n + weights.norm_w.n +
                weights.q_norm.n + weights.k_norm.n + weights.inv_freq.n) * sizeof(float);
    }
};

constexpr size_t kPackAlign = 256;
inline size_t align_up(size_t v, size_t a = kPackAlign) { return (v + a - 1) / a * a; }

// Points every matrix of `p.weights` at device memory: matrix i is at base + slot.offset. `owner` keeps the memory alive for as
// long as any matrix refers to it (pass a no-op owner for a reusable arena you manage yourself).
void bind_matrices(PackedLayer& p, const std::shared_ptr<void>& owner, uint8_t* base);

// Uploads the packed image into a new permanent device slab, binds the matrices to it and frees the host copy.
void make_resident(PackedLayer& p);

}  // namespace dara::model
