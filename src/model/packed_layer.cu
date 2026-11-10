#include "model/packed_layer.h"

#include <cstdlib>
#include <new>
#include <stdexcept>

#include <cuda_runtime.h>

namespace dara::model {

HostBuffer::HostBuffer(size_t bytes) : n_(bytes) {
    if (bytes == 0) return;
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess) {
        p_ = static_cast<uint8_t*>(p);
        pinned_ = true;
    } else {
        cudaGetLastError();  // clear the failed pinned allocation's error and fall back to pageable memory
        p_ = static_cast<uint8_t*>(std::malloc(bytes));
        if (!p_) throw std::bad_alloc();
    }
}

void HostBuffer::release() {
    if (p_) { if (pinned_) cudaFreeHost(p_); else std::free(p_); }
    p_ = nullptr; n_ = 0; pinned_ = false;
}
HostBuffer::~HostBuffer() { release(); }
HostBuffer::HostBuffer(HostBuffer&& o) noexcept : p_(o.p_), n_(o.n_), pinned_(o.pinned_) { o.p_ = nullptr; o.n_ = 0; o.pinned_ = false; }
HostBuffer& HostBuffer::operator=(HostBuffer&& o) noexcept {
    if (this != &o) { release(); p_ = o.p_; n_ = o.n_; pinned_ = o.pinned_; o.p_ = nullptr; o.n_ = 0; o.pinned_ = false; }
    return *this;
}

void bind_matrices(PackedLayer& p, const std::shared_ptr<void>& owner, uint8_t* base) {
    for (const MatrixSlot& s : p.slots) {
        DeviceMatrix m;
        m.quantized = true;
        m.qtype = s.qtype;
        m.data = std::shared_ptr<void>(owner, base + s.offset);  // aliasing: shares the owner's lifetime, points inside it
        m.rows = s.rows;
        m.cols = s.cols;
        m.ld = s.cols / dara::kernels::quant_format(s.qtype).block_weights;
        m.bytes = s.bytes;
        if (s.index >= 0) p.mats.at(static_cast<size_t>(s.index)) = std::move(m);
        else p.weights.*(s.member) = std::move(m);
    }
}

void make_resident(PackedLayer& p) {
    if (p.resident) return;
    if (p.host.size() < p.matrix_bytes) throw std::logic_error("packed layer has no host image to upload");
    uint8_t* d = static_cast<uint8_t*>(counted_malloc(p.matrix_bytes));
    const size_t bytes = p.matrix_bytes;
    std::shared_ptr<void> slab(d, [bytes](void* q) { counted_free(q, bytes); });
    cuda_check(cudaMemcpy(d, p.host.data(), p.matrix_bytes, cudaMemcpyHostToDevice), "upload layer slab");
    bind_matrices(p, slab, d);
    p.host.release();  // the device copy is the only one needed from now on
    p.resident = true;
}

}  // namespace dara::model
