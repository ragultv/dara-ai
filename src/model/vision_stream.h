#pragma once
// Streams the vision encoders' weight matrices from the memory-mapped mmproj file to the GPU while an image is encoded, instead of keeping them in
// VRAM. A matrix is cut into row chunks of at most kStreamSlotBytes; chunks go through two device slots in a fixed order (double buffering: the
// next chunk is copied by a second stream while the current one is used), so the encoder holds ~25 MB of weights at any moment instead of the whole
// mmproj (0.6-1 GB). The file's pages are file-backed (not private memory); the staging buffers are pinned host memory, allocated per image.
//
// Matrices are F16 [rows][cols] row-major in the file. The kernels read them as FP16 (tensor-core GPUs) or BF16 (the rest, rounded to nearest-even
// exactly as the resident upload does), zero-padded to [rows_pad][cols_pad]; the conversion and padding happen on the GPU, so the values -- and the
// encoder's output -- are bit-identical to the resident weights.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "model/device_buffer.h"
#include "model/layer_cuda.h"

namespace dara::model {

constexpr size_t kStreamSlotBytes = size_t{8} << 20;   // largest chunk (padded, in the kernel's 2-byte format)

struct StreamRef {                  // one weight matrix of the mmproj file
    const uint8_t* src = nullptr;   // F16 bits [rows][cols] (the mapped file)
    size_t rows = 0, cols = 0;
    size_t rows_pad = 0, cols_pad = 0;   // shape the kernels use (zero padded); >= rows / cols, cols_pad a multiple of 32
    bool valid() const { return src != nullptr; }
};

struct ChunkReq {                   // rows [r0, r1) of the padded matrix
    const StreamRef* ref = nullptr;
    size_t r0 = 0, r1 = 0;
};

// The chunks covering rows [r0, r1) of `w`, each at most kStreamSlotBytes in the padded 2-byte format.
std::vector<ChunkReq> chunks_for(const StreamRef& w, size_t r0, size_t r1);

class WeightStream {
public:
    // plan: every chunk the encoder will use, in order. f16: FP16 slots (tensor-core GEMM), else BF16.
    WeightStream(const std::vector<ChunkReq>& plan, bool f16);
    ~WeightStream();
    WeightStream(const WeightStream&) = delete;
    WeightStream& operator=(const WeightStream&) = delete;

    // The next planned chunk (must equal `expect`) as a matrix the matmul can read; the default stream waits until it has landed. Call done() once
    // the work reading it has been enqueued (the slot is then reloaded for a later chunk).
    const DeviceMatrix& next(const ChunkReq& expect);
    void done();

    // VRAM (device) and pinned host bytes a WeightStream takes
    static size_t device_bytes() { return 3 * kStreamSlotBytes + 4096; }
    static size_t host_bytes() { return 2 * kStreamSlotBytes; }

private:
    void issue(size_t j);   // starts loading planned chunk j
    void release();         // waits for the streams and frees events / pinned memory (idempotent)
    std::vector<ChunkReq> plan_;
    bool f16_;
    DevBuf<uint8_t> slot_[2], raw_;   // device: two slots (padded, 2-byte format) and the raw F16 staging
    uint8_t* pinned_[2] = {nullptr, nullptr};
    void* copy_stream_ = nullptr;
    void* loaded_[2] = {nullptr, nullptr};      // cudaEvent_t: the slot's chunk has landed
    void* used_[2] = {nullptr, nullptr};        // compute that read the slot has been enqueued
    void* h2d_[2] = {nullptr, nullptr};         // pinned buffer j % 2 has been copied out
    bool used_valid_[2] = {false, false}, h2d_valid_[2] = {false, false};
    size_t issued_ = 0, taken_ = 0;
    DeviceMatrix view_;
};

}  // namespace dara::model
