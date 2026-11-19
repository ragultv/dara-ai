#include "model/vision_stream.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace dara::model {
namespace {

// Same rounding as dara::kernels::float_to_bf16 (the resident upload): round to nearest even, NaN stays NaN.
__device__ __forceinline__ uint16_t float_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// out[r][c] (r < out_rows, c < cols_pad) = (r < valid_rows && c < cols) ? weight(src[r][c]) : 0, where src holds the chunk's valid rows contiguously
// ([valid_rows][cols] F16). FP16 target: the bits unchanged; BF16 target: bf16(half -> float).
__global__ void pad_convert_kernel(const uint16_t* __restrict__ src, uint16_t* __restrict__ out, size_t out_rows, size_t cols, size_t cols_pad, size_t valid_rows,
                                   bool bf16_target) {
    const size_t total = out_rows * cols_pad;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = i / cols_pad, c = i - r * cols_pad;
        uint16_t v = 0;
        if (r < valid_rows && c < cols) {
            const uint16_t h = src[r * cols + c];
            v = bf16_target ? float_to_bf16_bits(__half2float(__ushort_as_half(h))) : h;
        }
        out[i] = v;
    }
}

cudaEvent_t make_event() {
    cudaEvent_t e;
    cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "vision stream event");
    return e;
}

}  // namespace

std::vector<ChunkReq> chunks_for(const StreamRef& w, size_t r0, size_t r1) {
    std::vector<ChunkReq> v;
    const size_t row_bytes = w.cols_pad * 2;
    if (row_bytes == 0 || row_bytes > kStreamSlotBytes) throw std::invalid_argument("vision weight row does not fit a stream slot");
    size_t per = kStreamSlotBytes / row_bytes;
    if (per >= 128) per = per / 128 * 128;   // whole GEMM row tiles where possible
    for (size_t r = r0; r < r1; r += per) v.push_back({&w, r, std::min(r1, r + per)});
    return v;
}

WeightStream::WeightStream(const std::vector<ChunkReq>& plan, bool f16) : plan_(plan), f16_(f16) {
    try {
        slot_[0] = DevBuf<uint8_t>(kStreamSlotBytes);
        slot_[1] = DevBuf<uint8_t>(kStreamSlotBytes);
        raw_ = DevBuf<uint8_t>(kStreamSlotBytes);
        for (int i = 0; i < 2; ++i) {
            void* p = nullptr;
            if (cudaHostAlloc(&p, kStreamSlotBytes, cudaHostAllocDefault) != cudaSuccess) { cudaGetLastError(); throw std::runtime_error("cannot allocate pinned staging memory"); }
            pinned_[i] = static_cast<uint8_t*>(p);
        }
        cudaStream_t s;
        cuda_check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "vision stream");
        copy_stream_ = s;
        for (int i = 0; i < 2; ++i) { loaded_[i] = make_event(); used_[i] = make_event(); h2d_[i] = make_event(); }
    } catch (...) {
        release();
        throw;
    }
}

WeightStream::~WeightStream() { release(); }

void WeightStream::release() {
    if (copy_stream_) cudaStreamSynchronize(static_cast<cudaStream_t>(copy_stream_));
    cudaStreamSynchronize(nullptr);
    for (int i = 0; i < 2; ++i) {
        if (loaded_[i]) cudaEventDestroy(static_cast<cudaEvent_t>(loaded_[i]));
        if (used_[i]) cudaEventDestroy(static_cast<cudaEvent_t>(used_[i]));
        if (h2d_[i]) cudaEventDestroy(static_cast<cudaEvent_t>(h2d_[i]));
        if (pinned_[i]) cudaFreeHost(pinned_[i]);
        loaded_[i] = used_[i] = h2d_[i] = nullptr;
        pinned_[i] = nullptr;
    }
    if (copy_stream_) cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream_));
    copy_stream_ = nullptr;
}

void WeightStream::issue(size_t j) {
    const ChunkReq& c = plan_[j];
    const StreamRef& w = *c.ref;
    const int s = static_cast<int>(j % 2);
    auto cs = static_cast<cudaStream_t>(copy_stream_);
    const size_t valid_rows = c.r0 >= w.rows ? 0 : std::min(c.r1, w.rows) - c.r0;
    const size_t raw_bytes = valid_rows * w.cols * 2;
    // the pinned buffer must have been copied out (chunk j - 2) before the host overwrites it
    if (h2d_valid_[s]) cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(h2d_[s])), "vision stream pinned buffer");
    if (raw_bytes) std::memcpy(pinned_[s], w.src + c.r0 * w.cols * 2, raw_bytes);
    // the slot is free once the compute that read chunk j - 2 has finished
    if (used_valid_[s]) cuda_check(cudaStreamWaitEvent(cs, static_cast<cudaEvent_t>(used_[s]), 0), "vision stream wait");
    if (raw_bytes) cuda_check(cudaMemcpyAsync(raw_.p, pinned_[s], raw_bytes, cudaMemcpyHostToDevice, cs), "vision stream copy");
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(h2d_[s]), cs), "vision stream event");
    h2d_valid_[s] = true;
    const size_t out_rows = c.r1 - c.r0;
    const size_t total = out_rows * w.cols_pad;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>((total + 255) / 256, 4096));
    pad_convert_kernel<<<blocks, 256, 0, cs>>>(reinterpret_cast<const uint16_t*>(raw_.p), reinterpret_cast<uint16_t*>(slot_[s].p), out_rows, w.cols, w.cols_pad, valid_rows, !f16_);
    cuda_check(cudaGetLastError(), "vision stream convert");
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(loaded_[s]), cs), "vision stream event");
}

const DeviceMatrix& WeightStream::next(const ChunkReq& expect) {
    if (taken_ >= plan_.size()) throw std::logic_error("vision weight stream: more chunks requested than planned");
    const ChunkReq& c = plan_[taken_];
    if (c.ref != expect.ref || c.r0 != expect.r0 || c.r1 != expect.r1) throw std::logic_error("vision weight stream: chunk order differs from the plan");
    // keep the next two chunks in flight (the second needs the slot of chunk taken_ - 1, released by done())
    while (issued_ < plan_.size() && issued_ < taken_ + 2) {
        const size_t j = issued_;
        if (j >= 2 && j - 2 >= taken_) break;   // its slot is still in use by an unfinished chunk
        issue(j);
        ++issued_;
    }
    const int s = static_cast<int>(taken_ % 2);
    cuda_check(cudaStreamWaitEvent(nullptr, static_cast<cudaEvent_t>(loaded_[s]), 0), "vision stream wait");
    view_ = DeviceMatrix{};
    view_.f16 = f16_;
    view_.rows = c.r1 - c.r0;
    view_.cols = c.ref->cols_pad;
    view_.ld = c.ref->cols_pad;
    view_.bytes = view_.rows * view_.cols * 2;
    view_.data = std::shared_ptr<void>(slot_[s].p, [](void*) {});
    return view_;
}

void WeightStream::done() {
    const int s = static_cast<int>(taken_ % 2);
    cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(used_[s]), nullptr), "vision stream event");
    used_valid_[s] = true;
    ++taken_;
    // chunk taken_ + 1 may now take over this slot
    while (issued_ < plan_.size() && issued_ < taken_ + 2) {
        if (issued_ >= 2 && issued_ - 2 >= taken_) break;
        issue(issued_);
        ++issued_;
    }
}

}  // namespace dara::model
