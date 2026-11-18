#pragma once
// 4-bit K/V cache kernels for the K8V4 and Q4 (KIVI-style) formats (layer_cuda.h: KvFormat).
//
// Values (both formats): per token, groups of 32 consecutive values of one head vector; codes c = round((x - min) / s) in 0..15 with
// s = (max - min) / 15, stored as nibbles [position][kv_width/2] (even index in the low nibble) plus a half2 (s, min) per group.
// Q4 keys: per CHANNEL over groups of kKiviGroup positions (keys have outlier channels; one scale per channel follows them), same nibble layout,
// half2 (s, min) at [position / 32][kv_width]. The newest positions (>= kKiviResidual, see kivi_frontier) stay in a Q8 ring of kKiviRing slots
// (slot = position % kKiviRing) until their whole group is old enough; the attention kernels read the ring for positions >= the frontier.
#include <cstddef>
#include <cstdint>

namespace dara::model {

// Values: T vectors x [T][kv_heads][hd] -> codes / (scale, min) at positions pos[t]. hd % 32 == 0.
int v4_append_cuda(const float* d_x, const int64_t* d_pos, int T, int kv_heads, int hd, uint8_t* d_codes, float* d_sc, void* stream = nullptr);

// Q4 keys: Q8 append into the ring (slot = pos % kKiviRing).
int kivi_ring_append_cuda(const float* d_x, const int64_t* d_pos, int T, int kv_heads, int hd, uint8_t* d_ring, float* d_ring_sc, void* stream = nullptr);

// Q4 keys, after appending positions d_pos[0] .. d_pos[T-1]: quantizes (from the ring) every group that this append moved behind the frontier.
// Device-side decision (valid inside CUDA graphs).
int kivi_quantize_cuda(const int64_t* d_pos, int T, int kv_width, const uint8_t* d_ring, const float* d_ring_sc, uint8_t* d_codes, float* d_ksc, void* stream = nullptr);

// Q4 keys: positions [p0, p1) (whole groups' worth, p0 % 32 == 0) are re-expanded from their 4-bit groups into the ring (host-side rewinds).
int kivi_unquantize_cuda(int p0, int p1, int kv_width, const uint8_t* d_codes, const float* d_ksc, uint8_t* d_ring, float* d_ring_sc, void* stream = nullptr);

// What the attention kernels need besides K / V / their scales.
struct KvAux {
    const int8_t* ring = nullptr;      // Q4: key ring codes
    const float* ring_sc = nullptr;    //     and scales
    const int64_t* pend = nullptr;     // Q4: device pointer to the last appended position of this call (frontier = kivi_frontier(*pend + 1)), or
    int held = 0;                      //     null and `held` = positions held (host-side calls)
    int kv_width = 0;                  //     kv_heads * hd
    size_t base = 0;                   //     (prefill GEMM) element offset of the key pointer handed to the kernel
};

}  // namespace dara::model
