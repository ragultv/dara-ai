#pragma once
// Host-side FP16 / BF16 <-> FP32 conversion, bit-exact and independent of any compiler's half type.
#include <cstdint>
#include <cstring>

namespace dara::kernels {

enum class WeightType { F16, BF16 };

inline float bf16_to_float(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t float_to_bf16(float f) {  // round to nearest even; NaN stays NaN
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ffu, u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else {  // subnormal: normalize
            int e = -1;
            do { ++e; man <<= 1; } while (!(man & 0x400u));
            u = sign | (static_cast<uint32_t>(112 - e) << 23) | ((man & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7f800000u | (man << 13);
    } else {
        u = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t float_to_half(float f) {  // round to nearest even, saturating to inf
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    uint32_t abs = u & 0x7fffffffu;
    if (abs > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);
    if (abs >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // rounds to >= 65520 -> inf
    if (abs < 0x38800000u) {  // subnormal or zero in half
        if (abs < 0x33000000u) return static_cast<uint16_t>(sign);
        uint32_t m = (abs & 0x7fffffu) | 0x800000u;
        int shift = 126 - static_cast<int>(abs >> 23);  // 14..24
        uint32_t r = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return static_cast<uint16_t>(sign | r);
    }
    uint32_t r = ((abs - 0x38000000u) >> 13);
    uint32_t rem = abs & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return static_cast<uint16_t>(sign | r);
}

inline float to_float(uint16_t h, WeightType t) { return t == WeightType::BF16 ? bf16_to_float(h) : half_to_float(h); }
inline uint16_t from_float(float f, WeightType t) { return t == WeightType::BF16 ? float_to_bf16(f) : float_to_half(f); }

}  // namespace dara::kernels
