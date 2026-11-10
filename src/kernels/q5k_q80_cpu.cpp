// CPU references and simple quantizers for Q5_K and Q8_0.
#include <algorithm>
#include <cmath>

#include "kernels/half.h"
#include "kernels/q4k.h"  // q4k_scale_min: Q5_K uses the identical scale/min packing
#include "kernels/q5k.h"
#include "kernels/q8_0.h"

namespace dara::kernels {

// ---- Q5_K ------------------------------------------------------------------------------------

template <class Out>
static void q5k_dequant_impl(const BlockQ5K* b, size_t nblocks, Out* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const Out d = half_to_float(b[i].d);
        const Out mn = half_to_float(b[i].dmin);
        const uint8_t* ql = b[i].qs;
        const uint8_t* qh = b[i].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_scale_min(is + 0, b[i].scales, sc, m);
            const Out d1 = d * sc, m1 = mn * m;
            q4k_scale_min(is + 1, b[i].scales, sc, m);
            const Out d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
    }
}

void q5k_dequant(const BlockQ5K* b, size_t nblocks, float* out) { q5k_dequant_impl<float>(b, nblocks, out); }
void q5k_dequant_f64(const BlockQ5K* b, size_t nblocks, double* out) { q5k_dequant_impl<double>(b, nblocks, out); }

void q5k_quantize(const float* x, size_t cols, BlockQ5K* out) {
    for (size_t bi = 0; bi < cols / 256; ++bi) {
        const float* xb = x + bi * 256;
        float scale[8], minv[8], max_scale = 0, max_min = 0;
        for (int s = 0; s < 8; ++s) {
            float lo = 0.0f, hi = 0.0f;  // the offset is subtracted, so the range must include 0
            for (int k = 0; k < 32; ++k) { lo = std::min(lo, xb[s * 32 + k]); hi = std::max(hi, xb[s * 32 + k]); }
            scale[s] = (hi - lo) / 31.0f;
            minv[s] = -lo;
            max_scale = std::max(max_scale, scale[s]);
            max_min = std::max(max_min, minv[s]);
        }
        BlockQ5K& blk = out[bi];
        blk.d = float_to_half(max_scale / 63.0f);
        blk.dmin = float_to_half(max_min / 63.0f);
        const float d = half_to_float(blk.d), dmin = half_to_float(blk.dmin);
        uint8_t ls[8], lm[8];
        for (int s = 0; s < 8; ++s) {
            ls[s] = d > 0 ? static_cast<uint8_t>(std::min(63.0f, std::nearbyint(scale[s] / d))) : 0;
            lm[s] = dmin > 0 ? static_cast<uint8_t>(std::min(63.0f, std::nearbyint(minv[s] / dmin))) : 0;
        }
        std::fill(std::begin(blk.scales), std::end(blk.scales), uint8_t{0});
        for (int j = 0; j < 8; ++j) {
            if (j < 4) {
                blk.scales[j] = ls[j];
                blk.scales[j + 4] = lm[j];
            } else {
                blk.scales[j + 4] = static_cast<uint8_t>((ls[j] & 0xF) | ((lm[j] & 0xF) << 4));
                blk.scales[j - 4] |= static_cast<uint8_t>((ls[j] >> 4) << 6);
                blk.scales[j] |= static_cast<uint8_t>((lm[j] >> 4) << 6);
            }
        }
        std::fill(std::begin(blk.qh), std::end(blk.qh), uint8_t{0});
        for (int chunk = 0; chunk < 4; ++chunk) {
            for (int l = 0; l < 32; ++l) {
                uint8_t q[2];
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * chunk + h;
                    const float step = d * ls[s];
                    const float v = step > 0 ? (xb[64 * chunk + 32 * h + l] + dmin * lm[s]) / step : 0.0f;
                    q[h] = static_cast<uint8_t>(std::min(31.0f, std::max(0.0f, std::nearbyint(v))));
                }
                blk.qs[32 * chunk + l] = static_cast<uint8_t>((q[0] & 0xF) | ((q[1] & 0xF) << 4));
                blk.qh[l] = static_cast<uint8_t>(blk.qh[l] | ((q[0] >> 4) << (2 * chunk)) | ((q[1] >> 4) << (2 * chunk + 1)));
            }
        }
    }
}

// ---- Q8_0 ------------------------------------------------------------------------------------

void q80_dequant(const BlockQ80* b, size_t nblocks, float* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const float d = half_to_float(b[i].d);
        for (int j = 0; j < 32; ++j) *out++ = d * b[i].qs[j];
    }
}

void q80_dequant_f64(const BlockQ80* b, size_t nblocks, double* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const double d = half_to_float(b[i].d);
        for (int j = 0; j < 32; ++j) *out++ = d * b[i].qs[j];
    }
}

void q80_quantize(const float* x, size_t cols, BlockQ80* out) {
    for (size_t bi = 0; bi < cols / 32; ++bi) {
        const float* xb = x + bi * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
        out[bi].d = float_to_half(amax / 127.0f);
        const float d = half_to_float(out[bi].d);
        for (int j = 0; j < 32; ++j) {
            const float v = d > 0 ? std::nearbyint(xb[j] / d) : 0.0f;
            out[bi].qs[j] = static_cast<int8_t>(std::min(127.0f, std::max(-127.0f, v)));
        }
    }
}

}  // namespace dara::kernels
