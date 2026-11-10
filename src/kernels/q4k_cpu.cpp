#include "kernels/q4k.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/half.h"

namespace dara::kernels {

void q4k_dequant(const BlockQ4K* b, size_t nblocks, float* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const float d = half_to_float(b[i].d);
        const float mn = half_to_float(b[i].dmin);
        const uint8_t* q = b[i].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_scale_min(is + 0, b[i].scales, sc, m);
            const float d1 = d * sc, m1 = mn * m;
            q4k_scale_min(is + 1, b[i].scales, sc, m);
            const float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void q4k_dequant_f64(const BlockQ4K* b, size_t nblocks, double* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const double d = half_to_float(b[i].d);
        const double mn = half_to_float(b[i].dmin);
        const uint8_t* q = b[i].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            q4k_scale_min(is + 0, b[i].scales, sc, m);
            const double d1 = d * sc, m1 = mn * m;
            q4k_scale_min(is + 1, b[i].scales, sc, m);
            const double d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void q4k_quantize(const float* x, size_t cols, BlockQ4K* out) {
    for (size_t bi = 0; bi < cols / 256; ++bi) {
        const float* xb = x + bi * 256;
        float scale[8], minv[8];  // per sub-block: step size and (non-negative) offset
        float max_scale = 0, max_min = 0;
        for (int s = 0; s < 8; ++s) {
            float lo = 0.0f, hi = 0.0f;  // the offset is subtracted, so the range must include 0
            for (int k = 0; k < 32; ++k) { lo = std::min(lo, xb[s * 32 + k]); hi = std::max(hi, xb[s * 32 + k]); }
            scale[s] = (hi - lo) / 15.0f;
            minv[s] = -lo;
            max_scale = std::max(max_scale, scale[s]);
            max_min = std::max(max_min, minv[s]);
        }
        BlockQ4K& blk = out[bi];
        blk.d = float_to_half(max_scale / 63.0f);
        blk.dmin = float_to_half(max_min / 63.0f);
        const float d = half_to_float(blk.d), dmin = half_to_float(blk.dmin);
        uint8_t ls[8], lm[8];
        for (int s = 0; s < 8; ++s) {
            ls[s] = d > 0 ? static_cast<uint8_t>(std::min(63.0f, std::nearbyint(scale[s] / d))) : 0;
            lm[s] = dmin > 0 ? static_cast<uint8_t>(std::min(63.0f, std::nearbyint(minv[s] / dmin))) : 0;
        }
        for (int j = 0; j < 12; ++j) blk.scales[j] = 0;
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
        uint8_t* q = blk.qs;
        for (int chunk = 0; chunk < 4; ++chunk) {
            for (int l = 0; l < 32; ++l) {
                uint8_t nib[2];
                for (int h = 0; h < 2; ++h) {
                    const int s = 2 * chunk + h;
                    const float step = d * ls[s];
                    float v = step > 0 ? (xb[64 * chunk + 32 * h + l] + dmin * lm[s]) / step : 0.0f;
                    nib[h] = static_cast<uint8_t>(std::min(15.0f, std::max(0.0f, std::nearbyint(v))));
                }
                q[l] = static_cast<uint8_t>(nib[0] | (nib[1] << 4));
            }
            q += 32;
        }
    }
}

void gemv_q4k_cpu_f32(const BlockQ4K* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols) {
    const size_t nb = cols / 256;
    std::vector<float> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        q4k_dequant(W + r * ld_blocks, nb, w.data());
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) acc += w[c] * x[c];
        y[r] = acc;
    }
}

void gemv_q4k_cpu_f64(const BlockQ4K* W, size_t ld_blocks, const float* x, double* y, double* sum_abs, size_t rows,
                      size_t cols) {
    const size_t nb = cols / 256;
    std::vector<double> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        q4k_dequant_f64(W + r * ld_blocks, nb, w.data());
        double acc = 0.0, abs_acc = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            const double t = w[c] * static_cast<double>(x[c]);
            acc += t;
            abs_acc += t < 0 ? -t : t;
        }
        y[r] = acc;
        if (sum_abs) sum_abs[r] = abs_acc;
    }
}

}  // namespace dara::kernels
