#include "kernels/q6k.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kernels/half.h"

namespace dara::kernels {

void q6k_dequant(const BlockQ6K* b, size_t nblocks, float* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const float d = half_to_float(b[i].d);
        const uint8_t* ql = b[i].ql;
        const uint8_t* qh = b[i].qh;
        const int8_t* sc = b[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l + 0] = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void q6k_dequant_f64(const BlockQ6K* b, size_t nblocks, double* out) {
    for (size_t i = 0; i < nblocks; ++i) {
        const double d = half_to_float(b[i].d);
        const uint8_t* ql = b[i].ql;
        const uint8_t* qh = b[i].qh;
        const int8_t* sc = b[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                out[l + 0] = d * sc[is + 0] * q1;
                out[l + 32] = d * sc[is + 2] * q2;
                out[l + 64] = d * sc[is + 4] * q3;
                out[l + 96] = d * sc[is + 6] * q4;
            }
            out += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void q6k_quantize(const float* x, size_t cols, BlockQ6K* out) {
    for (size_t bi = 0; bi < cols / 256; ++bi) {
        const float* xb = x + bi * 256;
        BlockQ6K& blk = out[bi];
        // Output weight w uses scales[w / 16] (16 consecutive weights per sub-block) -- see q6k.h.
        float sub_max[16], gmax = 0.0f;
        for (int g = 0; g < 16; ++g) {
            float m = 0.0f;
            for (int k = 0; k < 16; ++k) m = std::max(m, std::fabs(xb[g * 16 + k]));
            sub_max[g] = m;
            gmax = std::max(gmax, m);
        }
        // q - 32 spans [-32, 31]; per-sub-block step = max/31; scales are int8 multiples of d.
        blk.d = float_to_half(gmax / 31.0f / 127.0f);
        const float d = half_to_float(blk.d);
        float step[16];
        for (int g = 0; g < 16; ++g) {
            const int ls = d > 0 ? static_cast<int>(std::min(127.0f, std::nearbyint(sub_max[g] / 31.0f / d))) : 0;
            blk.scales[g] = static_cast<int8_t>(ls);
            step[g] = d * ls;
        }
        std::fill(std::begin(blk.ql), std::end(blk.ql), uint8_t{0});
        std::fill(std::begin(blk.qh), std::end(blk.qh), uint8_t{0});
        for (int w = 0; w < 256; ++w) {
            const int n = w / 128, k = (w % 128) / 32, l = w % 32;  // half, which of y[l + 32k], position
            const float st = step[w / 16];
            const int v = st > 0 ? static_cast<int>(std::nearbyint(xb[w] / st)) : 0;
            const int q = std::min(31, std::max(-32, v)) + 32;       // 0..63
            const int lo = q & 0xF, hi = q >> 4;
            const int qlpos = 64 * n + l + 32 * (k & 1);
            blk.ql[qlpos] = static_cast<uint8_t>(blk.ql[qlpos] | (k < 2 ? lo : lo << 4));
            blk.qh[32 * n + l] = static_cast<uint8_t>(blk.qh[32 * n + l] | (hi << (2 * k)));
        }
    }
}

void gemv_q6k_cpu_f32(const BlockQ6K* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols) {
    const size_t nb = cols / 256;
    std::vector<float> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        q6k_dequant(W + r * ld_blocks, nb, w.data());
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) acc += w[c] * x[c];
        y[r] = acc;
    }
}

void gemv_q6k_cpu_f64(const BlockQ6K* W, size_t ld_blocks, const float* x, double* y, double* sum_abs, size_t rows,
                      size_t cols) {
    const size_t nb = cols / 256;
    std::vector<double> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        q6k_dequant_f64(W + r * ld_blocks, nb, w.data());
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
