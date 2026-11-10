#include "kernels/gemv.h"

namespace dara::kernels {

void gemv_cpu_f32(WeightType t, const uint16_t* W, size_t ld, const float* x, float* y, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t* w = W + r * ld;
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) acc += to_float(w[c], t) * x[c];
        y[r] = acc;
    }
}

void gemv_cpu_f64(WeightType t, const uint16_t* W, size_t ld, const float* x, double* y, double* sum_abs,
                  size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        const uint16_t* w = W + r * ld;
        double acc = 0.0, abs_acc = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            double term = static_cast<double>(to_float(w[c], t)) * static_cast<double>(x[c]);
            acc += term;
            abs_acc += term < 0 ? -term : term;
        }
        y[r] = acc;
        if (sum_abs) sum_abs[r] = abs_acc;
    }
}

}  // namespace dara::kernels
