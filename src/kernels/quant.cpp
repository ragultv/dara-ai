#include "kernels/quant.h"

#include <stdexcept>
#include <vector>

#include "kernels/half.h"
#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/q6k.h"
#include "kernels/q8_0.h"

namespace dara::kernels {
namespace {

template <class B> const B* as(const void* p) { return static_cast<const B*>(p); }
template <class B> B* as(void* p) { return static_cast<B*>(p); }

const QuantFormat kFormats[] = {
    {QuantType::Q4_K, "Q4_K", kQ4KBlockWeights, kQ4KBlockBytes, 16,
     [](const void* b, size_t n, float* o) { q4k_dequant(as<BlockQ4K>(b), n, o); },
     [](const void* b, size_t n, double* o) { q4k_dequant_f64(as<BlockQ4K>(b), n, o); },
     [](const float* x, size_t c, void* o) { q4k_quantize(x, c, as<BlockQ4K>(o)); },
     [](void* b, float d, float dmin) { as<BlockQ4K>(b)->d = float_to_half(d); as<BlockQ4K>(b)->dmin = float_to_half(dmin); }},
    {QuantType::Q5_K, "Q5_K", kQ5KBlockWeights, kQ5KBlockBytes, 16,
     [](const void* b, size_t n, float* o) { q5k_dequant(as<BlockQ5K>(b), n, o); },
     [](const void* b, size_t n, double* o) { q5k_dequant_f64(as<BlockQ5K>(b), n, o); },
     [](const float* x, size_t c, void* o) { q5k_quantize(x, c, as<BlockQ5K>(o)); },
     [](void* b, float d, float dmin) { as<BlockQ5K>(b)->d = float_to_half(d); as<BlockQ5K>(b)->dmin = float_to_half(dmin); }},
    {QuantType::Q6_K, "Q6_K", kQ6KBlockWeights, kQ6KBlockBytes, 2,
     [](const void* b, size_t n, float* o) { q6k_dequant(as<BlockQ6K>(b), n, o); },
     [](const void* b, size_t n, double* o) { q6k_dequant_f64(as<BlockQ6K>(b), n, o); },
     [](const float* x, size_t c, void* o) { q6k_quantize(x, c, as<BlockQ6K>(o)); },
     [](void* b, float d, float) { as<BlockQ6K>(b)->d = float_to_half(d); }},
    {QuantType::Q8_0, "Q8_0", kQ80BlockWeights, kQ80BlockBytes, 2,
     [](const void* b, size_t n, float* o) { q80_dequant(as<BlockQ80>(b), n, o); },
     [](const void* b, size_t n, double* o) { q80_dequant_f64(as<BlockQ80>(b), n, o); },
     [](const float* x, size_t c, void* o) { q80_quantize(x, c, as<BlockQ80>(o)); },
     [](void* b, float d, float) { as<BlockQ80>(b)->d = float_to_half(d); }},
};

}  // namespace

const QuantFormat& quant_format(QuantType t) {
    for (auto& f : kFormats)
        if (f.type == t) return f;
    throw std::invalid_argument("unknown QuantType");
}

bool quant_type_from_name(const std::string& n, QuantType& out) {
    for (auto& f : kFormats)
        if (n == f.name) { out = f.type; return true; }
    return false;
}

void gemv_quant_cpu_f32(QuantType t, const void* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols) {
    const QuantFormat& f = quant_format(t);
    const size_t nb = cols / f.block_weights;
    std::vector<float> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        f.dequant_f32(static_cast<const uint8_t*>(W) + r * ld_blocks * f.block_bytes, nb, w.data());
        float acc = 0.0f;
        for (size_t c = 0; c < cols; ++c) acc += w[c] * x[c];
        y[r] = acc;
    }
}

void gemv_quant_cpu_f64(QuantType t, const void* W, size_t ld_blocks, const float* x, double* y, double* sum_abs,
                        size_t rows, size_t cols) {
    const QuantFormat& f = quant_format(t);
    const size_t nb = cols / f.block_weights;
    std::vector<double> w(cols);
    for (size_t r = 0; r < rows; ++r) {
        f.dequant_f64(static_cast<const uint8_t*>(W) + r * ld_blocks * f.block_bytes, nb, w.data());
        double acc = 0.0, abs_acc = 0.0;
        for (size_t c = 0; c < cols; ++c) {
            const double term = w[c] * static_cast<double>(x[c]);
            acc += term;
            abs_acc += term < 0 ? -term : term;
        }
        y[r] = acc;
        if (sum_abs) sum_abs[r] = abs_acc;
    }
}

}  // namespace dara::kernels
