#include "model/ops.h"

#include <cmath>

namespace dara::model {

float rmsnorm_inv_rms(const float* x, int n, float eps) {
    double ms = 0.0;
    for (int i = 0; i < n; ++i) ms += static_cast<double>(x[i]) * x[i];
    return static_cast<float>(1.0 / std::sqrt(ms / n + eps));
}

void rmsnorm(const float* x, const float* scale, int n, float eps, float* y) {
    const float r = rmsnorm_inv_rms(x, n, eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * scale[i];
}

std::vector<float> fold_plus_one(const std::vector<float>& w) {
    std::vector<float> s(w.size());
    for (size_t i = 0; i < w.size(); ++i) s[i] = w[i] + 1.0f;
    return s;
}

// Mirrors the reference: 1.0 / (base ** (arange(0, dim, 2).float() / dim)) with every step in float32.
std::vector<float> rope_inv_freq(int rot_dim, double theta) {
    std::vector<float> f(rot_dim / 2);
    for (int i = 0; i < rot_dim / 2; ++i) {
        const float expo = static_cast<float>(2 * i) / static_cast<float>(rot_dim);
        f[i] = 1.0f / std::pow(static_cast<float>(theta), expo);
    }
    return f;
}

// Correctly rounded table (double pow, rounded once). Differs from the reference by 1 ulp in many entries.
std::vector<float> rope_inv_freq_exact(int rot_dim, double theta) {
    std::vector<float> f(rot_dim / 2);
    for (int i = 0; i < rot_dim / 2; ++i) f[i] = static_cast<float>(1.0 / std::pow(theta, static_cast<double>(2 * i) / rot_dim));
    return f;
}

void rope_cos_sin(int64_t pos, const std::vector<float>& inv_freq, float* cos_out, float* sin_out) {
    for (size_t i = 0; i < inv_freq.size(); ++i) {
        const float a = rope_angle_fp32(pos, inv_freq[i]);
        cos_out[i] = std::cos(a);
        sin_out[i] = std::sin(a);
    }
}

void rope_apply(float* x, int heads, int head_dim, int rot_dim, const float* cos, const float* sin) {
    const int half = rot_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float* p = x + static_cast<size_t>(h) * head_dim;
        for (int i = 0; i < half; ++i) {
            const float a = p[i], b = p[i + half];
            p[i] = a * cos[i] - b * sin[i];
            p[i + half] = b * cos[i] + a * sin[i];
        }
    }
}

}  // namespace dara::model
