// Image input for the vision encoders: decode (PNG, JPEG, BMP, GIF, TIFF via Windows Imaging Component) to 8-bit RGB, and resize with
// the same arithmetic as Pillow's Image.resize(BICUBIC) (the reference image processors use Pillow), so preprocessing matches exactly.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dara::io {

struct RgbImage {
    int width = 0, height = 0;
    std::vector<uint8_t> px;   // [height][width][3], R G B
};

// Throws std::runtime_error when the file cannot be read or decoded.
RgbImage load_image_rgb(const std::string& path);

// Pillow-compatible separable bicubic resampling (a = -0.5, support scaled when shrinking, 22-bit fixed-point coefficients, horizontal
// pass first with 8-bit rounding between the passes).
RgbImage resize_bicubic(const RgbImage& in, int width, int height);

}  // namespace dara::io
