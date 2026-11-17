#include "io/image.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif

namespace dara::io {

#ifdef _WIN32
namespace {
template <class T> struct Com {
    T* p = nullptr;
    ~Com() { if (p) p->Release(); }
    T** operator&() { return &p; }
    T* operator->() const { return p; }
};
void hr_check(HRESULT hr, const char* what, const std::string& path) {
    if (FAILED(hr)) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "0x%08lx", static_cast<unsigned long>(hr));
        throw std::runtime_error(std::string("image ") + what + " failed (" + buf + "): " + path);
    }
}
}  // namespace

RgbImage load_image_rgb(const std::string& path) {
    const HRESULT ci = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninit = SUCCEEDED(ci);
    struct Uninit { bool on; ~Uninit() { if (on) CoUninitialize(); } } guard{uninit};
    const int wn = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::wstring wpath(static_cast<size_t>(std::max(wn, 1)), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn);
    RgbImage img;
    {
        Com<IWICImagingFactory> factory;
        hr_check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "WIC factory", path);
        Com<IWICBitmapDecoder> dec;
        hr_check(factory->CreateDecoderFromFilename(wpath.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &dec), "open", path);
        Com<IWICBitmapFrameDecode> frame;
        hr_check(dec->GetFrame(0, &frame), "decode", path);
        Com<IWICBitmapSource> rgb;
        hr_check(WICConvertBitmapSource(GUID_WICPixelFormat24bppRGB, frame.p, &rgb), "convert to RGB", path);
        UINT w = 0, h = 0;
        hr_check(rgb->GetSize(&w, &h), "size", path);
        if (w == 0 || h == 0 || w > 32768 || h > 32768) throw std::runtime_error("image has unsupported dimensions: " + path);
        img.width = static_cast<int>(w);
        img.height = static_cast<int>(h);
        img.px.resize(static_cast<size_t>(w) * h * 3);
        const UINT stride = w * 3;
        hr_check(rgb->CopyPixels(nullptr, stride, static_cast<UINT>(img.px.size()), img.px.data()), "copy pixels", path);
    }
    return img;
}
#else
RgbImage load_image_rgb(const std::string& path) { throw std::runtime_error("image decoding is only implemented on Windows: " + path); }
#endif

namespace {
constexpr int kPrecisionBits = 32 - 8 - 2;   // Pillow: PRECISION_BITS

double bicubic(double x) {   // Pillow's bicubic_filter, a = -0.5
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

// Pillow's precompute_coeffs + normalize_coeffs_8bpc for one axis.
int coeffs(int in_size, int out_size, std::vector<int>& bounds, std::vector<int32_t>& kk) {
    const double scale = static_cast<double>(in_size) / out_size;
    const double filterscale = std::max(scale, 1.0);
    const double support = 2.0 * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    std::vector<double> pre(static_cast<size_t>(out_size) * ksize, 0.0);
    bounds.assign(static_cast<size_t>(out_size) * 2, 0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double* k = &pre[static_cast<size_t>(xx) * ksize];
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) { const double w = bicubic((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (int x = 0; x < xmax; ++x) if (ww != 0.0) k[x] /= ww;
        bounds[static_cast<size_t>(xx) * 2] = xmin;
        bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    kk.resize(pre.size());
    for (size_t i = 0; i < pre.size(); ++i)
        kk[i] = pre[i] < 0 ? static_cast<int32_t>(-0.5 + pre[i] * (1 << kPrecisionBits)) : static_cast<int32_t>(0.5 + pre[i] * (1 << kPrecisionBits));
    return ksize;
}

inline uint8_t clip8(int64_t in) {
    if (in >= (static_cast<int64_t>(256) << kPrecisionBits)) return 255;
    if (in <= 0) return 0;
    return static_cast<uint8_t>(in >> kPrecisionBits);
}
}  // namespace

RgbImage resize_bicubic(const RgbImage& in, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("resize to an empty image");
    if (width == in.width && height == in.height) return in;
    std::vector<int> bh, bv;
    std::vector<int32_t> kh, kv;
    const int ksh = coeffs(in.width, width, bh, kh);
    const int ksv = coeffs(in.height, height, bv, kv);
    // horizontal pass over the source rows the vertical pass needs (as Pillow), 8-bit result
    const int y_first = bv[0], y_last = bv[static_cast<size_t>(height) * 2 - 2] + bv[static_cast<size_t>(height) * 2 - 1];
    const bool need_h = width != in.width, need_v = height != in.height;
    RgbImage tmp;
    const RgbImage* src = &in;
    int row_off = 0;
    if (need_h) {
        tmp.width = width;
        tmp.height = y_last - y_first;
        tmp.px.resize(static_cast<size_t>(tmp.width) * tmp.height * 3);
        for (int yy = 0; yy < tmp.height; ++yy) {
            const uint8_t* row = &in.px[static_cast<size_t>(yy + y_first) * in.width * 3];
            for (int xx = 0; xx < width; ++xx) {
                const int xmin = bh[static_cast<size_t>(xx) * 2], xmax = bh[static_cast<size_t>(xx) * 2 + 1];
                const int32_t* k = &kh[static_cast<size_t>(xx) * ksh];
                int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xmax; ++x) {
                    const uint8_t* p = row + static_cast<size_t>(x + xmin) * 3;
                    s0 += static_cast<int64_t>(p[0]) * k[x]; s1 += static_cast<int64_t>(p[1]) * k[x]; s2 += static_cast<int64_t>(p[2]) * k[x];
                }
                uint8_t* o = &tmp.px[(static_cast<size_t>(yy) * width + xx) * 3];
                o[0] = clip8(s0); o[1] = clip8(s1); o[2] = clip8(s2);
            }
        }
        src = &tmp;
        row_off = y_first;
    }
    if (!need_v) return *src;
    RgbImage out;
    out.width = src->width;
    out.height = height;
    out.px.resize(static_cast<size_t>(out.width) * height * 3);
    for (int yy = 0; yy < height; ++yy) {
        const int ymin = bv[static_cast<size_t>(yy) * 2] - row_off, ymax = bv[static_cast<size_t>(yy) * 2 + 1];
        const int32_t* k = &kv[static_cast<size_t>(yy) * ksv];
        for (int xx = 0; xx < out.width; ++xx) {
            int64_t s0 = 1 << (kPrecisionBits - 1), s1 = s0, s2 = s0;
            for (int y = 0; y < ymax; ++y) {
                const uint8_t* p = &src->px[(static_cast<size_t>(y + ymin) * src->width + xx) * 3];
                s0 += static_cast<int64_t>(p[0]) * k[y]; s1 += static_cast<int64_t>(p[1]) * k[y]; s2 += static_cast<int64_t>(p[2]) * k[y];
            }
            uint8_t* o = &out.px[(static_cast<size_t>(yy) * out.width + xx) * 3];
            o[0] = clip8(s0); o[1] = clip8(s1); o[2] = clip8(s2);
        }
    }
    return out;
}

}  // namespace dara::io
