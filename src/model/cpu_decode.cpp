// CPU decode of host-resident Qwen layers: see cpu_decode.h.
#include "model/cpu_decode.h"

#include <cuda_runtime.h>
#include <immintrin.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <thread>

#include "kernels/half.h"
#include "model/device_buffer.h"
#include "model/ops.h"

namespace dara::model {
namespace {

using dara::kernels::QuantType;

// ---- CPU feature check --------------------------------------------------------------------------------------------------------------
bool has_avx2_f16c() {
#ifdef _MSC_VER
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool f16c = (r[2] >> 29) & 1, fma = (r[2] >> 12) & 1, osx = (r[2] >> 27) & 1;
    __cpuidex(r, 7, 0);
    const bool avx2 = (r[1] >> 5) & 1;
    if (!(f16c && fma && osx && avx2)) return false;
    return (_xgetbv(0) & 6) == 6;   // the OS saves YMM state
#else
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#endif
}

// ---- persistent spinning thread pool ------------------------------------------------------------------------------------------------
// parallel_for(n, fn): fn(i) for i in [0, n) over the workers and the calling thread. Workers spin briefly between jobs (a token runs ~100 jobs back
// to back), then yield, then sleep, so an idle pool costs no CPU while the GPU works.
class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i + 1 < n; ++i) th_.emplace_back([this] { loop(); });
    }
    ~Pool() {
        stop_.store(true);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : th_) t.join();
    }
    int size() const { return static_cast<int>(th_.size()) + 1; }
    // While `hot` is set the idle workers keep spinning instead of yielding/sleeping, so the next job starts on all threads at once.
    void set_hot(bool on) { hot_.store(on, std::memory_order_release); if (on) gen_wake_.fetch_add(1, std::memory_order_release); }
    void parallel_for(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        if (th_.empty() || n == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
        fn_ = &fn;
        n_ = n;
        next_.store(0, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        run();
        while (done_.load(std::memory_order_acquire) < n || active_.load(std::memory_order_acquire) != 0) _mm_pause();
    }

private:
    void run() {
        int i;
        while ((i = next_.fetch_add(1, std::memory_order_relaxed)) < n_) {
            (*fn_)(i);
            done_.fetch_add(1, std::memory_order_release);
        }
    }
    void loop() {
        uint64_t seen = gen_.load();
        while (true) {
            uint64_t g;
            int spins = 0;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                if (hot_.load(std::memory_order_relaxed)) { _mm_pause(); continue; }
                if (++spins < 20000) _mm_pause();
                else if (spins < 40000) std::this_thread::yield();
                else std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
            seen = g;
            if (stop_.load()) return;
            active_.fetch_add(1, std::memory_order_acq_rel);
            run();
            active_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    std::vector<std::thread> th_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> next_{0}, done_{0}, active_{0};
    std::atomic<bool> stop_{false}, hot_{false};
    std::atomic<uint64_t> gen_wake_{0};
    const std::function<void(int)>* fn_ = nullptr;
    int n_ = 0;
};

// ---- int8 activations, q8_K style: ONE float scale per 256 values and the int sum of every 16 (as llama.cpp's CPU K-quant kernels).
// With a single activation scale per 256-weight block, the per-sub-block weight scales fold into the INTEGER accumulation (one madd_epi16 per 32
// weights) and the block needs a single float multiply-add.
struct Q8Act {
    std::vector<int8_t> q;
    std::vector<float> d;     // [K/256]
    std::vector<int16_t> s16; // [K/16]
    size_t K = 0;
    void quantize(const float* x, size_t k) {
        K = k;
        q.resize(k); d.resize(k / 256); s16.resize(k / 16);
        for (size_t b = 0; b < k / 256; ++b) {
            const float* xb = x + b * 256;
            __m256 mx = _mm256_setzero_ps();
            const __m256 sign = _mm256_set1_ps(-0.0f);
            for (int i = 0; i < 256; i += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + i)));
            __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
            const float amax = _mm_cvtss_f32(m4);
            const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            d[b] = amax / 127.0f;
            const __m256 vinv = _mm256_set1_ps(inv);
            for (int g = 0; g < 16; ++g) {   // 16 values: round to nearest even (cvtps), pack to int8, and their sum
                const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + g * 16), vinv));
                const __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + g * 16 + 8), vinv));
                const __m256i s = _mm256_add_epi32(i0, i1);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_hadd_epi32(s4, s4);
                s4 = _mm_hadd_epi32(s4, s4);
                s16[b * 16 + g] = static_cast<int16_t>(_mm_cvtsi128_si32(s4));
                const __m256i p16 = _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8);   // 16 x int16, in order
                const __m128i p8 = _mm_packs_epi16(_mm256_castsi256_si128(p16), _mm256_extracti128_si256(p16, 1));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(q.data() + b * 256 + g * 16), p8);
            }
        }
    }
};

inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
    return _mm_cvtss_f32(lo);
}
inline float h2f(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(v))); }   // F16C
inline __m256i srl16(__m256i v, int n) { return _mm256_srl_epi16(v, _mm_cvtsi32_si128(n)); }

// shuffle masks: broadcast int16 element j of each 128-bit lane to all 8 int16 lanes of that lane
alignas(32) const uint8_t kBcast16[8][32] = {
#define B16(j) {2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1, 2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1,2*j,2*j+1}
    B16(0), B16(1), B16(2), B16(3), B16(4), B16(5), B16(6), B16(7)
#undef B16
};
// Q6_K: low lane <- element 2k, high lane <- element 2k+1 (the two 16-value scale groups of a 32-value chunk)
alignas(32) const uint8_t kPair16[4][32] = {
#define P16(k) {4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1,4*k,4*k+1, 4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3,4*k+2,4*k+3}
    P16(0), P16(1), P16(2), P16(3)
#undef P16
};

template <bool Q5>
float dot_q45k(const uint8_t* row, size_t nblk, const Q8Act& a) {
    __m256 acc = _mm256_setzero_ps();
    __m128 accm = _mm_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    constexpr uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
    for (size_t b = 0; b < nblk; ++b) {
        const uint8_t* blk = row + b * (Q5 ? 176 : 144);
        const float dx = a.d[b];
        const float d = h2f(blk) * dx, dmin = h2f(blk + 2) * dx;
        uint32_t u[4];
        std::memcpy(u, blk + 4, 12);   // 6-bit scales/mins -> bytes 0-7 scales, 8-15 mins (ggml's unpacking)
        u[3] = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
        const uint32_t ua = u[1] & km1;
        u[1] = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
        u[2] = ua;
        u[0] &= km1;
        const __m256i sm = _mm256_cvtepu8_epi16(_mm_set_epi32(static_cast<int>(u[3]), static_cast<int>(u[2]), static_cast<int>(u[1]), static_cast<int>(u[0])));
        // mins term: sum_j m_j * (sum of the 32 activations of sub-block j)
        const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a.s16.data() + b * 16)),
                                          _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.s16.data() + b * 16 + 8)));
        accm = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_madd_epi16(_mm256_extracti128_si256(sm, 1), bs)), _mm_set1_ps(dmin), accm);
        const __m256i sc = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sm));   // 8 int16 scales in both lanes
        const uint8_t* qs = blk + (Q5 ? 48 : 16);
        __m256i hb = _mm256_setzero_si256();
        if (Q5) hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16));
        const int8_t* aq = a.q.data() + b * 256;
        __m256i isum = _mm256_setzero_si256();
        for (int g = 0; g < 4; ++g) {
            const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * g));
            __m256i lo = _mm256_and_si256(w, m4), hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m4);
            if (Q5) {   // bit 2g / 2g+1 of each qh byte: consume two bits per group with constant shifts
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(hb, one), 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 1), one), 4));
                hb = _mm256_srli_epi16(hb, 2);
            }
            const __m256i plo = _mm256_maddubs_epi16(lo, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 64 * g)));
            const __m256i phi = _mm256_maddubs_epi16(hi, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 64 * g + 32)));
            const __m256i s0 = _mm256_shuffle_epi8(sc, _mm256_load_si256(reinterpret_cast<const __m256i*>(kBcast16[2 * g])));
            const __m256i s1 = _mm256_shuffle_epi8(sc, _mm256_load_si256(reinterpret_cast<const __m256i*>(kBcast16[2 * g + 1])));
            isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_madd_epi16(plo, s0), _mm256_madd_epi16(phi, s1)));
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(d), acc);
    }
    __m128 m = _mm_add_ps(accm, _mm_movehl_ps(accm, accm));
    m = _mm_add_ss(m, _mm_movehdup_ps(m));
    return hsum(acc) - _mm_cvtss_f32(m);
}

float dot_q6k(const uint8_t* row, size_t nblk, const Q8Act& a) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3);
    for (size_t b = 0; b < nblk; ++b) {
        const uint8_t* blk = row + b * 210;
        const float d = h2f(blk + 208) * a.d[b];
        // the 16 int8 scales as int16; the -32 offset folds in as - 32 * scale * (sum of 16 activations)
        const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 192)));
        const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.s16.data() + b * 16));
        __m256i isum = _mm256_mullo_epi32(_mm256_madd_epi16(sc16, bs), _mm256_set1_epi32(-32));
        for (int h = 0; h < 2; ++h) {
            const __m256i Q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * h));
            const __m256i Q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * h + 32));
            const __m256i QH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 128 + 32 * h));
            const __m256i sch = _mm256_broadcastsi128_si256(h == 0 ? _mm256_castsi256_si128(sc16) : _mm256_extracti128_si256(sc16, 1));   // scales of half h
            const __m256i q[4] = {
                _mm256_or_si256(_mm256_and_si256(Q0, m4), _mm256_slli_epi16(_mm256_and_si256(QH, m2), 4)),
                _mm256_or_si256(_mm256_and_si256(Q1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 2), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(Q0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 4), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(Q1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 6), m2), 4)),
            };
            const int8_t* aq = a.q.data() + b * 256 + 128 * h;
            for (int k = 0; k < 4; ++k) {
                const __m256i p = _mm256_maddubs_epi16(q[k], _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 32 * k)));   // lanes 0-7: values 0-15
                const __m256i s = _mm256_shuffle_epi8(sch, _mm256_load_si256(reinterpret_cast<const __m256i*>(kPair16[k])));
                isum = _mm256_add_epi32(isum, _mm256_madd_epi16(p, s));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(d), acc);
    }
    return hsum(acc);
}

float dot_q80(const uint8_t* row, size_t nblk32, const Q8Act& a) {
    __m256 acc = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nblk32; ++b) {
        const uint8_t* blk = row + b * 34;
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 2));
        const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.q.data() + b * 32));
        const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), ones);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(h2f(blk) * a.d[b / 8]), acc);
    }
    return hsum(acc);
}

float dot_row(const HostMatrix& m, size_t r, const Q8Act& a) {
    const uint8_t* row = m.data + r * m.row_bytes;
    switch (m.qtype) {
        case QuantType::Q4_K: return dot_q45k<false>(row, m.cols / 256, a);
        case QuantType::Q5_K: return dot_q45k<true>(row, m.cols / 256, a);
        case QuantType::Q6_K: return dot_q6k(row, m.cols / 256, a);
        case QuantType::Q8_0: return dot_q80(row, m.cols / 32, a);
    }
    return 0.0f;
}

// ---- two rows at a time: the block's 8 activation vectors are loaded once and used by both rows, and the two rows' dependency chains interleave
template <bool Q5>
inline void q45k_block(const uint8_t* blk, float dx, const __m256i av[8], const int16_t* bs16, __m256& acc, __m128& accm) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    constexpr uint32_t km1 = 0x3f3f3f3f, km2 = 0x0f0f0f0f, km3 = 0x03030303;
    const float d = h2f(blk) * dx, dmin = h2f(blk + 2) * dx;
    uint32_t u[4];
    std::memcpy(u, blk + 4, 12);
    u[3] = ((u[2] >> 4) & km2) | (((u[1] >> 6) & km3) << 4);
    const uint32_t ua = u[1] & km1;
    u[1] = (u[2] & km2) | (((u[0] >> 6) & km3) << 4);
    u[2] = ua;
    u[0] &= km1;
    const __m256i sm = _mm256_cvtepu8_epi16(_mm_set_epi32(static_cast<int>(u[3]), static_cast<int>(u[2]), static_cast<int>(u[1]), static_cast<int>(u[0])));
    const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bs16)), _mm_loadu_si128(reinterpret_cast<const __m128i*>(bs16 + 8)));
    accm = _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_madd_epi16(_mm256_extracti128_si256(sm, 1), bs)), _mm_set1_ps(dmin), accm);
    const __m256i sc = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(sm));
    const uint8_t* qs = blk + (Q5 ? 48 : 16);
    __m256i hb = _mm256_setzero_si256();
    if (Q5) hb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 16));
    __m256i isum = _mm256_setzero_si256();
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32 * g));
        __m256i lo = _mm256_and_si256(w, m4), hi = _mm256_and_si256(_mm256_srli_epi16(w, 4), m4);
        if (Q5) {
            lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(hb, one), 4));
            hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 1), one), 4));
            hb = _mm256_srli_epi16(hb, 2);
        }
        const __m256i s0 = _mm256_shuffle_epi8(sc, _mm256_load_si256(reinterpret_cast<const __m256i*>(kBcast16[2 * g])));
        const __m256i s1 = _mm256_shuffle_epi8(sc, _mm256_load_si256(reinterpret_cast<const __m256i*>(kBcast16[2 * g + 1])));
        isum = _mm256_add_epi32(isum, _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, av[2 * g]), s0),
                                                       _mm256_madd_epi16(_mm256_maddubs_epi16(hi, av[2 * g + 1]), s1)));
    }
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(d), acc);
}

template <bool Q5>
void dot2_q45k(const uint8_t* r0, const uint8_t* r1, size_t nblk, const Q8Act& a, float* y0, float* y1) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m128 m0 = _mm_setzero_ps(), m1 = _mm_setzero_ps();
    constexpr size_t BB = Q5 ? 176 : 144;
    for (size_t b = 0; b < nblk; ++b) {
        __m256i av[8];
        const int8_t* aq = a.q.data() + b * 256;
        for (int i = 0; i < 8; ++i) av[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 32 * i));
        const int16_t* bs = a.s16.data() + b * 16;
        q45k_block<Q5>(r0 + b * BB, a.d[b], av, bs, acc0, m0);
        q45k_block<Q5>(r1 + b * BB, a.d[b], av, bs, acc1, m1);
    }
    auto fin = [](__m256 acc, __m128 m) { m = _mm_add_ps(m, _mm_movehl_ps(m, m)); m = _mm_add_ss(m, _mm_movehdup_ps(m)); return hsum(acc) - _mm_cvtss_f32(m); };
    *y0 = fin(acc0, m0);
    *y1 = fin(acc1, m1);
}

inline void q6k_block(const uint8_t* blk, float dx, const __m256i av[8], const int16_t* bs16, __m256& acc) {
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3);
    const float d = h2f(blk + 208) * dx;
    const __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 192)));
    const __m256i bs = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bs16));
    __m256i isum = _mm256_mullo_epi32(_mm256_madd_epi16(sc16, bs), _mm256_set1_epi32(-32));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const __m256i Q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * h));
        const __m256i Q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 64 * h + 32));
        const __m256i QH = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk + 128 + 32 * h));
        const __m256i sch = _mm256_broadcastsi128_si256(h == 0 ? _mm256_castsi256_si128(sc16) : _mm256_extracti128_si256(sc16, 1));
        const __m256i q[4] = {
            _mm256_or_si256(_mm256_and_si256(Q0, m4), _mm256_slli_epi16(_mm256_and_si256(QH, m2), 4)),
            _mm256_or_si256(_mm256_and_si256(Q1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 2), m2), 4)),
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(Q0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 4), m2), 4)),
            _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(Q1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(QH, 6), m2), 4)),
        };
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const __m256i s = _mm256_shuffle_epi8(sch, _mm256_load_si256(reinterpret_cast<const __m256i*>(kPair16[k])));
            isum = _mm256_add_epi32(isum, _mm256_madd_epi16(_mm256_maddubs_epi16(q[k], av[4 * h + k]), s));
        }
    }
    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(d), acc);
}

void dot2_q6k(const uint8_t* r0, const uint8_t* r1, size_t nblk, const Q8Act& a, float* y0, float* y1) {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (size_t b = 0; b < nblk; ++b) {
        __m256i av[8];
        const int8_t* aq = a.q.data() + b * 256;
        for (int i = 0; i < 8; ++i) av[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(aq + 32 * i));
        const int16_t* bs = a.s16.data() + b * 16;
        q6k_block(r0 + b * 210, a.d[b], av, bs, acc0);
        q6k_block(r1 + b * 210, a.d[b], av, bs, acc1);
    }
    *y0 = hsum(acc0);
    *y1 = hsum(acc1);
}

// rows [r0, r1) of m into y, in pairs where the format has a two-row kernel
void dot_rows(const HostMatrix& m, size_t r0, size_t r1, const Q8Act& a, float* y) {
    size_t r = r0;
    const size_t nb = m.cols / 256;
    // two-row kernels MEASURED SLOWER (9B decode 8.7 vs 9.4-10.0 tok/s, alternated: register pressure): opt-in only, DARA_CPU_PAIR=1
    static const bool pairs = [] { const char* e = std::getenv("DARA_CPU_PAIR"); return e && e[0] == '1'; }();
    if (pairs && (m.qtype == QuantType::Q4_K || m.qtype == QuantType::Q5_K || m.qtype == QuantType::Q6_K)) {
        for (; r + 2 <= r1; r += 2) {
            const uint8_t* p0 = m.data + r * m.row_bytes;
            const uint8_t* p1 = p0 + m.row_bytes;
            if (m.qtype == QuantType::Q4_K) dot2_q45k<false>(p0, p1, nb, a, &y[r], &y[r + 1]);
            else if (m.qtype == QuantType::Q5_K) dot2_q45k<true>(p0, p1, nb, a, &y[r], &y[r + 1]);
            else dot2_q6k(p0, p1, nb, a, &y[r], &y[r + 1]);
        }
    }
    for (; r < r1; ++r) y[r] = dot_row(m, r, a);
}

inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

void rms(const float* x, const float* w, int n, float eps, float* y) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const float r = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

}  // namespace

// ---- decoder -----------------------------------------------------------------------------------------------------------------------------
struct CpuDecoder::Impl {
    Pool pool;
    Q8Act act;
    std::vector<float> h, x1, m, qkv, conv, z, b, a, core, normed, qn, kn, g, u, qg, q, gate, k, v, att, cs, sn;
    explicit Impl(int n) : pool(n) {}

    // y[r] = M x for all rows, x already quantised into `act`
    void matvec(const HostMatrix& M, float* y) {
        // a few contiguous row ranges per thread (dynamic only at that granularity: little synchronisation, still load-balanced)
        const size_t kRowsPerTask = std::max<size_t>(8, (M.rows + pool.size() * 4 - 1) / (pool.size() * 4));
        const auto t0 = std::chrono::steady_clock::now();
        const int tasks = static_cast<int>((M.rows + kRowsPerTask - 1) / kRowsPerTask);
        pool.parallel_for(tasks, [&](int t) {
            const size_t r0 = static_cast<size_t>(t) * kRowsPerTask, r1 = std::min(M.rows, r0 + kRowsPerTask);
            dot_rows(M, r0, r1, act, y);
        });
        mv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        mv_bytes += M.rows * M.row_bytes;
        ++mv_calls;
        { const int q = static_cast<int>(M.qtype); q_s[q] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); q_b[q] += M.rows * M.row_bytes; }
    }
    // y_i = M_i x for several matrices that read the same quantised input: ONE parallel job over all their row chunks
    void matvecs(std::initializer_list<std::pair<const HostMatrix*, float*>> list) {
        chunks.clear();
        size_t rows = 0;
        for (const auto& e : list) rows += e.first->rows;
        const size_t per = std::max<size_t>(8, (rows + pool.size() * 4 - 1) / (pool.size() * 4));
        for (const auto& e : list)
            for (size_t r0 = 0; r0 < e.first->rows; r0 += per) chunks.push_back({e.first, e.second, r0, std::min(e.first->rows, r0 + per)});
        const auto t0 = std::chrono::steady_clock::now();
        pool.parallel_for(static_cast<int>(chunks.size()), [&](int i) {
            const auto& ch = chunks[static_cast<size_t>(i)];
            dot_rows(*ch.m, ch.r0, ch.r1, act, ch.y);
        });
        mv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (const auto& e : list) mv_bytes += e.first->rows * e.first->row_bytes;
        ++mv_calls;
    }
    // y_i[t * rows_i + r] = M_i x_t for T quantised inputs acts[0..T): weights read from RAM once (a chunk of rows stays in cache across the tokens)
    void matvecs_multi(std::initializer_list<std::pair<const HostMatrix*, float*>> list, int T) {
        if (T == 1) { act_swap(); matvecs(list); act_swap(); return; }
        chunks.clear();
        size_t rows = 0;
        for (const auto& e : list) rows += e.first->rows;
        const size_t per = std::max<size_t>(8, (rows + pool.size() * 4 - 1) / (pool.size() * 4));
        for (const auto& e : list)
            for (size_t r0 = 0; r0 < e.first->rows; r0 += per) chunks.push_back({e.first, e.second, r0, std::min(e.first->rows, r0 + per)});
        const auto t0 = std::chrono::steady_clock::now();
        pool.parallel_for(static_cast<int>(chunks.size()), [&](int i) {
            const auto& ch = chunks[static_cast<size_t>(i)];
            for (int t = 0; t < T; ++t) dot_rows(*ch.m, ch.r0, ch.r1, acts[static_cast<size_t>(t)], ch.y + static_cast<size_t>(t) * ch.m->rows);
        });
        mv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        for (const auto& e : list) mv_bytes += e.first->rows * e.first->row_bytes;
        ++mv_calls;
    }
    void act_swap() { std::swap(act, acts[0]); }   // the single-token path quantises into `act`
    std::vector<Q8Act> acts;
    std::vector<float> hT, x1T, mT, qkvT, zT, bT, aT, normedT, qgT, kT, vT, attT, gT, uT;
    struct ChunkT { const HostMatrix* m; float* y; size_t r0, r1; };
    std::vector<ChunkT> chunks;
    // elementwise work split over the pool in blocks of 1024
    template <class F>
    void parallel_elems(int n, F&& f) {
        const int blocks = (n + 1023) / 1024;
        pool.parallel_for(blocks, [&](int b) { const int a = b * 1024, e = std::min(n, a + 1024); for (int i = a; i < e; ++i) f(i); });
    }
    // DARA_CPU_PROF=1: per-token breakdown printed every 8 tokens
    double t_rec = 0, t_att = 0;
    double mv_s = 0, total_s = 0, q_s[4] = {0, 0, 0, 0};
    size_t q_b[4] = {0, 0, 0, 0};
    size_t mv_bytes = 0;
    long mv_calls = 0, tokens = 0;
    bool prof = [] { const char* e = std::getenv("DARA_CPU_PROF"); return e && e[0] == '1'; }();
};

bool cpu_decode_available() {
    static const bool ok = has_avx2_f16c();
    return ok;
}
bool cpu_decode_supports(KvFormat f) { return f == KvFormat::F32 || f == KvFormat::Q8; }

CpuDecoder::CpuDecoder(int threads) {
    // default: about 2/3 of the hardware threads (measured on a 6-core/12-thread Ryzen 5600H: 8 threads beat 6 and 12 for decode)
    if (threads <= 0) threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) * 2 / 3);
    impl_ = std::make_unique<Impl>(threads);
}
CpuDecoder::~CpuDecoder() = default;
int CpuDecoder::threads() const { return impl_->pool.size(); }
void CpuDecoder::set_hot(bool on) { impl_->pool.set_hot(on); }

static std::vector<float> down(const DevBuf<float>& b) {
    std::vector<float> v(b.n);
    if (b.n) cuda_check(cudaMemcpy(v.data(), b.p, b.n * sizeof(float), cudaMemcpyDeviceToHost), "download layer parameter");
    return v;
}

CpuLayerWeights CpuDecoder::make_weights(const PackedLayer& p) {
    const GpuLayerWeights& w = p.weights;
    if (!p.host.data()) throw std::logic_error("cpu decode: layer has no host image");
    CpuLayerWeights c;
    c.kind = w.kind; c.hidden = w.hidden; c.inter = w.inter; c.eps = w.eps;
    c.dn = w.dn; c.order = w.order; c.at = w.at;
    c.input_norm = down(w.input_norm); c.post_norm = down(w.post_norm);
    c.conv_w = down(w.conv_w); c.ssm_a = down(w.ssm_a); c.dt_bias = down(w.dt_bias); c.norm_w = down(w.norm_w);
    c.q_norm = down(w.q_norm); c.k_norm = down(w.k_norm); c.inv_freq = down(w.inv_freq);
    for (const MatrixSlot& s : p.slots) {
        HostMatrix hm{p.host.data() + s.offset, s.qtype, s.rows, s.cols, s.rows ? s.bytes / s.rows : 0};
        if (s.cols % 256 != 0 && s.qtype != QuantType::Q8_0) throw std::runtime_error("cpu decode: K-quant row not a multiple of 256");
        if (s.member == &GpuLayerWeights::gate) c.gate = hm;
        else if (s.member == &GpuLayerWeights::up) c.up = hm;
        else if (s.member == &GpuLayerWeights::down) c.down = hm;
        else if (s.member == &GpuLayerWeights::in_qkv) c.in_qkv = hm;
        else if (s.member == &GpuLayerWeights::in_z) c.in_z = hm;
        else if (s.member == &GpuLayerWeights::in_b) c.in_b = hm;
        else if (s.member == &GpuLayerWeights::in_a) c.in_a = hm;
        else if (s.member == &GpuLayerWeights::dn_out) c.dn_out = hm;
        else if (s.member == &GpuLayerWeights::q_proj) c.q_proj = hm;
        else if (s.member == &GpuLayerWeights::k_proj) c.k_proj = hm;
        else if (s.member == &GpuLayerWeights::v_proj) c.v_proj = hm;
        else if (s.member == &GpuLayerWeights::o_proj) c.o_proj = hm;
    }
    return c;
}

CpuLayerState CpuDecoder::make_state(const CpuLayerWeights& w, size_t cap, KvFormat fmt) {
    CpuLayerState s;
    s.kv_format = fmt;
    if (w.kind == LayerKind::DeltaNet) {
        s.conv.assign(static_cast<size_t>(w.dn.conv_dim()) * (w.dn.conv_kernel - 1), 0.0f);
        s.S.assign(static_cast<size_t>(w.dn.v_heads) * w.dn.k_dim * w.dn.v_dim, 0.0f);
    } else {
        s.kv_cap = cap;
        const size_t n = cap * static_cast<size_t>(w.at.kv_width());
        if (fmt == KvFormat::Q8) { s.Kq.resize(n); s.Vq.resize(n); s.Kn.resize(n / 32); s.Vn.resize(n / 32); }
        else { s.K.resize(n); s.V.resize(n); }
    }
    return s;
}

void CpuDecoder::download(const GpuLayerState& g, CpuLayerState& c, size_t kv_len) {
    auto cp = [](void* dst, const void* src, size_t bytes) { if (bytes) cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), "download layer state"); };
    if (g.kind == LayerKind::DeltaNet) {
        cp(c.conv.data(), g.conv_state.p, c.conv.size() * 4);
        cp(c.S.data(), g.S.p, c.S.size() * 4);
    } else if (c.kv_format == KvFormat::Q8) {
        const size_t n = kv_len * (c.Kq.size() / c.kv_cap);
        cp(c.Kq.data(), g.Kq.p, n); cp(c.Vq.data(), g.Vq.p, n); cp(c.Kn.data(), g.Kn.p, n / 32 * 4); cp(c.Vn.data(), g.Vn.p, n / 32 * 4);
    } else {
        const size_t n = kv_len * (c.K.size() / c.kv_cap);
        cp(c.K.data(), g.Kc.p, n * 4); cp(c.V.data(), g.Vc.p, n * 4);
    }
}

void CpuDecoder::upload(const CpuLayerState& c, GpuLayerState& g, size_t kv_len) {
    auto cp = [](void* dst, const void* src, size_t bytes) { if (bytes) cuda_check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "upload layer state"); };
    if (g.kind == LayerKind::DeltaNet) {
        cp(g.conv_state.p, c.conv.data(), c.conv.size() * 4);
        cp(g.S.p, c.S.data(), c.S.size() * 4);
    } else if (c.kv_format == KvFormat::Q8) {
        const size_t n = kv_len * (c.Kq.size() / c.kv_cap);
        cp(g.Kq.p, c.Kq.data(), n); cp(g.Vq.p, c.Vq.data(), n); cp(g.Kn.p, c.Kn.data(), n / 32 * 4); cp(g.Vn.p, c.Vn.data(), n / 32 * 4);
    } else {
        const size_t n = kv_len * (c.K.size() / c.kv_cap);
        cp(g.Kc.p, c.K.data(), n * 4); cp(g.Vc.p, c.V.data(), n * 4);
    }
}

void CpuDecoder::matvec(const HostMatrix& M, const float* x, float* y) {
    impl_->act.quantize(x, M.cols);
    impl_->matvec(M, y);
}

// ---- per-token cores (shared by the single-token and the multi-token paths, so both do bitwise the same arithmetic) ---------------------------
// DeltaNet: conv + l2norm + recurrence + gated RMSNorm for ONE token whose projections are qkv/z/b/a; advances s.conv and s.S; writes normed [value_dim].
static void dn_core(CpuDecoder::Impl& I, const CpuLayerWeights& w, CpuLayerState& s, const float* qkv, const float* z, const float* bb, const float* aa,
                    float* normed) {
    const DeltaNetDims& d = w.dn;
    const int cd = d.conv_dim(), kd = d.key_dim(), taps = d.conv_kernel, hist = taps - 1;
    I.conv.resize(cd);
    I.qn.resize(static_cast<size_t>(d.k_heads) * d.k_dim); I.kn.resize(I.qn.size());
    I.parallel_elems(cd, [&](int c) {   // causal depthwise conv (state: the last 3 inputs, oldest first) + silu
        float* hs = &s.conv[static_cast<size_t>(c) * hist];
        const float* wc = &w.conv_w[static_cast<size_t>(c) * taps];
        float acc = 0.0f;
        for (int k = 0; k < hist; ++k) acc += wc[k] * hs[k];
        acc += wc[taps - 1] * qkv[c];
        I.conv[c] = siluf(acc);
        for (int k = 0; k + 1 < hist; ++k) hs[k] = hs[k + 1];
        hs[hist - 1] = qkv[c];
    });
    const float qscale = 1.0f / std::sqrt(static_cast<float>(d.k_dim));
    const float* qc = I.conv.data();
    const float* kc = I.conv.data() + kd;
    const float* vc = I.conv.data() + 2 * kd;
    for (int h = 0; h < d.k_heads; ++h) {
        double sq = 0, sk = 0;
        for (int i = 0; i < d.k_dim; ++i) { sq += static_cast<double>(qc[h * d.k_dim + i]) * qc[h * d.k_dim + i]; sk += static_cast<double>(kc[h * d.k_dim + i]) * kc[h * d.k_dim + i]; }
        const float rq = static_cast<float>(1.0 / std::sqrt(sq + 1e-6)), rk = static_cast<float>(1.0 / std::sqrt(sk + 1e-6));
        for (int i = 0; i < d.k_dim; ++i) { I.qn[h * d.k_dim + i] = qc[h * d.k_dim + i] * rq * qscale; I.kn[h * d.k_dim + i] = kc[h * d.k_dim + i] * rk; }
    }
    // recurrence, one task per V head: S <- S*exp(g); kv = S^T k; delta = (v - kv)*beta; S <- S + k (x) delta; o = S^T q;  then the gated RMSNorm
    const auto t_r = std::chrono::steady_clock::now();
    I.pool.parallel_for(d.v_heads, [&](int h) {
        const int kh = key_head_for_v(w.order, h, d);
        const float beta = sigmoidf(bb[h]);
        const float decay = std::exp(w.ssm_a[h] * softplusf(aa[h] + w.dt_bias[h]));
        float* S = &s.S[static_cast<size_t>(h) * d.k_dim * d.v_dim];
        const float* k = &I.kn[static_cast<size_t>(kh) * d.k_dim];
        const float* q = &I.qn[static_cast<size_t>(kh) * d.k_dim];
        const float* v = vc + static_cast<size_t>(h) * d.v_dim;
        float o[512];
        float kv[512];
        for (int vi = 0; vi < d.v_dim; ++vi) kv[vi] = 0.0f;
        for (int ki = 0; ki < d.k_dim; ++ki) {          // row-wise over S (contiguous in v): decay and kv = S^T k
            float* Sr = S + static_cast<size_t>(ki) * d.v_dim;
            const float kk = k[ki];
            for (int vi = 0; vi < d.v_dim; ++vi) { Sr[vi] *= decay; kv[vi] += Sr[vi] * kk; }
        }
        for (int vi = 0; vi < d.v_dim; ++vi) { kv[vi] = (v[vi] - kv[vi]) * beta; o[vi] = 0.0f; }   // kv now holds delta
        for (int ki = 0; ki < d.k_dim; ++ki) {
            float* Sr = S + static_cast<size_t>(ki) * d.v_dim;
            const float kk = k[ki], qq = q[ki];
            for (int vi = 0; vi < d.v_dim; ++vi) { Sr[vi] += kk * kv[vi]; o[vi] += Sr[vi] * qq; }
        }
        if (!normed) return;   // replay: only the state matters
        double ms = 0.0;
        for (int i = 0; i < d.v_dim; ++i) ms += static_cast<double>(o[i]) * o[i];
        const float r = static_cast<float>(1.0 / std::sqrt(ms / d.v_dim + d.eps));
        for (int i = 0; i < d.v_dim; ++i)
            normed[static_cast<size_t>(h) * d.v_dim + i] = w.norm_w[i] * (o[i] * r) * siluf(z[static_cast<size_t>(h) * d.v_dim + i]);
    });
    I.t_rec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_r).count();
}

// Attention for ONE token at absolute position pos: qg [q_rows] (query|gate per head), k/v [kv_width] projections; appends K/V; writes att [out_width].
static void attn_core(CpuDecoder::Impl& I, const CpuLayerWeights& w, CpuLayerState& s, const float* qg, const float* kin, const float* vin, int64_t pos,
                      float* att) {
    const AttentionDims& d = w.at;
    const int hd = d.head_dim, NH = d.heads, KH = d.kv_heads, OW = d.out_width(), KW = d.kv_width();
    I.q.resize(static_cast<size_t>(NH) * hd); I.gate.resize(OW); I.k.assign(kin, kin + KW); I.v.assign(vin, vin + KW);
    for (int h = 0; h < NH; ++h) {
        std::copy_n(&qg[static_cast<size_t>(h) * 2 * hd], hd, &I.q[static_cast<size_t>(h) * hd]);
        std::copy_n(&qg[static_cast<size_t>(h) * 2 * hd + hd], hd, &I.gate[static_cast<size_t>(h) * hd]);
    }
    for (int h = 0; h < NH; ++h) rms(&I.q[static_cast<size_t>(h) * hd], w.q_norm.data(), hd, d.eps, &I.q[static_cast<size_t>(h) * hd]);
    for (int h = 0; h < KH; ++h) rms(&I.k[static_cast<size_t>(h) * hd], w.k_norm.data(), hd, d.eps, &I.k[static_cast<size_t>(h) * hd]);
    I.cs.resize(d.rot_dim / 2); I.sn.resize(d.rot_dim / 2);
    rope_cos_sin(pos, w.inv_freq, I.cs.data(), I.sn.data());
    rope_apply(I.q.data(), NH, hd, d.rot_dim, I.cs.data(), I.sn.data());
    rope_apply(I.k.data(), KH, hd, d.rot_dim, I.cs.data(), I.sn.data());
    if (static_cast<size_t>(pos) >= s.kv_cap) throw std::length_error("K/V cache capacity exceeded");
    const bool q8 = s.kv_format == KvFormat::Q8;
    if (q8) {   // same quantiser as q8kv_append_cuda: per 32 values, scale = max|x| / 127, round to nearest even
        auto put = [&](const float* src, int8_t* codes, float* scales) {
            for (int g = 0; g < KW / 32; ++g) {
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(src[g * 32 + i]));
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                for (int i = 0; i < 32; ++i) codes[g * 32 + i] = static_cast<int8_t>(std::nearbyint(src[g * 32 + i] * inv));
                scales[g] = amax / 127.0f;
            }
        };
        put(I.k.data(), &s.Kq[static_cast<size_t>(pos) * KW], &s.Kn[static_cast<size_t>(pos) * KW / 32]);
        put(I.v.data(), &s.Vq[static_cast<size_t>(pos) * KW], &s.Vn[static_cast<size_t>(pos) * KW / 32]);
    } else {
        std::copy(I.k.begin(), I.k.end(), &s.K[static_cast<size_t>(pos) * KW]);
        std::copy(I.v.begin(), I.v.end(), &s.V[static_cast<size_t>(pos) * KW]);
    }
    const size_t n = static_cast<size_t>(pos) + 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto t_a = std::chrono::steady_clock::now();
    I.pool.parallel_for(NH, [&](int h) {
        const int kh = h / d.groups();
        const float* qh = &I.q[static_cast<size_t>(h) * hd];
        std::vector<float> sc(n);
        float mx = -INFINITY;
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            const size_t base = j * KW + static_cast<size_t>(kh) * hd;
            if (q8) {
                for (int g = 0; g < hd / 32; ++g) {
                    float dd = 0.0f;
                    for (int i = 0; i < 32; ++i) dd += qh[g * 32 + i] * static_cast<float>(s.Kq[base + g * 32 + i]);
                    acc += dd * s.Kn[base / 32 + g];
                }
            } else {
                for (int i = 0; i < hd; ++i) acc += qh[i] * s.K[base + i];
            }
            sc[j] = acc * scale;
            mx = std::max(mx, sc[j]);
        }
        float denom = 0.0f;
        for (size_t j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - mx); denom += sc[j]; }
        float* o = &att[static_cast<size_t>(h) * hd];
        for (int i = 0; i < hd; ++i) o[i] = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            const size_t base = j * KW + static_cast<size_t>(kh) * hd;
            const float p = sc[j] / denom;
            if (q8) {
                for (int g = 0; g < hd / 32; ++g) {
                    const float f = p * s.Vn[base / 32 + g];
                    for (int i = 0; i < 32; ++i) o[g * 32 + i] += f * static_cast<float>(s.Vq[base + g * 32 + i]);
                }
            } else {
                for (int i = 0; i < hd; ++i) o[i] += p * s.V[base + i];
            }
        }
        for (int i = 0; i < hd; ++i) o[i] *= sigmoidf(I.gate[static_cast<size_t>(h) * hd + i]);   // output gate before o_proj
    });
    I.t_att += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_a).count();
}

void CpuDecoder::forward(const CpuLayerWeights& w, CpuLayerState& s, const float* x, float* y, int64_t pos) {
    forward_multi(w, s, x, y, pos, 1, nullptr);
}

size_t CpuDecoder::tape_floats(const CpuLayerWeights& w, int T) {
    if (w.kind != LayerKind::DeltaNet) return 0;
    const DeltaNetDims& d = w.dn;
    return static_cast<size_t>(T) * (static_cast<size_t>(d.conv_dim()) + d.value_dim() + 2 * static_cast<size_t>(d.v_heads));
}

// T tokens through one layer. Every matrix row is read from RAM once and applied to all T activations while it is in cache (the int8 kernels are
// the single-token ones, called per token: same arithmetic, so T steps here == T single-token forwards, bitwise). The recurrence / attention run
// token by token. tape (optional, DeltaNet): the per-token recurrence inputs, for replay() after a speculative rollback.
void CpuDecoder::forward_multi(const CpuLayerWeights& w, CpuLayerState& s, const float* x, float* y, int64_t pos0, int T, float* tape) {
    Impl& I = *impl_;
    const auto t_fwd = std::chrono::steady_clock::now();
    const int H = w.hidden;
    const size_t TH = static_cast<size_t>(T) * H;
    if (static_cast<int>(I.acts.size()) < T) I.acts.resize(static_cast<size_t>(T));
    I.hT.resize(TH); I.x1T.resize(TH); I.mT.resize(TH);
    for (int t = 0; t < T; ++t) {
        rms(x + static_cast<size_t>(t) * H, w.input_norm.data(), H, w.eps, &I.hT[static_cast<size_t>(t) * H]);
        I.acts[t].quantize(&I.hT[static_cast<size_t>(t) * H], H);
    }
    if (w.kind == LayerKind::DeltaNet) {
        const DeltaNetDims& d = w.dn;
        const size_t cd = static_cast<size_t>(d.conv_dim()), vd = static_cast<size_t>(d.value_dim()), vh = static_cast<size_t>(d.v_heads);
        I.qkvT.resize(T * cd); I.zT.resize(T * vd); I.bT.resize(T * vh); I.aT.resize(T * vh); I.normedT.resize(T * vd);
        I.matvecs_multi({{&w.in_qkv, I.qkvT.data()}, {&w.in_z, I.zT.data()}, {&w.in_b, I.bT.data()}, {&w.in_a, I.aT.data()}}, T);
        if (tape) {
            float* tp = tape;
            std::copy(I.qkvT.begin(), I.qkvT.begin() + T * cd, tp); tp += T * cd;
            std::copy(I.zT.begin(), I.zT.begin() + T * vd, tp); tp += T * vd;
            std::copy(I.bT.begin(), I.bT.begin() + T * vh, tp); tp += T * vh;
            std::copy(I.aT.begin(), I.aT.begin() + T * vh, tp);
        }
        for (int t = 0; t < T; ++t)
            dn_core(I, w, s, &I.qkvT[t * cd], &I.zT[t * vd], &I.bT[t * vh], &I.aT[t * vh], &I.normedT[t * vd]);
        for (int t = 0; t < T; ++t) I.acts[t].quantize(&I.normedT[t * vd], vd);
        I.matvecs_multi({{&w.dn_out, I.mT.data()}}, T);
    } else {
        const AttentionDims& d = w.at;
        const size_t qr = static_cast<size_t>(d.q_rows()), kw = static_cast<size_t>(d.kv_width()), ow = static_cast<size_t>(d.out_width());
        I.qgT.resize(T * qr); I.kT.resize(T * kw); I.vT.resize(T * kw); I.attT.resize(T * ow);
        I.matvecs_multi({{&w.q_proj, I.qgT.data()}, {&w.k_proj, I.kT.data()}, {&w.v_proj, I.vT.data()}}, T);
        for (int t = 0; t < T; ++t) attn_core(I, w, s, &I.qgT[t * qr], &I.kT[t * kw], &I.vT[t * kw], pos0 + t, &I.attT[t * ow]);
        for (int t = 0; t < T; ++t) I.acts[t].quantize(&I.attT[t * ow], ow);
        I.matvecs_multi({{&w.o_proj, I.mT.data()}}, T);
    }
    for (size_t i = 0; i < TH; ++i) I.x1T[i] = x[i] + I.mT[i];

    // MLP: down(silu(gate(h2)) * up(h2))
    const size_t IN = static_cast<size_t>(w.inter);
    for (int t = 0; t < T; ++t) {
        rms(&I.x1T[static_cast<size_t>(t) * H], w.post_norm.data(), H, w.eps, &I.hT[static_cast<size_t>(t) * H]);
        I.acts[t].quantize(&I.hT[static_cast<size_t>(t) * H], H);
    }
    I.gT.resize(T * IN); I.uT.resize(T * IN);
    I.matvecs_multi({{&w.gate, I.gT.data()}, {&w.up, I.uT.data()}}, T);
    I.parallel_elems(static_cast<int>(T * IN), [&](int i) { I.gT[i] = siluf(I.gT[i]) * I.uT[i]; });
    for (int t = 0; t < T; ++t) I.acts[t].quantize(&I.gT[t * IN], IN);
    I.matvecs_multi({{&w.down, I.mT.data()}}, T);
    for (size_t i = 0; i < TH; ++i) y[i] = I.x1T[i] + I.mT[i];
    if (I.prof) {
        I.total_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_fwd).count();
        I.tokens += T;
        if (I.tokens % 64 < static_cast<long>(T)) {
            std::fprintf(stderr, "[cpu decode] %ld layer-token-steps: %.2f ms per layer-token total, matvec %.2f ms (%ld calls, %.1f GB/s of weights), other %.2f ms\n",
                         I.tokens, I.total_s * 1e3 / I.tokens, I.mv_s * 1e3 / I.tokens, I.mv_calls, I.mv_bytes / I.mv_s / 1e9, (I.total_s - I.mv_s) * 1e3 / I.tokens);
            std::fprintf(stderr, "    of 'other': deltanet recurrence %.2f ms, attention core %.2f ms per layer-token\n", I.t_rec * 1e3 / I.tokens, I.t_att * 1e3 / I.tokens);
        }
    }
}

void CpuDecoder::replay(const CpuLayerWeights& w, CpuLayerState& s, const float* tape, int T, int keep) {
    if (w.kind != LayerKind::DeltaNet) return;
    const DeltaNetDims& d = w.dn;
    const size_t cd = static_cast<size_t>(d.conv_dim()), vd = static_cast<size_t>(d.value_dim()), vh = static_cast<size_t>(d.v_heads);
    const float* qkv = tape;
    const float* z = qkv + T * cd;
    const float* b = z + T * vd;
    const float* a = b + T * vh;
    for (int t = 0; t < keep; ++t) dn_core(*impl_, w, s, qkv + t * cd, z + t * vd, b + t * vh, a + t * vh, nullptr);
}

void CpuDecoder::matvec_multi(const HostMatrix& M, const float* x, int T, float* y) {
    Impl& I = *impl_;
    if (static_cast<int>(I.acts.size()) < T) I.acts.resize(static_cast<size_t>(T));
    for (int t = 0; t < T; ++t) I.acts[t].quantize(x + static_cast<size_t>(t) * M.cols, M.cols);
    I.matvecs_multi({{&M, y}}, T);
}

}  // namespace dara::model
