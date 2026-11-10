#include "tokenizer/unicode.h"

#include <algorithm>
#include <cstdint>

namespace dara::tokenizer {
namespace {
#include "tokenizer/unicode_data.inc"

template <size_t N>
bool in_ranges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

constexpr uint32_t SBASE = 0xAC00, LBASE = 0x1100, VBASE = 0x1161, TBASE = 0x11A7;
constexpr uint32_t LCOUNT = 19, VCOUNT = 21, TCOUNT = 28, NCOUNT = VCOUNT * TCOUNT, SCOUNT = LCOUNT * NCOUNT;

uint8_t ccc_of(uint32_t cp) {
    size_t lo = 0, hi = sizeof(kCccRuns) / sizeof(kCccRuns[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < kCccRuns[mid].lo) hi = mid;
        else if (cp > kCccRuns[mid].hi) lo = mid + 1;
        else return kCccRuns[mid].ccc;
    }
    return 0;
}

bool decomposition_of(uint32_t cp, const uint32_t*& begin, size_t& len) {
    size_t lo = 0, hi = sizeof(kDecompIndex) / sizeof(kDecompIndex[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < kDecompIndex[mid].cp) hi = mid;
        else if (cp > kDecompIndex[mid].cp) lo = mid + 1;
        else { begin = &kDecompPool[kDecompIndex[mid].offset]; len = kDecompIndex[mid].len; return true; }
    }
    return false;
}

bool compose_pair(uint32_t a, uint32_t b, uint32_t& c) {
    size_t lo = 0, hi = sizeof(kCompose) / sizeof(kCompose[0]);
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const ComposePair& p = kCompose[mid];
        if (a < p.a || (a == p.a && b < p.b)) hi = mid;
        else if (a > p.a || b > p.b) lo = mid + 1;
        else { c = p.c; return true; }
    }
    return false;
}
}  // namespace

bool is_letter(uint32_t cp) { return in_ranges(kLetterRanges, cp); }
bool is_mark(uint32_t cp) { return in_ranges(kMarkRanges, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumberRanges, cp); }
bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::vector<uint32_t> nfc(const std::vector<uint32_t>& in) {
    // Fast path: pure ASCII is invariant under NFC and is by far the common case.
    bool ascii = true;
    for (uint32_t c : in) if (c >= 0x80) { ascii = false; break; }
    if (ascii) return in;

    // 1. canonical decomposition
    std::vector<uint32_t> d;
    d.reserve(in.size() + 8);
    for (uint32_t cp : in) {
        if (cp >= SBASE && cp < SBASE + SCOUNT) {
            const uint32_t si = cp - SBASE;
            d.push_back(LBASE + si / NCOUNT);
            d.push_back(VBASE + (si % NCOUNT) / TCOUNT);
            if (si % TCOUNT) d.push_back(TBASE + si % TCOUNT);
            continue;
        }
        const uint32_t* b; size_t n;
        if (decomposition_of(cp, b, n)) d.insert(d.end(), b, b + n);
        else d.push_back(cp);
    }
    // 2. canonical ordering: stable sort of each run of non-starters by combining class
    for (size_t i = 0; i < d.size();) {
        if (ccc_of(d[i]) == 0) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && ccc_of(d[j]) != 0) ++j;
        std::stable_sort(d.begin() + static_cast<long>(i), d.begin() + static_cast<long>(j),
                         [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
        i = j;
    }
    // 3. canonical composition
    std::vector<uint32_t> out;
    out.reserve(d.size());
    long starter = -1;   // index in `out` of the last starter
    int last_ccc = -1;   // ccc of the last character appended after the starter; -1 if none
    for (uint32_t cp : d) {
        const int c = ccc_of(cp);
        if (starter >= 0 && (last_ccc == -1 || last_ccc < c)) {  // not blocked
            const uint32_t st = out[static_cast<size_t>(starter)];
            uint32_t comp = 0;
            bool ok = false;
            if (st >= LBASE && st < LBASE + LCOUNT && cp >= VBASE && cp < VBASE + VCOUNT) {
                comp = SBASE + ((st - LBASE) * VCOUNT + (cp - VBASE)) * TCOUNT;
                ok = true;
            } else if (st >= SBASE && st < SBASE + SCOUNT && (st - SBASE) % TCOUNT == 0 && cp > TBASE && cp < TBASE + TCOUNT) {
                comp = st + (cp - TBASE);
                ok = true;
            } else {
                ok = compose_pair(st, cp, comp);
            }
            if (ok) { out[static_cast<size_t>(starter)] = comp; continue; }
        }
        if (c == 0) { starter = static_cast<long>(out.size()); last_ccc = -1; }
        else last_ccc = c;
        out.push_back(cp);
    }
    return out;
}

// ---- UTF-8 --------------------------------------------------------------------------------------------------
size_t utf8_next(const uint8_t* p, size_t n, uint32_t& cp, Utf8Status& status) {
    status = Utf8Status::Ok;
    if (n == 0) { status = Utf8Status::Truncated; cp = 0; return 0; }
    const uint8_t b0 = p[0];
    if (b0 < 0x80) { cp = b0; return 1; }
    size_t need;
    uint32_t lo2 = 0x80, hi2 = 0xBF;  // allowed range of the second byte
    if (b0 >= 0xC2 && b0 <= 0xDF) { need = 2; cp = b0 & 0x1F; }
    else if (b0 >= 0xE0 && b0 <= 0xEF) { need = 3; cp = b0 & 0x0F; if (b0 == 0xE0) lo2 = 0xA0; if (b0 == 0xED) hi2 = 0x9F; }
    else if (b0 >= 0xF0 && b0 <= 0xF4) { need = 4; cp = b0 & 0x07; if (b0 == 0xF0) lo2 = 0x90; if (b0 == 0xF4) hi2 = 0x8F; }
    else { cp = 0xFFFD; status = Utf8Status::Invalid; return 1; }  // 0x80-0xC1, 0xF5-0xFF
    size_t i = 1;
    for (; i < need; ++i) {
        if (i >= n) { status = Utf8Status::Truncated; cp = 0; return n; }  // valid prefix, input ended
        const uint8_t b = p[i];
        const uint32_t lo = i == 1 ? lo2 : 0x80, hi = i == 1 ? hi2 : 0xBF;
        if (b < lo || b > hi) { cp = 0xFFFD; status = Utf8Status::Invalid; return i; }  // maximal subpart so far
        cp = (cp << 6) | (b & 0x3F);
    }
    return need;
}

std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data());
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp; Utf8Status st;
        const size_t used = utf8_next(p + i, s.size() - i, cp, st);
        out.push_back(st == Utf8Status::Ok ? cp : 0xFFFD);  // a truncated tail is one replacement character
        i += used;
    }
    return out;
}

void utf8_append(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += static_cast<char>(cp);
    else if (cp < 0x800) { o += static_cast<char>(0xC0 | (cp >> 6)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += static_cast<char>(0xE0 | (cp >> 12)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { o += static_cast<char>(0xF0 | (cp >> 18)); o += static_cast<char>(0x80 | ((cp >> 12) & 0x3F)); o += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); o += static_cast<char>(0x80 | (cp & 0x3F)); }
}

std::string codepoints_to_utf8(const std::vector<uint32_t>& cps) {
    std::string s;
    s.reserve(cps.size());
    for (uint32_t c : cps) utf8_append(s, c);
    return s;
}

}  // namespace dara::tokenizer
