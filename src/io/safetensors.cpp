#include "io/safetensors.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace dara::io {
namespace {

// Minimal parser for the safetensors header: objects, strings, unsigned ints, arrays.
struct P {
    const std::string& s;
    size_t i = 0;
    [[noreturn]] void fail(const char* m) { throw std::runtime_error(std::string("safetensors: ") + m); }
    void ws() { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
    void expect(char c) { ws(); if (i >= s.size() || s[i] != c) fail("malformed header json"); ++i; }
    bool peek(char c) { ws(); return i < s.size() && s[i] == c; }
    bool comma() { if (peek(',')) { ++i; return true; } return false; }
    std::string str() {
        expect('"');
        std::string o;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) ++i;
            o += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        ++i;
        return o;
    }
    uint64_t num() {
        ws();
        if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) fail("expected number");
        uint64_t v = 0;
        while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
            uint64_t d = static_cast<uint64_t>(s[i++] - '0');
            if (v > (UINT64_MAX - d) / 10) fail("number overflow");
            v = v * 10 + d;
        }
        return v;
    }
    std::vector<uint64_t> nums() {
        std::vector<uint64_t> v;
        expect('[');
        if (peek(']')) { ++i; return v; }
        do v.push_back(num()); while (comma());
        expect(']');
        return v;
    }
    void skip_value() {
        if (peek('"')) { str(); return; }
        if (peek('{')) {
            ++i;
            if (peek('}')) { ++i; return; }
            do { str(); expect(':'); skip_value(); } while (comma());
            expect('}');
            return;
        }
        if (peek('[')) {
            ++i;
            if (peek(']')) { ++i; return; }
            do skip_value(); while (comma());
            expect(']');
            return;
        }
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    }
};

}  // namespace

SafetensorsFile read_safetensors_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("safetensors: cannot open " + path);
    f.seekg(0, std::ios::end);
    auto size = static_cast<uint64_t>(f.tellg());
    f.seekg(0);
    uint64_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 8);
    if (!f || size < 8 || n > size - 8 || n > (100ull << 20))
        throw std::runtime_error("safetensors: bad header length");
    std::string h(n, '\0');
    f.read(h.data(), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("safetensors: truncated header");

    SafetensorsFile out;
    out.data_offset = 8 + n;
    P p{h};
    p.expect('{');
    if (!p.peek('}')) {
        do {
            auto key = p.str();
            p.expect(':');
            if (key == "__metadata__") { p.skip_value(); continue; }
            StTensor t;
            t.name = key;
            p.expect('{');
            do {
                auto k = p.str();
                p.expect(':');
                if (k == "dtype") t.dtype = p.str();
                else if (k == "shape") t.shape = p.nums();
                else if (k == "data_offsets") {
                    auto o = p.nums();
                    if (o.size() != 2 || o[0] > o[1] || o[1] > size - out.data_offset)
                        throw std::runtime_error("safetensors: bad data_offsets for " + key);
                    t.begin = o[0];
                    t.end = o[1];
                } else p.skip_value();
            } while (p.comma());
            p.expect('}');
            out.tensors.push_back(std::move(t));
        } while (p.comma());
    }
    p.expect('}');
    return out;
}

}  // namespace dara::io
