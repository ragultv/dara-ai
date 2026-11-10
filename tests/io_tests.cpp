// Dependency-free tests for now; migrate to GoogleTest once vcpkg is baselined.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#include "io/gguf.h"
#include "io/safetensors.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++failures; } } while (0)

template <class T> static void put(std::ofstream& f, T v) { f.write(reinterpret_cast<char*>(&v), sizeof v); }
static void put_str(std::ofstream& f, const std::string& s) {
    put<uint64_t>(f, s.size());
    f.write(s.data(), (std::streamsize)s.size());
}

static void write_st(const char* path, const std::string& h, int payload) {
    std::ofstream f(path, std::ios::binary);
    put<uint64_t>(f, h.size());
    f.write(h.data(), (std::streamsize)h.size());
    for (int i = 0; i < payload; ++i) f.put(0);
}

static void test_safetensors() {
    std::string h = "{\"__metadata__\":{\"format\":\"pt\"},\"w\":{\"dtype\":\"F16\",\"shape\":[2,3],\"data_offsets\":[0,12]}}";
    write_st("t.safetensors", h, 12);
    auto s = dara::io::read_safetensors_header("t.safetensors");
    CHECK(s.tensors.size() == 1);
    CHECK(s.tensors[0].name == "w" && s.tensors[0].dtype == "F16");
    CHECK(s.tensors[0].shape.size() == 2 && s.tensors[0].shape[1] == 3);
    CHECK(s.tensors[0].end == 12);
    CHECK(s.data_offset == 8 + h.size());
}

static void test_safetensors_rejects_oob() {
    write_st("bad.safetensors", "{\"w\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,999]}}", 0);
    bool threw = false;
    try { dara::io::read_safetensors_header("bad.safetensors"); } catch (...) { threw = true; }
    CHECK(threw);
}

static void test_gguf() {
    std::ofstream f("t.gguf", std::ios::binary);
    f.write("GGUF", 4);
    put<uint32_t>(f, 3); put<uint64_t>(f, 1); put<uint64_t>(f, 2);
    put_str(f, "general.architecture"); put<uint32_t>(f, 8); put_str(f, "qwen35");
    put_str(f, "x.arr"); put<uint32_t>(f, 9); put<uint32_t>(f, 4); put<uint64_t>(f, 3);
    put<uint32_t>(f, 1); put<uint32_t>(f, 2); put<uint32_t>(f, 3);
    put_str(f, "blk.0.w"); put<uint32_t>(f, 2); put<uint64_t>(f, 4); put<uint64_t>(f, 8);
    put<uint32_t>(f, 12); put<uint64_t>(f, 0);
    f.close();
    auto g = dara::io::read_gguf_header("t.gguf");
    CHECK(g.version == 3 && g.tensor_count == 1);
    CHECK(g.metadata.size() == 2 && g.metadata[0].second == "qwen35");
    CHECK(g.tensors[0].dims.size() == 2 && g.tensors[0].dims[1] == 8 && g.tensors[0].type == 12);
    CHECK(g.data_offset % 32 == 0);
}

static void test_gguf_bad_magic() {
    std::ofstream f("bad.gguf", std::ios::binary);
    f.write("NOPE0000", 8);
    f.close();
    bool threw = false;
    try { dara::io::read_gguf_header("bad.gguf"); } catch (...) { threw = true; }
    CHECK(threw);
}

int main() {
    test_safetensors();
    test_safetensors_rejects_oob();
    test_gguf();
    test_gguf_bad_magic();
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
