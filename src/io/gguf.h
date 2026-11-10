#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dara::io {

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;      // ggml type id
    uint64_t offset = 0;    // relative to data section
};

struct GgufFile {
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    std::vector<std::pair<std::string, std::string>> metadata;  // scalar values stringified; arrays summarised
    // Arrays under "tokenizer.*" are kept in full: the vocabulary and merges are the tokenizer's only source.
    std::map<std::string, std::vector<std::string>> string_arrays;  // e.g. tokenizer.ggml.tokens, tokenizer.ggml.merges
    std::map<std::string, std::vector<int64_t>> int_arrays;         // e.g. tokenizer.ggml.token_type
    std::vector<GgufTensor> tensors;
    uint64_t alignment = 32;
    uint64_t data_offset = 0;  // absolute file offset of tensor data
};

// Reads header, metadata and tensor table only. Throws std::runtime_error on malformed input.
GgufFile read_gguf_header(const std::string& path);

}  // namespace dara::io
