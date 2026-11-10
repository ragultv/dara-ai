#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace dara::io {

struct StTensor {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t begin = 0, end = 0;  // relative to data section
};

struct SafetensorsFile {
    uint64_t data_offset = 0;  // absolute file offset of data section
    std::vector<StTensor> tensors;
};

SafetensorsFile read_safetensors_header(const std::string& path);

}  // namespace dara::io
