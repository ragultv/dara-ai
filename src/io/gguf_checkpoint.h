#pragma once
// GGUF file with on-demand tensor reads (header parsed once; tensor bytes read from disk when asked), so a quantized
// model is streamed layer by layer like the BF16 checkpoint.
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "io/gguf.h"

namespace dara::io {

class GgufCheckpoint {
public:
    explicit GgufCheckpoint(const std::string& path);  // throws on a malformed header or a tensor outside the file

    struct Info {
        std::vector<uint64_t> dims;   // ggml order: dims[0] is the contiguous (inner) dimension = columns
        uint32_t type = 0;            // ggml type id (0 F32, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K, ...)
        uint64_t offset = 0;          // absolute file offset
        uint64_t bytes = 0;
        uint64_t rows() const { uint64_t r = 1; for (size_t i = 1; i < dims.size(); ++i) r *= dims[i]; return r; }
        uint64_t cols() const { return dims.empty() ? 0 : dims[0]; }
    };
    bool has(const std::string& name) const { return tensors_.count(name) != 0; }
    const Info& info(const std::string& name) const;
    std::vector<uint8_t> read(const std::string& name) const;                         // all bytes of a tensor
    void read_into(const std::string& name, void* dst) const;                         // all bytes straight into dst (info(name).bytes long)
    std::vector<uint8_t> read_rows(const std::string& name, uint64_t row0, uint64_t nrows) const;  // whole rows
    std::vector<float> read_f32(const std::string& name) const;                       // tensor must be F32
    const GgufFile& header() const { return hdr_; }

private:
    GgufFile hdr_;
    std::map<std::string, Info> tensors_;
    mutable std::ifstream f_;
    std::string path_;
};

// Bytes per row for a type with `cols` columns (cols must be a whole number of blocks); 0 if the type is unsupported.
uint64_t gguf_row_bytes(uint32_t type, uint64_t cols);

}  // namespace dara::io
