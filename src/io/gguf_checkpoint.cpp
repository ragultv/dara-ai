#include "io/gguf_checkpoint.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace dara::io {

uint64_t gguf_row_bytes(uint32_t type, uint64_t cols) {
    struct B { uint32_t type; uint64_t block_weights, block_bytes; };
    static const B kinds[] = {{0, 1, 4}, {1, 1, 2}, {30, 1, 2}, {8, 32, 34}, {12, 256, 144}, {13, 256, 176}, {14, 256, 210}};
    for (const B& b : kinds)
        if (b.type == type) {
            if (cols % b.block_weights != 0) throw std::runtime_error("gguf: row length is not a whole number of blocks");
            return cols / b.block_weights * b.block_bytes;
        }
    return 0;
}

GgufCheckpoint::GgufCheckpoint(const std::string& path) : hdr_(read_gguf_header(path)), path_(path) {
    const uint64_t file_size = std::filesystem::file_size(path);
    for (const auto& t : hdr_.tensors) {
        Info in;
        in.dims = t.dims;
        in.type = t.type;
        in.offset = hdr_.data_offset + t.offset;
        const uint64_t rb = gguf_row_bytes(t.type, in.cols());
        if (rb == 0) {  // a tensor type this runtime cannot decode: record it, fail only if somebody asks for it
            in.bytes = 0;
        } else {
            in.bytes = rb * in.rows();
            if (in.offset + in.bytes > file_size) throw std::runtime_error("gguf: tensor " + t.name + " extends past the end of the file");
        }
        if (!tensors_.emplace(t.name, std::move(in)).second) throw std::runtime_error("gguf: duplicate tensor " + t.name);
    }
    f_.open(path, std::ios::binary);
    if (!f_) throw std::runtime_error("gguf: cannot open " + path);
}

const GgufCheckpoint::Info& GgufCheckpoint::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) throw std::runtime_error("gguf: no tensor " + name);
    return it->second;
}

std::vector<uint8_t> GgufCheckpoint::read(const std::string& name) const {
    const Info& in = info(name);
    if (in.bytes == 0) throw std::runtime_error("gguf: unsupported type " + std::to_string(in.type) + " for " + name);
    std::vector<uint8_t> v(in.bytes);
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(in.offset));
    f_.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(in.bytes));
    if (!f_) throw std::runtime_error("gguf: short read of " + name);
    return v;
}

void GgufCheckpoint::read_into(const std::string& name, void* dst) const {
    const Info& in = info(name);
    if (in.bytes == 0) throw std::runtime_error("gguf: unsupported type " + std::to_string(in.type) + " for " + name);
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(in.offset));
    f_.read(static_cast<char*>(dst), static_cast<std::streamsize>(in.bytes));
    if (!f_) throw std::runtime_error("gguf: short read of " + name);
}

std::vector<uint8_t> GgufCheckpoint::read_rows(const std::string& name, uint64_t row0, uint64_t nrows) const {
    const Info& in = info(name);
    if (in.bytes == 0) throw std::runtime_error("gguf: unsupported type for " + name);
    if (row0 + nrows > in.rows()) throw std::out_of_range("gguf: row range outside " + name);
    const uint64_t rb = gguf_row_bytes(in.type, in.cols());
    std::vector<uint8_t> v(rb * nrows);
    f_.clear();
    f_.seekg(static_cast<std::streamoff>(in.offset + row0 * rb));
    f_.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size()));
    if (!f_) throw std::runtime_error("gguf: short row read of " + name);
    return v;
}

std::vector<float> GgufCheckpoint::read_f32(const std::string& name) const {
    const Info& in = info(name);
    if (in.type != 0) throw std::runtime_error("gguf: " + name + " is not F32");
    const auto raw = read(name);
    std::vector<float> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

}  // namespace dara::io
