#pragma once
// Attention KV cache interface. Attention reads the cache as a list of SPANS (contiguous runs), not as one flat
// array, so a paged implementation (fixed-size blocks addressed through a page table) can sit behind the same
// interface without pretending its memory is contiguous. Layout inside a span: [position][kv_heads][head_dim].
#include <cstddef>
#include <vector>

namespace dara::model {

struct KvSpan {
    const float* k = nullptr;  // first position of the span; positions are contiguous within it
    const float* v = nullptr;
    size_t start = 0;          // absolute position of the span's first entry
    size_t count = 0;
};

class IAttentionCache {
public:
    virtual ~IAttentionCache() = default;
    virtual size_t size() const = 0;
    // Appends one position (kv_heads*head_dim floats each for k and v). Throws std::length_error when full.
    virtual void append(const float* k, const float* v) = 0;
    // Drops positions >= n. Restoring a checkpoint is truncating to its recorded length.
    virtual void truncate(size_t n) = 0;
    // Spans covering positions [start, start + length) in order. A contiguous cache returns one span; a paged cache
    // returns one per page touched.
    virtual std::vector<KvSpan> spans(size_t start, size_t length) const = 0;
};

}  // namespace dara::model
