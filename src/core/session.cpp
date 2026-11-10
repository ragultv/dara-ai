#include "core/session.h"

#include <algorithm>
#include <stdexcept>

namespace dara::core {

BlockPool::BlockPool(uint32_t num_blocks, uint32_t block_tokens)
    : total_(num_blocks), block_tokens_(block_tokens) {
    if (block_tokens == 0) throw std::invalid_argument("block_tokens must be > 0");
    free_.reserve(num_blocks);
    for (uint32_t i = num_blocks; i > 0; --i) free_.push_back(i - 1);
}

bool BlockPool::alloc(uint32_t& id) {
    if (free_.empty()) return false;
    id = free_.back();
    free_.pop_back();
    return true;
}

void BlockPool::release(uint32_t id) {
    if (id >= total_) throw std::out_of_range("block id");
    free_.push_back(id);
}

uint64_t hash_tokens(const std::vector<int32_t>& t, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n && i < t.size(); ++i) {
        h ^= static_cast<uint32_t>(t[i]);
        h *= 1099511628211ull;
    }
    return h;
}

bool Session::append(const std::vector<int32_t>& toks) {
    const uint64_t new_total = tokens_.size() + toks.size();
    const uint64_t need = pool_->blocks_for(new_total);
    const size_t had = table_.size();
    while (table_.size() < need) {
        uint32_t id;
        if (!pool_->alloc(id)) {
            while (table_.size() > had) {
                pool_->release(table_.back());
                table_.pop_back();
            }
            return false;
        }
        table_.push_back(id);
    }
    tokens_.insert(tokens_.end(), toks.begin(), toks.end());
    return true;
}

void Session::truncate(uint64_t n) {
    if (n >= tokens_.size()) return;
    // Backend state describes the old length. Drop it so a backend cannot silently continue from
    // stale state; rewind through IModelBackend::restore (which reinstalls a matching state).
    backend_state.reset();
    tokens_.resize(n);
    const uint64_t keep = pool_->blocks_for(n);
    while (table_.size() > keep) {
        pool_->release(table_.back());
        table_.pop_back();
    }
}

uint64_t Session::reuse_prefix(const std::vector<int32_t>& next) {
    const size_t lim = std::min(tokens_.size(), next.size());
    size_t i = 0;
    while (i < lim && tokens_[i] == next[i]) ++i;
    // Always leave at least one token to prefill so the engine produces fresh logits.
    if (i == next.size() && i > 0) --i;
    truncate(i);
    return i;
}

}  // namespace dara::core
