#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace dara::core {

// Fixed-size token blocks shared by every session. Kernels receive a page table of block ids.
class BlockPool {
public:
    BlockPool(uint32_t num_blocks, uint32_t block_tokens);
    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t free_blocks() const { return static_cast<uint32_t>(free_.size()); }
    uint32_t total_blocks() const { return total_; }
    bool can_fit(uint64_t tokens) const { return blocks_for(tokens) <= free_blocks(); }
    uint64_t blocks_for(uint64_t tokens) const { return (tokens + block_tokens_ - 1) / block_tokens_; }
    bool alloc(uint32_t& id);
    void release(uint32_t id);

private:
    uint32_t total_, block_tokens_;
    std::vector<uint32_t> free_;
};

// Architecture-specific per-session state (DeltaNet recurrent state for Qwen, window bookkeeping
// for Gemma, ...). Defined by each backend; the Session only stores it. Real backends keep
// device buffers behind it.
class BackendState {
public:
    virtual ~BackendState() = default;
    virtual std::unique_ptr<BackendState> clone() const = 0;
    // Identifies the backend/model that produced it; restore refuses a mismatch.
    virtual std::string owner() const = 0;
};

uint64_t hash_tokens(const std::vector<int32_t>& t, uint64_t n);  // FNV-1a over t[0, n)

// A resumable point in a session: KV position plus a copy of the backend state at that position.
struct Checkpoint {
    uint64_t n_tokens = 0;
    uint64_t prefix_hash = 0;                      // guards against restoring into a diverged session
    std::shared_ptr<const BackendState> state;     // may be null for stateless backends
};

class Session {
public:
    explicit Session(BlockPool& pool) : pool_(&pool) {}
    ~Session() { clear(); }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    const std::vector<int32_t>& tokens() const { return tokens_; }
    const std::vector<uint32_t>& page_table() const { return table_; }
    std::unique_ptr<BackendState> backend_state;  // owned by the session, populated by the backend

    // Appends tokens, allocating blocks. All-or-nothing: on failure the session is unchanged.
    bool append(const std::vector<int32_t>& toks);
    // Drops everything past n tokens and frees whole blocks no longer needed. Used for cancel
    // (keep only delivered tokens) and for rejected speculative drafts.
    void truncate(uint64_t n);
    // Returns how many leading tokens of `next` match the retained history exactly, and trims
    // the session to that prefix so only the remainder needs prefill.
    uint64_t reuse_prefix(const std::vector<int32_t>& next);
    void clear() { truncate(0); }

private:
    BlockPool* pool_;
    std::vector<int32_t> tokens_;
    std::vector<uint32_t> table_;
};

}  // namespace dara::core
