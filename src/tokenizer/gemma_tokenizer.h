#pragma once
// GGUF-native tokenizer for Gemma 4 (tokenizer.ggml.model = "gemma4"): SentencePiece-style BPE. Vocabulary, merges, token types and
// special ids all come from the GGUF file; Hugging Face files are only a test oracle.
//
// Pipeline (as the reference tokenizer.json defines it):
//   * special tokens are cut out of the RAW text (leftmost-longest): USER-DEFINED always, CONTROL only when parse_special is true
//   * each remaining segment: every ' ' becomes '▁' (U+2581); there is NO regex pre-tokenization (the split on ' ' finds nothing after
//     the replacement), so BPE runs over the whole segment
//   * initial symbols are code points; a code point missing from the vocabulary becomes its UTF-8 bytes as <0xHH> byte tokens
//   * ranked pairwise merges, lowest rank first
//   * <bos> is prepended (tokenizer.ggml.add_bos_token)
// Decoding reverses it: <0xHH> tokens are raw bytes, '▁' becomes ' ', and the byte stream is decoded as lossy UTF-8.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/gguf.h"

namespace dara::tokenizer {

class GemmaTokenizer {
public:
    // Throws std::runtime_error on an unsupported tokenizer model or inconsistent data.
    static GemmaTokenizer from_gguf(const dara::io::GgufFile& g);

    // Text -> ids (with <bos> first when add_bos). Ill-formed UTF-8 becomes U+FFFD.
    std::vector<int32_t> encode(const std::string& utf8, bool parse_special = false, bool add_bos = true) const;

    // Ids -> UTF-8. Special tokens print their literal text when render_special is true. Throws std::out_of_range on a bad id.
    std::string decode(const std::vector<int32_t>& ids, bool render_special = true) const;

    // Streaming detokenizer: only complete UTF-8 characters are returned; flush() emits what is left.
    class StreamDecoder {
    public:
        explicit StreamDecoder(const GemmaTokenizer& t, bool render_special = true) : t_(t), render_special_(render_special) {}
        std::string push(int32_t id);
        std::string flush();

    private:
        const GemmaTokenizer& t_;
        bool render_special_;
        std::string pending_;
    };

    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }          // the GGUF's end-of-generation token (<turn|> for the instruct model)
    bool is_special(int32_t id) const;
    bool is_user_defined(int32_t id) const { return id >= 0 && static_cast<size_t>(id) < types_.size() && types_[static_cast<size_t>(id)] == 4; }  // recognized even without parse_special
    size_t vocab_size() const { return tokens_.size(); }
    const std::string& token_text(int32_t id) const { return tokens_.at(static_cast<size_t>(id)); }

private:
    struct Special { std::string text; int32_t id; bool user_defined; };
    struct Merge { int32_t rank, result; };
    void encode_segment(const std::string& raw_utf8, std::vector<int32_t>& out) const;
    const Special* find_special(const std::string& s, size_t i, bool parse_special) const;
    void append_token_bytes(int32_t id, bool render_special, std::string& out) const;

    std::vector<std::string> tokens_;
    std::vector<int> types_;                          // GGUF token_type: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte
    std::unordered_map<std::string, int32_t> vocab_;  // normal + byte tokens
    std::unordered_map<uint64_t, Merge> merges_;      // (left id, right id) -> rank, merged id
    int32_t byte_token_[256];                         // id of <0xHH> for each byte (-1 if absent)
    std::vector<Special> specials_;                   // longest first
    bool any_first_[256] = {};
    bool user_first_[256] = {};
    int32_t bos_ = -1, eos_ = -1;
};

}  // namespace dara::tokenizer
