#pragma once
// GGUF-native byte-level BPE tokenizer. The vocabulary, merges, token types and special-token ids all come from the GGUF
// file; nothing is read from Hugging Face files at runtime (those are only a test oracle).
//
// What the GGUF does NOT carry is the pre-tokenizer pattern itself -- only its name (tokenizer.ggml.pre). This
// implementation hard-codes the pattern for the names it knows, exactly as llama.cpp does, and refuses any other name.
// Supported: "qwen35", i.e. the regex
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// preceded by NFC normalization (the checkpoint's normalizer), then the GPT-2 byte-to-unicode mapping and ranked merges.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "io/gguf.h"

namespace dara::tokenizer {

class Tokenizer {
public:
    // Throws std::runtime_error on an unsupported tokenizer model / pre-tokenizer name or inconsistent data.
    static Tokenizer from_gguf(const dara::io::GgufFile& g);

    // Text -> ids. Ill-formed UTF-8 input is replaced with U+FFFD. Special tokens follow llama.cpp's convention, driven by
    // the GGUF token types: USER-DEFINED tokens (e.g. "<think>", "<tool_call>") are always recognized; CONTROL tokens
    // (e.g. "<|im_end|>") only when parse_special is true, so user text cannot inject them. They are extracted from the RAW
    // text (leftmost-longest) BEFORE normalization, as the reference does; each text segment between them is then
    // NFC-normalized on its own, pre-tokenized and BPE-encoded. (The reference tokenizer recognizes a few more tokens
    // unconditionally -- <|fim_*|>, <|repo_name|>, <|file_sep|> -- which the GGUF types as CONTROL.)
    std::vector<int32_t> encode(const std::string& utf8, bool parse_special = false) const;

    // Ids -> UTF-8 text. All bytes are concatenated first and then decoded lossily (as the reference does), so a character
    // split across tokens decodes correctly. Special tokens print their literal text when render_special is true.
    // Throws std::out_of_range for an id outside the vocabulary.
    std::string decode(const std::vector<int32_t>& ids, bool render_special = true) const;

    // Streaming detokenizer: push ids one at a time and get back only complete UTF-8 characters. Bytes of a character
    // that is still arriving are held back until it completes; flush() emits whatever is left (as U+FFFD if incomplete).
    class StreamDecoder {
    public:
        explicit StreamDecoder(const Tokenizer& t, bool render_special = true) : t_(t), render_special_(render_special) {}
        std::string push(int32_t id);
        std::string flush();

    private:
        const Tokenizer& t_;
        bool render_special_;
        std::string pending_;  // raw bytes not yet emitted
    };

    // Pre-tokenization pieces of `utf8` (after NFC), each in the byte-level mapped form -- for validation against the
    // reference pre-tokenizer, independent of BPE.
    std::vector<std::string> pretokenize(const std::string& utf8) const;

    int32_t eos_id() const { return eos_; }
    size_t vocab_size() const { return tokens_.size(); }
    bool is_special(int32_t id) const;                 // control or user-defined
    const std::string& token_text(int32_t id) const { return tokens_.at(static_cast<size_t>(id)); }
    const std::string& pre_type() const { return pre_; }
    size_t special_count() const { return specials_.size(); }

private:
    struct Special { std::string text; int32_t id; bool user_defined; };
    // A segment between special tokens: NFC-normalize it on its own, then pre-tokenize + BPE.
    void encode_segment(const std::string& raw_utf8, std::vector<int32_t>& out) const;
    const Special* find_special(const std::string& s, size_t i, bool parse_special) const;
    void bpe(const std::string& piece_bytes, std::vector<int32_t>& out) const;
    void append_token_bytes(int32_t id, bool render_special, std::string& out) const;

    std::string pre_;
    std::vector<std::string> tokens_;
    std::vector<int> types_;                            // GGUF token_type: 1 normal, 3 control, 4 user-defined, 5 unused
    std::unordered_map<std::string, int32_t> vocab_;    // normal tokens only: BPE symbols
    struct Merge { int32_t rank, result; };
    std::unordered_map<uint64_t, Merge> merges_;        // (left id, right id) -> rank, merged id
    int32_t byte_token_[256];                           // id of the single-character token for each byte
    uint32_t byte_to_cp_[256];                          // GPT-2 byte -> unicode mapping
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;
    std::vector<Special> specials_;                     // sorted by length, longest first
    bool any_first_[256] = {};                          // bytes that can start a control or user-defined token
    bool user_first_[256] = {};                         // bytes that can start a user-defined token
    int32_t eos_ = -1;
};

}  // namespace dara::tokenizer
