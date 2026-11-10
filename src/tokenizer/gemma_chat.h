#pragma once
// Gemma 4 chat template for text conversations, implemented from the template shipped with the model (chat_template.jinja) and validated
// against the Hugging Face renderer (tests/gemma_chat_test.cpp).
//
// Supported: system / developer / user / assistant text messages, optional thinking mode, generation prompt. NOT supported (rejected with an
// exception rather than approximated): tools and tool calls/responses, images, audio, video, reasoning fields, non-text content parts.
//
// Structural tokens (<bos>, <|turn>, <turn|>, <|think|>) are emitted by id; message CONTENT is tokenized as plain text, so a user
// cannot inject control tokens by typing them (the reference renders everything to one string and lets the tokenizer parse it).
#include <cstdint>
#include <string>
#include <vector>

#include "tokenizer/gemma_tokenizer.h"

namespace dara::tokenizer {

struct ChatMessage {
    std::string role;     // "system", "developer", "user", "assistant"
    std::string content;  // UTF-8 text
};

struct ChatPiece {
    std::string text;
    bool structural;  // true: one special token of the template (<bos>, <|turn>, <turn|>, <|think|>); false: plain text (role names,
                      // newlines, message content), which the tokenizer encodes as one run
};

// The template's output as ordered pieces. Throws std::invalid_argument on an unsupported role.
std::vector<ChatPiece> render_gemma_chat_pieces(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true, bool enable_thinking = false);

// The same output as one string (what the reference renderer returns).
std::string render_gemma_chat(const std::vector<ChatMessage>& messages, bool add_generation_prompt = true, bool enable_thinking = false);

// Token ids: structural pieces by their special-token ids, content as plain text.
std::vector<int32_t> tokenize_gemma_chat(const GemmaTokenizer& tok, const std::vector<ChatMessage>& messages, bool add_generation_prompt = true,
                                         bool enable_thinking = false);

}  // namespace dara::tokenizer
