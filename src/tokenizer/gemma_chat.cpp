#include "tokenizer/gemma_chat.h"

#include <stdexcept>

#include "tokenizer/unicode.h"

namespace dara::tokenizer {
namespace {

// Jinja's `trim` is Python str.strip(): every Unicode whitespace character (White_Space plus U+001C..U+001F).
bool py_space(uint32_t cp) { return is_space(cp) || (cp >= 0x1C && cp <= 0x1F); }

std::string trim(const std::string& s) {
    const std::vector<uint32_t> cps = utf8_to_codepoints(s);
    size_t b = 0, e = cps.size();
    while (b < e && py_space(cps[b])) ++b;
    while (e > b && py_space(cps[e - 1])) --e;
    return codepoints_to_utf8(std::vector<uint32_t>(cps.begin() + static_cast<long>(b), cps.begin() + static_cast<long>(e)));
}

// The template's strip_thinking macro: drop every <|channel> ... <channel|> block (an unterminated one drops to the end), then trim.
std::string strip_thinking(const std::string& text) {
    const std::string open = "<|channel>", close = "<channel|>";
    std::string result;
    size_t pos = 0;
    while (true) {
        const size_t c = text.find(close, pos);
        const std::string part = text.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        const size_t o = part.find(open);
        result += o == std::string::npos ? part : part.substr(0, o);
        if (c == std::string::npos) break;
        pos = c + close.size();
    }
    return trim(result);
}

}  // namespace

std::vector<ChatPiece> render_gemma_chat_pieces(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_thinking) {
    for (const ChatMessage& m : messages)
        if (m.role != "system" && m.role != "developer" && m.role != "user" && m.role != "assistant")
            throw std::invalid_argument("unsupported chat role '" + m.role + "' (tool messages are not supported)");

    std::vector<ChatPiece> out;
    auto special = [&](const char* s) { out.push_back({s, true}); };
    auto text = [&](std::string s) { out.push_back({std::move(s), false}); };

    special("<bos>");
    size_t first = 0;
    const bool has_system = !messages.empty() && (messages[0].role == "system" || messages[0].role == "developer");
    if (enable_thinking || has_system) {
        special("<|turn>");
        text("system\n");
        if (enable_thinking) { special("<|think|>"); text("\n"); }
        if (has_system) {
            text(trim(messages[0].content));
            first = 1;
        }
        special("<turn|>");
        text("\n");
    }
    std::string prev_role;  // the previous message's own role (the template's prev_non_tool_role)
    for (size_t i = first; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        const bool model = m.role == "assistant";
        const std::string role = model ? "model" : m.role;
        const bool continue_same_turn = model && prev_role == "assistant";
        if (!continue_same_turn) {
            special("<|turn>");
            text(role + "\n");
        }
        text(model ? strip_thinking(m.content) : trim(m.content));
        const bool continues_into_next = model && i + 1 < messages.size() && messages[i + 1].role == "assistant";
        if (!continues_into_next) { special("<turn|>"); text("\n"); }
        prev_role = m.role;
    }
    if (add_generation_prompt) {
        special("<|turn>");
        text("model\n");
    }
    return out;
}

std::string render_gemma_chat(const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_thinking) {
    std::string s;
    for (const ChatPiece& p : render_gemma_chat_pieces(messages, add_generation_prompt, enable_thinking)) s += p.text;
    return s;
}

std::vector<int32_t> tokenize_gemma_chat(const GemmaTokenizer& tok, const std::vector<ChatMessage>& messages, bool add_generation_prompt, bool enable_thinking) {
    std::vector<int32_t> ids;
    std::string run;  // adjacent plain text is tokenized as ONE run, because BPE merges can cross the template/content boundary
    auto flush = [&] {
        if (run.empty()) return;
        const std::vector<int32_t> piece = tok.encode(run, /*parse_special=*/false, /*add_bos=*/false);
        ids.insert(ids.end(), piece.begin(), piece.end());
        run.clear();
    };
    for (const ChatPiece& p : render_gemma_chat_pieces(messages, add_generation_prompt, enable_thinking)) {
        if (!p.structural) { run += p.text; continue; }
        flush();
        const std::vector<int32_t> sp = tok.encode(p.text, /*parse_special=*/true, /*add_bos=*/false);
        if (sp.size() != 1) throw std::runtime_error("the vocabulary has no single special token for " + p.text);
        ids.push_back(sp[0]);
    }
    flush();
    return ids;
}

}  // namespace dara::tokenizer
