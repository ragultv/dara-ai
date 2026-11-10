#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

#include "tokenizer/unicode.h"

namespace dara::tokenizer {
namespace {

// ---- the Qwen3.5 pre-tokenizer pattern, matched by hand ---------------------------------------------------------
// (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ | \p{N} | ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` |
// \s*[\r\n]+ | \s+(?!\S) | \s+        (alternatives tried in this order at every position, first match wins)
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }
inline bool is_lm(uint32_t c) { return is_letter(c) || is_mark(c); }
inline uint32_t lower_ascii(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

// Length of a contraction ('s 't 're 've 'm 'll 'd, case-insensitive) starting at i, or 0.
size_t match_contraction(const std::vector<uint32_t>& s, size_t i) {
    if (s[i] != '\'' || i + 1 >= s.size()) return 0;
    const uint32_t c1 = lower_ascii(s[i + 1]);
    // (?i) uses Unicode simple case folding: U+017F (long s) folds to 's'
    if (c1 == 's' || s[i + 1] == 0x17F) return 2;
    if (c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (i + 2 < s.size()) {
        const uint32_t c2 = lower_ascii(s[i + 2]);
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
    }
    return 0;
}

std::vector<std::pair<size_t, size_t>> split_qwen35(const std::vector<uint32_t>& s) {
    std::vector<std::pair<size_t, size_t>> out;
    const size_t n = s.size();
    size_t i = 0;
    while (i < n) {
        size_t end = 0;
        // 1. contractions
        if (size_t m = match_contraction(s, i)) end = i + m;
        // 2. optional single non-[\r\n L N] char, then one or more letters/marks
        if (!end) {
            size_t j = i;
            if (!is_nl(s[j]) && !is_letter(s[j]) && !is_number(s[j]) && j + 1 < n && is_lm(s[j + 1])) j = i + 1;  // take the prefix char
            if (j < n && is_lm(s[j])) {
                while (j < n && is_lm(s[j])) ++j;
                end = j;
            }
        }
        // 3. a single digit-like character
        if (!end && is_number(s[i])) end = i + 1;
        // 4. optional space, one or more "other" characters, then trailing newlines
        if (!end) {
            auto other = [&](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_mark(c) && !is_number(c); };
            size_t j = i;
            if (s[j] == ' ' && j + 1 < n && other(s[j + 1])) ++j;
            if (j < n && other(s[j])) {
                while (j < n && other(s[j])) ++j;
                while (j < n && is_nl(s[j])) ++j;
                end = j;
            }
        }
        // 5. whitespace ending in a newline: the run up to and including its last \r or \n
        size_t ws_end = i;
        if (!end && is_space(s[i])) {
            while (ws_end < n && is_space(s[ws_end])) ++ws_end;
            size_t last_nl = 0;
            bool has = false;
            for (size_t k = i; k < ws_end; ++k) if (is_nl(s[k])) { last_nl = k; has = true; }
            if (has) end = last_nl + 1;
        }
        // 6. \s+(?!\S): the whole run if it reaches the end of the text; otherwise all but its last character
        if (!end && is_space(s[i])) {
            if (ws_end == n) end = ws_end;
            else if (ws_end - i >= 2) end = ws_end - 1;
        }
        // 7. \s+
        if (!end && is_space(s[i])) end = ws_end;
        if (!end) end = i + 1;  // unreachable for valid patterns; guarantees progress
        out.emplace_back(i, end);
        i = end;
    }
    return out;
}

// GPT-2 byte <-> unicode mapping.
void build_byte_map(uint32_t byte_to_cp[256]) {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        byte_to_cp[b] = direct ? static_cast<uint32_t>(b) : static_cast<uint32_t>(256 + n++);
    }
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const dara::io::GgufFile& g) {
    auto meta = [&](const std::string& key) -> const std::string* {
        for (const auto& kv : g.metadata) if (kv.first == key) return &kv.second;
        return nullptr;
    };
    const std::string* model = meta("tokenizer.ggml.model");
    if (!model || *model != "gpt2") throw std::runtime_error("unsupported tokenizer model '" + (model ? *model : "<missing>") + "' (supported: gpt2)");
    const std::string* pre = meta("tokenizer.ggml.pre");
    if (!pre || *pre != "qwen35") throw std::runtime_error("unsupported pre-tokenizer '" + (pre ? *pre : "<missing>") + "' (supported: qwen35)");

    Tokenizer t;
    t.pre_ = *pre;
    auto ta = g.string_arrays.find("tokenizer.ggml.tokens");
    auto ma = g.string_arrays.find("tokenizer.ggml.merges");
    auto ty = g.int_arrays.find("tokenizer.ggml.token_type");
    if (ta == g.string_arrays.end() || ma == g.string_arrays.end() || ty == g.int_arrays.end())
        throw std::runtime_error("GGUF lacks tokenizer tokens / merges / token_type");
    t.tokens_ = ta->second;
    if (ty->second.size() != t.tokens_.size()) throw std::runtime_error("token_type length != token count");
    for (int64_t v : ty->second) t.types_.push_back(static_cast<int>(v));

    for (size_t i = 0; i < t.tokens_.size(); ++i) {
        if (t.types_[i] == 1) {
            if (!t.vocab_.emplace(t.tokens_[i], static_cast<int32_t>(i)).second) throw std::runtime_error("duplicate token text: " + t.tokens_[i]);
        } else if (t.types_[i] == 3 || t.types_[i] == 4) {
            t.specials_.push_back({t.tokens_[i], static_cast<int32_t>(i), t.types_[i] == 4});
        }
    }
    std::stable_sort(t.specials_.begin(), t.specials_.end(), [](const Special& a, const Special& b) { return a.text.size() > b.text.size(); });
    for (const Special& s : t.specials_)
        if (!s.text.empty()) {
            t.any_first_[static_cast<uint8_t>(s.text[0])] = true;
            if (s.user_defined) t.user_first_[static_cast<uint8_t>(s.text[0])] = true;
        }

    build_byte_map(t.byte_to_cp_);
    for (int b = 0; b < 256; ++b) {
        t.cp_to_byte_[t.byte_to_cp_[b]] = static_cast<uint8_t>(b);
        std::string s;
        utf8_append(s, t.byte_to_cp_[b]);
        auto it = t.vocab_.find(s);
        if (it == t.vocab_.end()) throw std::runtime_error("vocabulary lacks the single-byte token for byte " + std::to_string(b));
        t.byte_token_[b] = it->second;
    }

    t.merges_.reserve(ma->second.size() * 2);
    size_t skipped = 0;
    for (size_t r = 0; r < ma->second.size(); ++r) {
        const std::string& m = ma->second[r];
        const size_t sp = m.find(' ', 1);  // a merge half may itself start with the mapped-space char, never a literal space
        if (sp == std::string::npos) throw std::runtime_error("malformed merge: " + m);
        auto a = t.vocab_.find(m.substr(0, sp)), b = t.vocab_.find(m.substr(sp + 1)), c = t.vocab_.find(m.substr(0, sp) + m.substr(sp + 1));
        if (a == t.vocab_.end() || b == t.vocab_.end() || c == t.vocab_.end()) { ++skipped; continue; }
        t.merges_.emplace((static_cast<uint64_t>(a->second) << 32) | static_cast<uint32_t>(b->second), Merge{static_cast<int32_t>(r), c->second});
    }
    if (skipped) throw std::runtime_error(std::to_string(skipped) + " merges reference tokens missing from the vocabulary");

    for (const auto& kv : g.metadata)
        if (kv.first == "tokenizer.ggml.eos_token_id") t.eos_ = static_cast<int32_t>(std::stoll(kv.second));
    return t;
}

bool Tokenizer::is_special(int32_t id) const {
    return id >= 0 && static_cast<size_t>(id) < types_.size() && (types_[static_cast<size_t>(id)] == 3 || types_[static_cast<size_t>(id)] == 4);
}

void Tokenizer::bpe(const std::string& bytes, std::vector<int32_t>& out) const {
    const size_t n = bytes.size();
    if (n == 0) return;
    std::vector<int32_t> id(n);
    std::vector<int> prev(n), next(n);
    for (size_t i = 0; i < n; ++i) {
        id[i] = byte_token_[static_cast<uint8_t>(bytes[i])];
        prev[i] = static_cast<int>(i) - 1;
        next[i] = i + 1 < n ? static_cast<int>(i + 1) : -1;
    }
    struct Cand { int32_t rank; int left; int32_t lid, rid; };
    struct Cmp { bool operator()(const Cand& a, const Cand& b) const { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; } };
    std::priority_queue<Cand, std::vector<Cand>, Cmp> pq;
    auto push = [&](int l) {
        const int r = next[static_cast<size_t>(l)];
        if (r < 0) return;
        auto it = merges_.find((static_cast<uint64_t>(id[static_cast<size_t>(l)]) << 32) | static_cast<uint32_t>(id[static_cast<size_t>(r)]));
        if (it != merges_.end()) pq.push({it->second.rank, l, id[static_cast<size_t>(l)], id[static_cast<size_t>(r)]});
    };
    for (size_t i = 0; i + 1 < n; ++i) push(static_cast<int>(i));
    std::vector<char> dead(n, 0);
    while (!pq.empty()) {
        const Cand c = pq.top();
        pq.pop();
        const size_t l = static_cast<size_t>(c.left);
        if (dead[l] || id[l] != c.lid) continue;
        const int r = next[l];
        if (r < 0 || dead[static_cast<size_t>(r)] || id[static_cast<size_t>(r)] != c.rid) continue;  // stale candidate
        id[l] = merges_.at((static_cast<uint64_t>(c.lid) << 32) | static_cast<uint32_t>(c.rid)).result;
        dead[static_cast<size_t>(r)] = 1;
        next[l] = next[static_cast<size_t>(r)];
        if (next[l] >= 0) prev[static_cast<size_t>(next[l])] = c.left;
        if (prev[l] >= 0) push(prev[l]);
        push(c.left);
    }
    for (int i = 0; i >= 0; i = next[static_cast<size_t>(i)]) out.push_back(id[static_cast<size_t>(i)]);
}

// Leftmost-longest special token at s[i] in the RAW text: user-defined tokens always, control tokens only with parse_special.
const Tokenizer::Special* Tokenizer::find_special(const std::string& s, size_t i, bool parse_special) const {
    if (!(parse_special ? any_first_ : user_first_)[static_cast<uint8_t>(s[i])]) return nullptr;
    for (const Special& sp : specials_)  // longest first
        if ((parse_special || sp.user_defined) && s.compare(i, sp.text.size(), sp.text) == 0) return &sp;
    return nullptr;
}

// A text segment containing no special tokens: NFC, then pre-tokenization and BPE.
void Tokenizer::encode_segment(const std::string& raw, std::vector<int32_t>& out) const {
    if (raw.empty()) return;
    const std::vector<uint32_t> cps = nfc(utf8_to_codepoints(raw));
    for (const auto& span : split_qwen35(cps)) {
        std::string bytes;
        for (size_t k = span.first; k < span.second; ++k) utf8_append(bytes, cps[k]);  // the piece's UTF-8 bytes
        bpe(bytes, out);
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> out;
    size_t seg_start = 0, i = 0;
    while (i < text.size()) {
        if (const Special* hit = find_special(text, i, parse_special)) {
            encode_segment(text.substr(seg_start, i - seg_start), out);  // each gap is normalized on its own
            out.push_back(hit->id);
            i += hit->text.size();
            seg_start = i;
        } else {
            ++i;
        }
    }
    encode_segment(text.substr(seg_start), out);
    return out;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& utf8) const {
    const std::vector<uint32_t> cps = nfc(utf8_to_codepoints(utf8));
    std::vector<std::string> pieces;
    for (const auto& span : split_qwen35(cps)) {
        std::string mapped;
        std::string bytes;
        for (size_t k = span.first; k < span.second; ++k) utf8_append(bytes, cps[k]);
        for (unsigned char b : bytes) utf8_append(mapped, byte_to_cp_[b]);
        pieces.push_back(std::move(mapped));
    }
    return pieces;
}

void Tokenizer::append_token_bytes(int32_t id, bool render_special, std::string& out) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) throw std::out_of_range("token id outside the vocabulary");
    const int type = types_[static_cast<size_t>(id)];
    if (type == 5) return;                                  // unused placeholder: contributes nothing
    if ((type == 3 || type == 4) && !render_special) return;
    const std::string& s = tokens_[static_cast<size_t>(id)];
    // Map every character back to its byte; if any is outside the byte alphabet, use the token's own UTF-8 bytes.
    std::string bytes;
    size_t i = 0;
    bool ok = true;
    while (i < s.size()) {
        uint32_t cp; Utf8Status st;
        const size_t used = utf8_next(reinterpret_cast<const uint8_t*>(s.data()) + i, s.size() - i, cp, st);
        auto it = cp_to_byte_.find(cp);
        if (st != Utf8Status::Ok || it == cp_to_byte_.end()) { ok = false; break; }
        bytes += static_cast<char>(it->second);
        i += used;
    }
    out += ok ? bytes : s;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool render_special) const {
    std::string raw;
    for (int32_t id : ids) append_token_bytes(id, render_special, raw);
    return codepoints_to_utf8(utf8_to_codepoints(raw));  // lossy: one U+FFFD per maximal ill-formed subpart
}

std::string Tokenizer::StreamDecoder::push(int32_t id) {
    t_.append_token_bytes(id, render_special_, pending_);
    std::string out;
    size_t i = 0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pending_.data());
    while (i < pending_.size()) {
        uint32_t cp; Utf8Status st;
        const size_t used = utf8_next(p + i, pending_.size() - i, cp, st);
        if (st == Utf8Status::Truncated) break;  // the rest may still complete: hold it back
        utf8_append(out, st == Utf8Status::Ok ? cp : 0xFFFD);
        i += used;
    }
    pending_.erase(0, i);
    return out;
}

std::string Tokenizer::StreamDecoder::flush() {
    std::string out;
    if (!pending_.empty()) {  // an incomplete sequence at the very end is one replacement character
        out = codepoints_to_utf8(utf8_to_codepoints(pending_));
        pending_.clear();
    }
    return out;
}

}  // namespace dara::tokenizer
