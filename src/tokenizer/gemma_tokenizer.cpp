#include "tokenizer/gemma_tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>

#include "tokenizer/unicode.h"

namespace dara::tokenizer {
namespace {

constexpr const char* kSpace = "\xE2\x96\x81";  // U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's space

uint64_t pair_key(int32_t a, int32_t b) { return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b); }

}  // namespace

GemmaTokenizer GemmaTokenizer::from_gguf(const dara::io::GgufFile& g) {
    auto meta = [&](const std::string& key) -> const std::string* {
        for (const auto& kv : g.metadata) if (kv.first == key) return &kv.second;
        return nullptr;
    };
    const std::string* model = meta("tokenizer.ggml.model");
    if (!model || *model != "gemma4") throw std::runtime_error("unsupported tokenizer model '" + (model ? *model : "<missing>") + "' (supported: gemma4)");

    GemmaTokenizer t;
    auto ta = g.string_arrays.find("tokenizer.ggml.tokens");
    auto ma = g.string_arrays.find("tokenizer.ggml.merges");
    auto ty = g.int_arrays.find("tokenizer.ggml.token_type");
    if (ta == g.string_arrays.end() || ma == g.string_arrays.end() || ty == g.int_arrays.end()) throw std::runtime_error("GGUF lacks tokenizer tokens / merges / token_type");
    t.tokens_ = ta->second;
    if (ty->second.size() != t.tokens_.size()) throw std::runtime_error("token_type length != token count");
    for (int64_t v : ty->second) t.types_.push_back(static_cast<int>(v));
    for (int& b : t.byte_token_) b = -1;

    for (size_t i = 0; i < t.tokens_.size(); ++i) {
        const int ty_i = t.types_[i];
        if (ty_i == 1 || ty_i == 6) {
            if (!t.vocab_.emplace(t.tokens_[i], static_cast<int32_t>(i)).second) throw std::runtime_error("duplicate token text: " + t.tokens_[i]);
        } else if (ty_i == 3 || ty_i == 4) {
            t.specials_.push_back({t.tokens_[i], static_cast<int32_t>(i), ty_i == 4});
        }
    }
    std::stable_sort(t.specials_.begin(), t.specials_.end(), [](const Special& a, const Special& b) { return a.text.size() > b.text.size(); });
    for (const Special& s : t.specials_)
        if (!s.text.empty()) {
            t.any_first_[static_cast<uint8_t>(s.text[0])] = true;
            if (s.user_defined) t.user_first_[static_cast<uint8_t>(s.text[0])] = true;
        }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        auto it = t.vocab_.find(name);
        if (it != t.vocab_.end()) t.byte_token_[b] = it->second;
    }

    t.merges_.reserve(ma->second.size() * 2);
    size_t skipped = 0;
    for (size_t r = 0; r < ma->second.size(); ++r) {
        const std::string& m = ma->second[r];
        const size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) throw std::runtime_error("malformed merge: " + m);
        auto a = t.vocab_.find(m.substr(0, sp)), b = t.vocab_.find(m.substr(sp + 1)), c = t.vocab_.find(m.substr(0, sp) + m.substr(sp + 1));
        if (a == t.vocab_.end() || b == t.vocab_.end() || c == t.vocab_.end()) { ++skipped; continue; }
        t.merges_.emplace(pair_key(a->second, b->second), Merge{static_cast<int32_t>(r), c->second});
    }
    // A merge whose halves or result are missing from the vocabulary can never apply; tolerate (and count) a few, refuse a broken file.
    if (skipped > ma->second.size() / 100) throw std::runtime_error(std::to_string(skipped) + " merges reference tokens missing from the vocabulary");

    for (const auto& kv : g.metadata) {
        if (kv.first == "tokenizer.ggml.eos_token_id") t.eos_ = static_cast<int32_t>(std::stoll(kv.second));
        if (kv.first == "tokenizer.ggml.bos_token_id") t.bos_ = static_cast<int32_t>(std::stoll(kv.second));
    }
    return t;
}

bool GemmaTokenizer::is_special(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= types_.size()) return false;
    const int ty = types_[static_cast<size_t>(id)];
    return ty == 3 || ty == 4;
}

const GemmaTokenizer::Special* GemmaTokenizer::find_special(const std::string& s, size_t i, bool parse_special) const {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    if (!(parse_special ? any_first_[c] : user_first_[c])) return nullptr;
    for (const Special& sp : specials_) {
        if (!parse_special && !sp.user_defined) continue;
        if (sp.text.size() <= s.size() - i && s.compare(i, sp.text.size(), sp.text) == 0) return &sp;
    }
    return nullptr;
}

std::vector<int32_t> GemmaTokenizer::encode(const std::string& utf8, bool parse_special, bool add_bos) const {
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);
    size_t seg = 0, i = 0;
    while (i < utf8.size()) {
        const Special* sp = find_special(utf8, i, parse_special);
        if (!sp) { ++i; continue; }
        if (i > seg) encode_segment(utf8.substr(seg, i - seg), out);
        out.push_back(sp->id);
        i += sp->text.size();
        seg = i;
    }
    if (seg < utf8.size()) encode_segment(utf8.substr(seg), out);
    return out;
}

void GemmaTokenizer::encode_segment(const std::string& raw, std::vector<int32_t>& out) const {
    // Normalizer: every space becomes the SentencePiece space mark. Then one symbol per code point (byte fallback for unknown ones).
    std::vector<int32_t> sym;
    const std::vector<uint32_t> cps = utf8_to_codepoints(raw);
    sym.reserve(cps.size());
    for (uint32_t cp : cps) {
        std::string s;
        if (cp == ' ') s = kSpace; else utf8_append(s, cp);
        auto it = vocab_.find(s);
        if (it != vocab_.end()) { sym.push_back(it->second); continue; }
        for (unsigned char b : s) {
            if (byte_token_[b] < 0) throw std::runtime_error("vocabulary has no byte-fallback token for a byte");
            sym.push_back(byte_token_[b]);
        }
    }
    const int n = static_cast<int>(sym.size());
    if (n == 0) return;
    // BPE with a doubly linked list of symbols and a min-heap of candidate merges (rank, left index); stale entries are skipped.
    std::vector<int> prev(n), next(n);
    std::vector<uint8_t> alive(n, 1);
    for (int k = 0; k < n; ++k) { prev[k] = k - 1; next[k] = k + 1 < n ? k + 1 : -1; }
    struct Cand { int32_t rank; int left; int32_t lid, rid; };
    auto cmp = [](const Cand& a, const Cand& b) { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; };
    std::priority_queue<Cand, std::vector<Cand>, decltype(cmp)> heap(cmp);
    auto push_pair = [&](int left) {
        const int right = next[left];
        if (right < 0) return;
        auto it = merges_.find(pair_key(sym[left], sym[right]));
        if (it != merges_.end()) heap.push({it->second.rank, left, sym[left], sym[right]});
    };
    for (int k = 0; k + 1 < n; ++k) push_pair(k);
    while (!heap.empty()) {
        const Cand c = heap.top();
        heap.pop();
        const int l = c.left;
        if (!alive[l]) continue;
        const int r = next[l];
        if (r < 0 || sym[l] != c.lid || sym[r] != c.rid) continue;  // stale
        sym[l] = merges_.at(pair_key(c.lid, c.rid)).result;
        alive[r] = 0;
        next[l] = next[r];
        if (next[r] >= 0) prev[next[r]] = l;
        if (prev[l] >= 0) push_pair(prev[l]);
        push_pair(l);
    }
    for (int k = 0; k >= 0 && k < n; k = next[k]) out.push_back(sym[k]);
}

void GemmaTokenizer::append_token_bytes(int32_t id, bool render_special, std::string& out) const {
    if (id < 0 || static_cast<size_t>(id) >= tokens_.size()) throw std::out_of_range("token id outside the vocabulary");
    const std::string& s = tokens_[static_cast<size_t>(id)];
    const int ty = types_[static_cast<size_t>(id)];
    if (ty == 6) {  // <0xHH>: one raw byte
        if (s.size() == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            out.push_back(static_cast<char>(std::stoi(s.substr(3, 2), nullptr, 16)));
            return;
        }
    }
    if (ty == 3 || ty == 4 || ty == 2) { if (render_special) out += s; return; }
    for (size_t i = 0; i < s.size();) {  // '▁' -> ' '
        if (s.compare(i, 3, kSpace) == 0) { out.push_back(' '); i += 3; }
        else out.push_back(s[i++]);
    }
}

std::string GemmaTokenizer::decode(const std::vector<int32_t>& ids, bool render_special) const {
    std::string raw;
    for (int32_t id : ids) append_token_bytes(id, render_special, raw);
    return codepoints_to_utf8(utf8_to_codepoints(raw));
}

std::string GemmaTokenizer::StreamDecoder::push(int32_t id) {
    t_.append_token_bytes(id, render_special_, pending_);
    std::string out;
    size_t i = 0;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pending_.data());
    while (i < pending_.size()) {
        uint32_t cp; Utf8Status st;
        const size_t used = utf8_next(p + i, pending_.size() - i, cp, st);
        if (st == Utf8Status::Truncated) break;
        utf8_append(out, st == Utf8Status::Ok ? cp : 0xFFFD);
        i += used;
    }
    pending_.erase(0, i);
    return out;
}

std::string GemmaTokenizer::StreamDecoder::flush() {
    std::string out;
    if (!pending_.empty()) { out = codepoints_to_utf8(utf8_to_codepoints(pending_)); pending_.clear(); }
    return out;
}

}  // namespace dara::tokenizer
