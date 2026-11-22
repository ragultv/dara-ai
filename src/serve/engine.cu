#include "serve/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>

#include <cuda_runtime.h>

#include "io/gguf_checkpoint.h"
#include "io/image.h"
#include "model/flash_drafter.h"
#include "model/gemma_model.h"
#include "model/gemma_vision.h"
#include "model/qwen_model.h"
#include "model/qwen_vision.h"
#include "tokenizer/gemma_chat.h"
#include "tokenizer/gemma_tokenizer.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace dara::serve {

using Clock = std::chrono::steady_clock;
static std::string u8s(const std::filesystem::path& p) { const auto s = p.u8string(); return std::string(s.begin(), s.end()); }
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

std::string Message::text() const {
    std::string s;
    for (const ContentPart& p : parts)
        if (!p.is_image) { if (!s.empty()) s += "\n"; s += p.text; }
    return s;
}

bool Message::has_image() const {
    for (const ContentPart& p : parts) if (p.is_image) return true;
    return false;
}

namespace {

constexpr char kImageSlot = '\x01';   // marks an image's place in rendered template text (removed from user text first)

struct Piece { std::string text; bool special; };

std::string strip_slots(const std::string& s) {
    std::string o;
    o.reserve(s.size());
    for (char c : s) if (c != kImageSlot) o += c;
    return o;
}

bool py_space(uint32_t cp) { return dara::tokenizer::is_space(cp) || (cp >= 0x1C && cp <= 0x1F); }
std::string trim(const std::string& s) {
    const std::vector<uint32_t> cps = dara::tokenizer::utf8_to_codepoints(s);
    size_t b = 0, e = cps.size();
    while (b < e && py_space(cps[b])) ++b;
    while (e > b && py_space(cps[e - 1])) --e;
    return dara::tokenizer::codepoints_to_utf8(std::vector<uint32_t>(cps.begin() + static_cast<long>(b), cps.begin() + static_cast<long>(e)));
}

// Message content for the template: text parts as they are, images as slots (in order).
std::string content_with_slots(const Message& m) {
    std::string s;
    for (const ContentPart& p : m.parts) s += p.is_image ? std::string(1, kImageSlot) : strip_slots(p.text);
    return s;
}

std::string canon_role(const std::string& r) {
    if (r == "developer") return "system";
    return r;
}

uint64_t fnv(uint64_t h, const std::string& s) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    h ^= 0xff; h *= 1099511628211ull;
    return h;
}

uint64_t message_hash(uint64_t h, const Message& m) {
    h = fnv(h, m.role);
    for (const ContentPart& p : m.parts) h = p.is_image ? fnv(fnv(h, "\x01img"), p.image) : fnv(h, p.text);
    return h;
}

std::string exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return u8s(std::filesystem::path(buf).parent_path());
#endif
    return ".";
}

std::string quant_from_name(const std::string& path) {
    std::string up = u8s(std::filesystem::u8path(path).filename());
    for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    static const char* known[] = {"IQ4_XS", "IQ4_NL", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q3_K_M", "Q3_K_S", "Q3_K_L", "Q2_K", "BF16", "F16", "F32"};
    for (const char* k : known) if (up.find(k) != std::string::npos) return k;
    return "unknown";
}

// Holds back text that might be the start of a stop string; reports a stop string or a cancel.
class TextOut {
public:
    TextOut(const std::vector<std::string>& stops, const Engine::TextFn& fn) : fn_(fn) {
        for (const auto& s : stops) if (!s.empty()) stops_.push_back(s);
    }
    // returns false when generation must end (stop string found or the consumer cancelled)
    bool push(const std::string& chunk) {
        if (done_) return false;
        text_ += chunk;
        size_t cut = std::string::npos;
        const size_t from = emitted_ > max_stop_len() ? emitted_ - max_stop_len() : 0;
        for (const auto& s : stops_) {
            const size_t p = text_.find(s, from);
            if (p != std::string::npos && (cut == std::string::npos || p < cut)) cut = p;
        }
        if (cut != std::string::npos) {
            text_.resize(cut);
            stopped_ = true;
            done_ = true;
            emit_to(text_.size());
            return false;
        }
        size_t hold = 0;   // longest suffix that is a proper prefix of a stop string
        for (const auto& s : stops_)
            for (size_t k = std::min(s.size() - 1, text_.size()); k > hold; --k)
                if (text_.compare(text_.size() - k, k, s, 0, k) == 0) { hold = k; break; }
        if (!emit_to(text_.size() - hold)) { cancelled_ = true; done_ = true; return false; }
        return true;
    }
    void finish() { if (!done_) emit_to(text_.size()); done_ = true; }
    const std::string& text() const { return text_; }
    bool stopped() const { return stopped_; }
    bool cancelled() const { return cancelled_; }

private:
    size_t max_stop_len() const { size_t m = 0; for (const auto& s : stops_) m = std::max(m, s.size()); return m; }
    bool emit_to(size_t n) {
        if (n <= emitted_) return true;
        const std::string part = text_.substr(emitted_, n - emitted_);
        emitted_ = n;
        return fn_ ? fn_(part) : true;
    }
    const Engine::TextFn& fn_;
    std::vector<std::string> stops_;
    std::string text_;
    size_t emitted_ = 0;
    bool stopped_ = false, cancelled_ = false, done_ = false;
};

}  // namespace

struct Engine::Impl {
    EngineOptions opt;
    ModelInfo info;
    bool qwen = false;
    std::unique_ptr<dara::io::GgufCheckpoint> gguf;
    // Core / Pro (qwen35)
    dara::model::QwenConfig qcfg;
    std::unique_ptr<dara::model::IModelWeights> qweights;
    std::unique_ptr<dara::model::QwenModel> qm;
    std::unique_ptr<dara::tokenizer::Tokenizer> qtok;
    // Nano (gemma4)
    std::unique_ptr<dara::model::GemmaModel> gm;
    std::unique_ptr<dara::tokenizer::GemmaTokenizer> gtok;
    int hidden = 0, vocab = 0, tok_vocab = 0;
    // image encoder (loaded on the first image; its VRAM is reserved when the context is sized)
    std::unique_ptr<dara::io::GgufCheckpoint> mm;
    std::unique_ptr<dara::model::QwenVision> qvis;
    std::unique_ptr<dara::model::GemmaVision> gvis;
    // Flash drafter
    std::unique_ptr<dara::model::FlashDrafter> flash;
    // prompt cache
    std::vector<int32_t> kv_tokens;      // the tokens whose state the model holds (positions [0, pos))
    bool kv_images = false;              // that state includes image tokens (never reused)
    dara::model::QwenModel::DeviceCheckpoint qckpt;   // Qwen: state after qckpt_tokens.size() - 1 tokens of the last conversation
    std::vector<int32_t> qckpt_tokens;
    size_t flash_valid = 0;              // drafter positions known to match the held state
    bool flash_paused = false;           // Flash stood down in the conversation held now (cleared when a new conversation starts)
    bool spec_paused = false;            // all speculative decoding stood down in it (prompt lookup did not pay off either)
    // special ids
    std::vector<int32_t> eog;
    int32_t img_pad = -1, img_open = -1, img_close = -1;
    // thinking markers: Qwen <think> ... </think> (the generation prompt opens it); Gemma <|channel>thought ... <channel|> (the model opens it)
    int32_t think_close = -1, ch_open = -1, ch_close = -1;
    bool think_on = false;   // the request being rendered asks for thinking
    // stats
    mutable std::mutex smu;
    EngineStats st;
    // context maintainer: summaries of conversation prefixes, keyed by the hash of the summarized messages
    struct Summary { uint64_t key; int count; std::string text; uint64_t used; };
    std::vector<Summary> summaries;
    uint64_t tick = 0;
    // sampler buffers
    std::vector<int32_t> cand;
    std::vector<float> probs;

    // ---------------------------------------------------------------------------------------------------------------------------------
    std::vector<int32_t> encode_plain(const std::string& s) const {
        return qwen ? qtok->encode(s, false) : gtok->encode(s, false, false);
    }
    int32_t special_id(const std::string& s) const {
        const std::vector<int32_t> v = qwen ? qtok->encode(s, true) : gtok->encode(s, true, false);
        if (v.size() != 1) throw std::runtime_error("the vocabulary has no single token for " + s);
        return v[0];
    }
    size_t pos() const { return qwen ? qm->position() : gm->position(); }
    size_t capacity() const { return qwen ? qm->kv_capacity() : gm->kv_capacity(); }

    // ---- chat templates ---------------------------------------------------------------------------------------------------------------
    // Pieces of the rendered prompt; `gen_at` receives the index of the first generation-prompt piece (pieces.size() if none).
    std::vector<Piece> render(const std::vector<Message>& msgs, bool add_gen, size_t* gen_at) const {
        std::vector<Piece> out;
        if (qwen) {
            auto S = [&](const char* s) { out.push_back({s, true}); };
            auto T = [&](std::string s) { out.push_back({std::move(s), false}); };
            size_t first = 0;
            if (!msgs.empty() && canon_role(msgs[0].role) == "system") {
                S("<|im_start|>"); T("system\n" + trim(strip_slots(msgs[0].text()))); S("<|im_end|>"); T("\n");
                first = 1;
            }
            int last_user = -1;
            for (size_t i = first; i < msgs.size(); ++i) if (msgs[i].role == "user") last_user = static_cast<int>(i);
            for (size_t i = first; i < msgs.size(); ++i) {
                const Message& m = msgs[i];
                const std::string role = canon_role(m.role);
                if (role == "user" || role == "system") {
                    S("<|im_start|>"); T(role + "\n" + trim(role == "user" ? content_with_slots(m) : strip_slots(m.text()))); S("<|im_end|>"); T("\n");
                } else if (role == "assistant") {
                    std::string c = strip_slots(m.text());
                    const size_t e = c.rfind("</think>");
                    if (e != std::string::npos) { c = c.substr(e + 8); c.erase(0, c.find_first_not_of('\n') == std::string::npos ? c.size() : c.find_first_not_of('\n')); }
                    c = trim(c);
                    S("<|im_start|>");
                    T(static_cast<int>(i) > last_user ? "assistant\n<think>\n\n</think>\n\n" + c : "assistant\n" + c);
                    S("<|im_end|>"); T("\n");
                } else if (role == "tool") {
                    const bool open = i == first || msgs[i - 1].role != "tool";
                    const bool close = i + 1 == msgs.size() || msgs[i + 1].role != "tool";
                    if (open) { S("<|im_start|>"); T("user"); }
                    T("\n<tool_response>\n" + strip_slots(m.text()) + "\n</tool_response>");
                    if (close) { S("<|im_end|>"); T("\n"); }
                } else {
                    throw std::invalid_argument("unsupported message role '" + m.role + "'");
                }
            }
            *gen_at = out.size();
            if (add_gen) { S("<|im_start|>"); T(think_on ? "assistant\n<think>\n" : "assistant\n<think>\n\n</think>\n\n"); }
        } else {
            std::vector<dara::tokenizer::ChatMessage> cm;
            for (const Message& m : msgs) {
                std::string role = canon_role(m.role);
                if (role == "tool") role = "user";
                if (role != "system" && role != "user" && role != "assistant") throw std::invalid_argument("unsupported message role '" + m.role + "'");
                if (role == "system" && !cm.empty()) role = "user";   // the template supports one leading system turn
                cm.push_back({role, canon_role(m.role) == "user" ? content_with_slots(m) : strip_slots(m.text())});   // images: user messages only
            }
            const auto without = dara::tokenizer::render_gemma_chat_pieces(cm, false, think_on);
            const auto with = add_gen ? dara::tokenizer::render_gemma_chat_pieces(cm, true, think_on) : without;
            for (const auto& p : with) out.push_back({p.text, p.structural});
            *gen_at = without.size();
        }
        return out;
    }

    struct Prompt {
        std::vector<int32_t> ids;
        size_t conv_len = 0;                          // tokens before the generation prompt
        std::vector<std::pair<size_t, int>> images;   // (first image token position, image index)
    };

    // img_tokens[k]: token count of the k-th image slot
    Prompt tokenize(const std::vector<Piece>& pieces, size_t gen_at, const std::vector<int>& img_tokens) const {
        Prompt p;
        std::string run;
        int k = 0;
        auto flush = [&] {
            size_t a = 0;
            while (a <= run.size()) {
                const size_t b = run.find(kImageSlot, a);
                const std::string seg = run.substr(a, b == std::string::npos ? std::string::npos : b - a);
                if (!seg.empty()) { const auto v = encode_plain(seg); p.ids.insert(p.ids.end(), v.begin(), v.end()); }
                if (b == std::string::npos) break;
                if (k >= static_cast<int>(img_tokens.size())) throw std::runtime_error("image slots and images disagree");
                p.ids.push_back(img_open);
                p.images.push_back({p.ids.size(), k});
                p.ids.insert(p.ids.end(), static_cast<size_t>(img_tokens[static_cast<size_t>(k)]), img_pad);
                p.ids.push_back(img_close);
                ++k;
                a = b + 1;
            }
            run.clear();
        };
        for (size_t i = 0; i < pieces.size(); ++i) {
            if (i == gen_at) { flush(); p.conv_len = p.ids.size(); }
            if (!pieces[i].special) { run += pieces[i].text; continue; }
            flush();
            p.ids.push_back(special_id(pieces[i].text));
        }
        flush();
        if (gen_at >= pieces.size()) p.conv_len = p.ids.size();
        return p;
    }

    int image_token_estimate() const { return qwen ? std::max(64, opt.image_max_tokens) : 280; }

    int count(const std::vector<Message>& msgs, bool add_gen = true) const {
        size_t gen_at = 0;
        const auto pieces = render(msgs, add_gen, &gen_at);
        int n_img = 0;
        for (const auto& m : msgs) for (const auto& c : m.parts) if (c.is_image && canon_role(m.role) == "user") ++n_img;
        std::vector<int> est(static_cast<size_t>(n_img), image_token_estimate());
        return static_cast<int>(tokenize(pieces, gen_at, est).ids.size());
    }

    // ---- model steps ------------------------------------------------------------------------------------------------------------------
    void reset_state() {
        if (qwen) qm->reset();
        else { gm->rewind(0); gm->clear_soft_tokens(); }
        kv_tokens.clear();
        kv_images = false;
        held_imgs.clear();
        flash_valid = 0;
    }
    void invalidate() {   // after a failed request: nothing held is trusted
        try { reset_state(); } catch (...) {}
        qckpt.valid = false;
        qckpt_tokens.clear();
    }

    // Feeds tokens [a, b) of `ids` (model position must be a). want_last: produce the next token after position b - 1 (greedy id or logits).
    // With `drafter`, the drafter is extended with every position whose next token is known (all but b - 1 when want_last).
    void feed(const std::vector<int32_t>& ids, size_t a, size_t b, bool want_last, bool greedy, int32_t* next, std::vector<float>* logits,
              bool drafter, std::vector<float>* last_feat) {
        const size_t piece = drafter ? 1024 : (qwen ? b - a : b - a);
        std::vector<float> feats;
        std::vector<int32_t> am;
        dara::model::RunStats rs;
        for (size_t s = a; s < b; s += piece) {
            const size_t e = std::min(b, s + piece);
            const int n = static_cast<int>(e - s);
            const bool last = want_last && e == b;
            std::vector<float> lg;
            if (qwen) {
                if (last && greedy) lg = qm->forward(ids.data() + s, n, n - 1, nullptr, &rs, &am, drafter ? &feats : nullptr, 0);
                else lg = qm->forward(ids.data() + s, n, last ? n - 1 : n, nullptr, &rs, nullptr, drafter ? &feats : nullptr, 0);
            } else {
                lg = gm->forward(ids.data() + s, n, last ? n - 1 : n, nullptr, &rs, drafter ? &feats : nullptr);
            }
            if (last) {
                if (greedy) {
                    if (qwen) *next = am.at(0);
                    else *next = static_cast<int32_t>(std::max_element(lg.begin(), lg.begin() + tok_vocab) - lg.begin());
                } else {
                    *logits = std::move(lg);
                }
            }
            if (drafter) {
                const int known = last ? n - 1 : n;   // positions whose following token is in `ids`
                if (known > 0) {
                    std::vector<float> y; int32_t d = 0;
                    flash->extend(feats.data(), ids.data() + s + 1, known, y, d);
                }
                if (last) last_feat->assign(feats.end() - hidden, feats.end());
            }
        }
    }

    // The sampling distribution of one logits row (n floats, modified in place): repeat penalty, temperature, min_p, top_k, top_p. Fills
    // cand[0, keep) / probs[0, keep) (normalised) and returns keep. Requires temperature > 0.
    size_t dist(float* lg, size_t n, const SampleParams& sp, const std::vector<int32_t>& seq) {
        const int V = std::min(static_cast<int>(n), tok_vocab);
        if (sp.repeat_penalty != 1.0f && sp.repeat_penalty > 0) {
            std::unordered_set<int32_t> seen;
            const size_t m = std::min(seq.size(), static_cast<size_t>(std::max(0, sp.repeat_last_n)));
            for (size_t i = seq.size() - m; i < seq.size(); ++i) {
                const int32_t id = seq[i];
                if (id < 0 || id >= V || !seen.insert(id).second) continue;
                lg[id] = lg[id] > 0 ? lg[id] / sp.repeat_penalty : lg[id] * sp.repeat_penalty;
            }
        }
        const float T = sp.temperature;
        const float mx = *std::max_element(lg, lg + V);
        cand.clear();
        // min_p as a logit threshold first (cheap pruning of the long tail): p_i >= min_p * p_max  <=>  l_i >= l_max + T ln(min_p)
        const float thr = sp.min_p > 0 && sp.min_p < 1 ? mx + T * std::log(sp.min_p) : -INFINITY;
        for (int i = 0; i < V; ++i) if (lg[i] >= thr) cand.push_back(i);
        auto by_logit = [&](int32_t x, int32_t y) { return lg[x] > lg[y]; };
        if (sp.top_k > 0 && static_cast<int>(cand.size()) > sp.top_k) {
            std::nth_element(cand.begin(), cand.begin() + sp.top_k, cand.end(), by_logit);
            cand.resize(static_cast<size_t>(sp.top_k));
        }
        std::sort(cand.begin(), cand.end(), by_logit);
        probs.resize(cand.size());
        double sum = 0;
        for (size_t i = 0; i < cand.size(); ++i) { probs[i] = std::exp((lg[cand[i]] - mx) / T); sum += probs[i]; }
        size_t keep = cand.size();
        if (sp.top_p > 0 && sp.top_p < 1) {
            double c = 0;
            for (size_t i = 0; i < cand.size(); ++i) { c += probs[i] / sum; if (c >= sp.top_p) { keep = i + 1; break; } }
        }
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += probs[i];
        for (size_t i = 0; i < keep; ++i) probs[i] = static_cast<float>(probs[i] / tot);
        return keep;
    }

    // Draws from cand/probs[0, keep), skipping `exclude` (rejection residual of a deterministic draft: p with that token removed).
    int32_t draw(size_t keep, std::mt19937_64& rng, int32_t exclude = -1) {
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) if (cand[i] != exclude) tot += probs[i];
        if (tot <= 0) return cand[0];
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(rng);
        int32_t last = cand[0];
        for (size_t i = 0; i < keep; ++i) {
            if (cand[i] == exclude) continue;
            last = cand[i];
            r -= probs[i];
            if (r <= 0) return cand[i];
        }
        return last;
    }

    int32_t sample(float* lg, size_t n, const SampleParams& sp, const std::vector<int32_t>& seq, std::mt19937_64& rng) {
        if (sp.temperature <= 0 || sp.top_k == 1) {   // greedy (with the repeat penalty applied)
            SampleParams g = sp; g.temperature = 1.0f; g.top_k = 1; g.top_p = 1; g.min_p = 0;
            dist(lg, n, g, seq);   // top_k 1: nth_element keeps only the best, O(vocab)
            return cand[0];
        }
        return draw(dist(lg, n, sp, seq), rng);
    }
    int32_t sample(std::vector<float>& lg, const SampleParams& sp, const std::vector<int32_t>& seq, std::mt19937_64& rng) {
        return sample(lg.data(), lg.size(), sp, seq, rng);
    }

    // ---- one generation over a token prompt ---------------------------------------------------------------------------------------------
    // images: per image, the encoded rows and its merged grid (Qwen) / token count (Gemma)
    // Nano: the encode is deferred (`pending` holds the preprocessed patches) until the prompt cache says which images' positions must be fed at all.
    struct Img { std::vector<float> rows; int gh = 0, gw = 0, n = 0; uint64_t hash = 0; std::shared_ptr<dara::model::VisionInput> pending; };
    struct HeldImg { size_t p0; int n; uint64_t hash; };
    std::vector<HeldImg> held_imgs;          // Nano: the images whose soft tokens the held K/V state was computed from
    double image_ms_acc = 0;                 // encode time spent inside generate() (deferred encodes), added to timings.image_ms
    std::vector<std::pair<uint64_t, std::vector<float>>> img_cache;   // recently encoded images (hash -> soft-token rows), newest last
    static constexpr size_t kImgCacheMax = 8;

    void generate(Prompt& pr, std::vector<Img>& imgs, const SampleParams& sp, const TextFn& on_text, const TextFn& on_reasoning, GenResult& r, Clock::time_point t_start,
                  bool count_stats, bool allow_cache) {
        const size_t cap = capacity();
        const std::vector<int32_t>& ids = pr.ids;
        if (ids.empty()) throw std::invalid_argument("empty prompt");
        if (ids.size() >= cap) throw std::length_error("prompt (" + std::to_string(ids.size()) + " tokens) does not fit the context (" + std::to_string(cap) + ")");
        // thinking: the budget (plus the few tokens that close the thought) comes on top of max_tokens, which then limits the answer alone
        const bool think_mode = think_on && sp.think_budget != 0 && (qwen ? think_close >= 0 : (ch_open >= 0 && ch_close >= 0));
        const size_t think_allow = !think_mode ? 0 : (sp.think_budget > 0 ? static_cast<size_t>(sp.think_budget) + 16 : cap);
        const int max_new = static_cast<int>(std::min<size_t>(static_cast<size_t>(std::max(1, sp.max_tokens)) + think_allow, cap - ids.size()));
        const bool greedy = sp.temperature <= 0 || sp.top_k == 1;
        const bool has_img = !imgs.empty();

        // ---- prompt cache: how much of the held state can be kept ----
        size_t reuse = 0;
        if (!qwen) {   // Nano: image placeholders hold the pad id in the sequence (the per-layer table reads it), so compare in that form
            for (const auto& [p0, k] : pr.images) for (int j = 0; j < imgs[static_cast<size_t>(k)].n; ++j) pr.ids[p0 + static_cast<size_t>(j)] = 0;
        }
        if (allow_cache && qwen && !has_img && !kv_images) {
            if (qckpt.valid && !qckpt_tokens.empty() && qckpt_tokens.size() <= ids.size() && std::equal(qckpt_tokens.begin(), qckpt_tokens.end(), ids.begin()))
                reuse = qckpt_tokens.size() - 1;   // the checkpoint holds all but the last of these tokens
        } else if (allow_cache && !qwen) {
            size_t l = 0;
            while (l < kv_tokens.size() && l < ids.size() && kv_tokens[l] == ids[l]) ++l;
            // An image's K/V is only valid if the same image (same pixels) was held at the same position: cut the match before the first image
            // that differs or is only partly matched. The soft tokens are plain input embeddings (causal attention), so everything before such an
            // image stays exact.
            for (const auto& [p0, k] : pr.images) {
                const Img& im = imgs[static_cast<size_t>(k)];
                if (p0 >= l) break;
                bool same = p0 + static_cast<size_t>(im.n) <= l;
                if (same) {
                    same = false;
                    for (const HeldImg& h : held_imgs) if (h.p0 == p0 && h.n == im.n && h.hash == im.hash) { same = true; break; }
                }
                if (!same) { l = p0; break; }
            }
            reuse = l > 0 ? l - 1 : 0;              // re-feed one token: the drafter and the next-token logits need it
        }
        // speculative decoding (prompt lookup + the Flash drafter), for greedy and sampled requests alike; "flash": false turns it off
        if (reuse == 0) { flash_paused = false; spec_paused = false; }   // a new conversation: speculation gets a fresh chance
        const bool spec_on = opt.spec && sp.flash != 0 && !(spec_paused && sp.flash != 1);
        if (opt.spec && sp.flash == -1 && spec_paused) r.spec.note = "speculative decoding paused for this conversation: drafts were not paying off";
        bool use_flash = spec_on && flash != nullptr;
        if (reuse == 0) flash_paused = false;   // a new conversation: Flash gets a fresh chance
        if (use_flash && flash_paused && sp.flash != 1) { use_flash = false; r.spec.note = "flash paused for this conversation: drafts were not paying off"; }
        if (sp.flash == 1 && !flash) r.spec.note = "flash requested but no drafter is loaded" + (info.flash_note.empty() ? std::string() : ": " + info.flash_note);
        if (use_flash && std::min(flash_valid, flash->position()) < reuse) {
            const size_t gap = reuse - std::min(flash_valid, flash->position());
            // Nano can rewind anywhere: recompute a small part the drafter missed. A large gap means Flash stood down earlier in this conversation
            // (it was not paying off) or a request ran without it: keep the prompt cache and continue without the drafter.
            if (!qwen && gap <= 1024) reuse = std::min(flash_valid, flash->position());
            else { use_flash = false; r.spec.note = "flash paused for this conversation (drafts were not paying off, or a request ran without it); a new conversation tries again"; }
        }
        if (reuse == 0) {
            reset_state();
            // the checkpoint holds only the recurrent states: once the K/V it relied on is overwritten it must never be restored
            if (qwen) { qckpt.valid = false; qckpt_tokens.clear(); }
        }
        else if (qwen) { qm->restore_device(qckpt); qm->clear_images(); }
        else { gm->rewind(reuse); gm->clear_soft_tokens(); }
        kv_tokens.assign(ids.begin(), ids.begin() + static_cast<long>(reuse));
        if (use_flash) flash->rewind(reuse);
        flash_valid = std::min(flash_valid, reuse);

        // images: soft tokens at their placeholder positions. Nano feeds (and so encodes) only the images that reach past the reused prefix.
        for (const auto& [p0, k] : pr.images) {
            Img& im = imgs[static_cast<size_t>(k)];
            if (qwen) qm->add_image(p0, im.gh, im.gw, std::move(im.rows));
            else if (p0 + static_cast<size_t>(im.n) > reuse) {
                ensure_rows(im);
                gm->add_soft_tokens(p0, im.n, std::move(im.rows));
            }
        }
        kv_images = has_img;
        if (!qwen) {
            held_imgs.clear();
            for (const auto& [p0, k] : pr.images) held_imgs.push_back({p0, imgs[static_cast<size_t>(k)].n, imgs[static_cast<size_t>(k)].hash});
        }

        // ---- prefill ----
        const auto tp = Clock::now();
        int32_t first = -1;
        std::vector<float> row, last_feat;
        // (cache_write false: keep the existing checkpoint; this request only writes positions after it, so it stays valid)
        const size_t split = qwen && allow_cache && sp.cache_write && !has_img && pr.conv_len >= 1 && pr.conv_len - 1 > reuse && pr.conv_len < ids.size() ? pr.conv_len - 1 : reuse;
        if (split > reuse) {   // Qwen: checkpoint after all but the last conversation token (the next request resumes there)
            feed(ids, reuse, split, false, greedy, nullptr, nullptr, use_flash, nullptr);
            qm->save_device(qckpt);
            qckpt_tokens.assign(ids.begin(), ids.begin() + static_cast<long>(pr.conv_len));
        } else if (qwen && (has_img || !allow_cache)) {
            qckpt.valid = false;
            qckpt_tokens.clear();
        }
        feed(ids, split, ids.size(), true, greedy, &first, &row, use_flash, &last_feat);
        r.timings.prefill_ms = ms_since(tp);
        r.timings.prefilled = static_cast<int>(ids.size() - reuse);
        r.usage.prompt_tokens = static_cast<int>(ids.size());
        r.usage.cached_tokens = static_cast<int>(reuse);
        std::mt19937_64 rng(sp.seed ? sp.seed : std::random_device{}() * 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(Clock::now().time_since_epoch().count()));
        std::vector<int32_t> seq = ids;   // prompt + every generated token (n-gram lookup, repeat penalty, cache bookkeeping)
        if (!greedy) first = sample(row, sp, seq, rng);

        // ---- decode ----
        dara::tokenizer::Tokenizer::StreamDecoder* qd = nullptr;
        dara::tokenizer::GemmaTokenizer::StreamDecoder* gd = nullptr;
        std::unique_ptr<dara::tokenizer::Tokenizer::StreamDecoder> qdo;
        std::unique_ptr<dara::tokenizer::GemmaTokenizer::StreamDecoder> gdo;
        if (qwen) { qdo = std::make_unique<dara::tokenizer::Tokenizer::StreamDecoder>(*qtok, false); qd = qdo.get(); }
        else { gdo = std::make_unique<dara::tokenizer::GemmaTokenizer::StreamDecoder>(*gtok, false); gd = gdo.get(); }
        TextOut out(sp.stop, on_text);
        int generated = 0;
        bool eog_hit = false;
        Clock::time_point t_first;
        auto is_eog = [&](int32_t id) { return std::find(eog.begin(), eog.end(), id) != eog.end(); };
        // ---- thinking state ----
        bool thinking = think_mode && qwen;            // Qwen: the prompt already opened the thought; Gemma: it starts at the model's <|channel>
        bool think_done = !think_mode;
        bool in_channel = false, budget_hit = false;
        int think_count = 0, think_end_gen = 0;
        double think_end_ms = 0;
        bool label_pending = think_mode && !qwen;      // Gemma writes "thought\n" right after <|channel>: not part of the reasoning text
        std::string label_buf;
        bool rsn_cancelled = false;
        const int answer_cap = std::max(1, sp.max_tokens);
        auto emit_reasoning = [&](const std::string& piece) -> bool {
            std::string p = piece;
            if (label_pending) {
                label_buf += p;
                const size_t nl = label_buf.find('\n');
                if (nl == std::string::npos && label_buf.size() < 16) return true;
                if (label_buf.compare(0, 7, "thought") == 0 && nl != std::string::npos) p = label_buf.substr(nl + 1);
                else p = label_buf;
                label_pending = false;
                label_buf.clear();
            }
            if (p.empty()) return true;
            r.reasoning += p;
            if (on_reasoning && !on_reasoning(p)) { rsn_cancelled = true; return false; }
            return true;
        };
        auto end_thought = [&]() {
            think_done = true; thinking = false; budget_hit = false;
            think_end_gen = generated;
            think_end_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_first).count();
            if (label_pending) { label_pending = false; if (!label_buf.empty()) emit_reasoning(label_buf); label_buf.clear(); }
        };
        auto accept = [&](int32_t id) -> bool {   // one generated token; false: generation ends
            seq.push_back(id);
            ++generated;
            if (generated == 1) { t_first = Clock::now(); r.timings.ttft_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count(); }
            if (count_stats) {
                std::lock_guard<std::mutex> lk(smu);
                st.live_tokens = generated;
                const double dms = std::chrono::duration<double, std::milli>(Clock::now() - t_first).count();
                st.live_tps = generated > 1 && dms > 0 ? (generated - 1) * 1000.0 / dms : 0;
            }
            if (is_eog(id)) { eog_hit = true; return false; }
            if (think_mode && !think_done) {
                if (qwen) {
                    if (id == think_close) { end_thought(); return generated < max_new; }
                } else if (!in_channel) {
                    if (id == ch_open) { in_channel = true; thinking = true; return generated < max_new; }
                    end_thought();   // the model answered without thinking: this token is already the answer
                } else if (id == ch_close) {
                    in_channel = false; end_thought(); return generated < max_new;
                }
                if (thinking) {
                    ++think_count;
                    const std::string piece = qd ? qd->push(id) : gd->push(id);
                    if (!piece.empty() && !emit_reasoning(piece)) return false;
                    // the cap is checked between passes (not inside a verified run), so the state stays aligned with seq
                    if (sp.think_budget > 0 && think_count >= sp.think_budget) budget_hit = true;
                    return generated < max_new;
                }
            }
            const std::string piece = qd ? qd->push(id) : gd->push(id);
            if (!piece.empty() && !out.push(piece)) return false;
            if (think_mode && generated - think_end_gen >= answer_cap) return false;   // the answer has its own max_tokens
            return generated < max_new;
        };
        int passes = 0;
        bool go = accept(first);
        dara::model::RunStats plain_rs;
        auto plain_decode = [&]() {
            while (go && !budget_hit) {
                const int32_t cur = seq.back();
                int32_t nx;
                if (greedy) {
                    if (qwen) { std::vector<int32_t> a1; qm->forward(&cur, 1, 0, nullptr, &plain_rs, &a1); nx = a1.at(0); }
                    else nx = gm->forward_greedy(cur, &plain_rs);
                } else {
                    if (qwen) row = qm->forward(&cur, 1, 0, nullptr, &plain_rs);
                    else gm->forward_greedy(cur, &plain_rs, &row);
                    nx = sample(row, sp, seq, rng);
                }
                ++passes;
                go = accept(nx);
            }
        };
        if (go && spec_on) {
            // speculative decoding: drafts (prompt lookup first, else the Flash drafter) are verified in one forward and the state is rolled back past
            // the first rejected draft. Greedy: a draft is kept while it equals the model's argmax (output identical to plain greedy decoding).
            // Sampled: both drafters are deterministic (q = a point mass on the draft x), so the exact speculative-sampling rule is: keep x with
            // probability p(x) under the request's sampling distribution; on rejection draw from p with x removed. The output then follows exactly
            // the same distribution as plain sampling (only the random stream differs).
            const int kmax = qwen ? std::min(opt.spec_k > 0 ? opt.spec_k : 4, 7) : std::min(opt.spec_k > 0 ? opt.spec_k : 4, 15);
            std::vector<float> y; int32_t d1 = 0; float p1 = 1.0f;
            if (use_flash) { flash->extend(last_feat.data(), &first, 1, y, d1); p1 = flash->last_prob(); }
            r.spec.mode = use_flash ? "flash" : "ngram";
            std::vector<int32_t> draft, feed_t, vids;
            std::vector<float> feats, lgs;
            int ng_cool = 0, ng_back = 0, ng_len = qwen ? 4 : 7;
            // adaptive Flash: when the drafter's tokens keep being rejected (long or unfamiliar context), drafting costs more than it saves --
            // back off (2, 4, 8 ... 32 steps of plain decoding, the drafter still follows along) and try again; any accepted draft resets it
            int fl_cool = 0, fl_back = 0;
            float fl_ema = 1.0f;                  // running average of drafts accepted per verify step (Flash drafts only)
            // overall benefit while the drafter is kept in step: tokens per model pass over a window of passes, counting passes where the drafter
            // declined to draft (low confidence) -- those still pay for keeping it in step
            int fl_win_passes = 0, fl_win_tokens = 0;
            bool ngram_off = false;
            // Core 4B / 9B: measured on a GTX 1650. Nano: its verify pass is cheap and the drafter only drafts when confident (--flash-min-p), so
            // even ~0.8 accepted drafts per step wins (prose +12%, code +35% in the unthrottled driver); the Core thresholds stood it down in the
            // slow first words of an answer and left plain decoding for the whole conversation.
            const int kFlashWindowPasses = qwen ? 48 : 96;
            const float kFlashMinTokensPerPass = qwen ? 1.25f : 1.15f;
            const float kFlashMinAccept = qwen ? 0.7f : 0.5f;   // below this, a verify step of 1 + k tokens costs more than it saves
            const int ngram_max = 4, ngram_min = 3;
            auto lookup = [&](int limit) {
                draft.clear();
                const int n_sz = static_cast<int>(seq.size());
                for (int n = ngram_max; n >= ngram_min && draft.empty(); --n) {
                    if (n_sz <= n) continue;
                    for (int j = n_sz - n - 1; j >= 0; --j) {
                        bool eq = true;
                        for (int q = 0; q < n && eq; ++q) eq = seq[static_cast<size_t>(j + q)] == seq[static_cast<size_t>(n_sz - n + q)];
                        if (!eq) continue;
                        for (int q = j + n; q < n_sz && static_cast<int>(draft.size()) < limit; ++q) draft.push_back(seq[static_cast<size_t>(q)]);
                        if (static_cast<int>(draft.size()) >= std::min(2, limit)) break; draft.clear();   // too close to the end to copy enough: look further back
                    }
                }
            };
            dara::model::RunStats rs;
            while (go && !budget_hit) {
                const int budget = max_new - generated;   // tokens that may still be added after the verify step's own token
                const int32_t cur = seq.back();
                draft.clear();
                bool from_ngram = false;
                if (ngram_off) {}
                else if (ng_cool == 0) { lookup(std::min(ng_len, budget - 1)); from_ngram = draft.size() >= 2; if (!from_ngram) draft.clear(); }
                else --ng_cool;
                const size_t dpos = use_flash ? flash->position() : 0;
                bool from_flash = false;
                if (!from_ngram && use_flash && budget > 1) {
                    if (fl_cool > 0) --fl_cool;
                    else { flash->draft(y, d1, p1, std::min(kmax, budget - 1), opt.flash_min_p, draft); from_flash = !draft.empty(); }
                }
                if (static_cast<int>(draft.size()) > budget - 1) draft.resize(static_cast<size_t>(std::max(0, budget - 1)));
                const int kd = static_cast<int>(draft.size());
                feed_t.assign(1, cur);
                feed_t.insert(feed_t.end(), draft.begin(), draft.end());
                const size_t p0 = pos();
                int m = 0;
                int32_t last_tok = -1;
                if (greedy) {
                    if (qwen) {
                        if (kd > 0) qm->snapshot();
                        qm->forward(feed_t.data(), kd + 1, 0, nullptr, &rs, &vids, use_flash ? &feats : nullptr, 0);
                    } else if (use_flash || kd > 0) {
                        vids = gm->forward_argmax(feed_t.data(), kd + 1, &rs, use_flash ? &feats : nullptr);
                    } else {
                        vids.assign(1, gm->forward_greedy(cur, &rs));
                    }
                    while (m < kd && vids[static_cast<size_t>(m)] == draft[static_cast<size_t>(m)]) ++m;
                    last_tok = vids[static_cast<size_t>(m)];
                } else {
                    if (qwen) {
                        if (kd > 0) qm->snapshot();
                        lgs = qm->forward(feed_t.data(), kd + 1, 0, nullptr, &rs, nullptr, use_flash ? &feats : nullptr, 0);
                    } else if (use_flash || kd > 0) {
                        lgs = gm->forward(feed_t.data(), kd + 1, 0, nullptr, &rs, use_flash ? &feats : nullptr);
                    } else {
                        gm->forward_greedy(cur, &rs, &lgs);
                    }
                    const size_t V = lgs.size() / static_cast<size_t>(kd + 1);
                    const size_t base_len = seq.size();
                    std::uniform_real_distribution<double> u01(0.0, 1.0);
                    for (;; ++m) {
                        const size_t keep = dist(lgs.data() + static_cast<size_t>(m) * V, V, sp, seq);
                        if (m == kd) { last_tok = draw(keep, rng); break; }   // every draft kept: the bonus token
                        const int32_t x = draft[static_cast<size_t>(m)];
                        double px = 0;
                        for (size_t i = 0; i < keep; ++i) if (cand[i] == x) { px = probs[i]; break; }
                        if (u01(rng) < px) { seq.push_back(x); continue; }   // keep x with probability p(x)
                        last_tok = draw(keep, rng, x);                        // rejected: draw from p without x
                        break;
                    }
                    seq.resize(base_len);
                }
                ++passes;
                if (from_flash) {
                    // a single lucky acceptance must not keep an unprofitable drafter running: decide on the running average, back off
                    // exponentially (4 .. 64 plain steps) while it stays low, and give each probe after a back-off a neutral start
                    fl_ema = 0.7f * fl_ema + 0.3f * static_cast<float>(m);
                    if (fl_ema < kFlashMinAccept) {
                        if (fl_back >= 64) {
                            // still not paying after backing off to the limit: stand the drafter down for this conversation. Keeping it in step costs
                            // every token (the target's states for it, no CUDA graph on Nano), so a paused drafter is not free -- a stopped one is.
                            use_flash = false;
                            r.spec.note = "flash paused for this conversation: drafts were not paying off";
                            r.spec.stood_down_at = passes;
                            flash_paused = true;
                            flash_valid = std::min(flash_valid, dpos);
                        } else {
                            fl_back = fl_back ? std::min(fl_back * 2, 64) : 4;
                            fl_cool = fl_back;
                            fl_ema = kFlashMinAccept;
                        }
                    }
                    else fl_back = 0;
                }
                if (from_ngram) {
                    ng_len = m == kd ? std::min(ng_len * 2, 7) : std::max(2, m + 1);
                    if (m == 0) { ng_back = ng_back ? std::min(ng_back * 2, 32) : 2; ng_cool = ng_back; } else ng_back = 0;
                }
                if (kd > 0) { ++r.spec.verify_steps; r.spec.drafted += kd; r.spec.accepted += m; }
                if (m < kd) { if (qwen) qm->rollback(m + 1); else gm->rewind(p0 + static_cast<size_t>(m) + 1); }
                std::vector<int32_t> to_add(draft.begin(), draft.begin() + m);
                to_add.push_back(last_tok);
                if (use_flash) {
                    flash->rewind(dpos);
                    flash->extend(feats.data(), to_add.data(), m + 1, y, d1);
                    p1 = flash->last_prob();
                }
                if (!ngram_off) {   // benefit of speculation as a whole over a window of passes (passes without drafts count too)
                    ++fl_win_passes;
                    fl_win_tokens += static_cast<int>(to_add.size());
                    if (fl_win_passes >= kFlashWindowPasses) {
                        if (static_cast<float>(fl_win_tokens) < kFlashMinTokensPerPass * static_cast<float>(fl_win_passes)) {
                            if (use_flash) {
                                use_flash = false;   // not covering its cost: stand the drafter down for this conversation (it is in step up to here)
                                r.spec.note = "flash paused for this conversation: drafts were not paying off";
                                r.spec.stood_down_at = passes;
                                flash_paused = true;
                            } else {
                                ngram_off = true;    // prompt lookup is not paying off either: plain decoding for the rest of the conversation
                                spec_paused = true;
                                r.spec.note = "speculative decoding paused for this conversation: drafts were not paying off";
                            }
                        }
                        fl_win_passes = fl_win_tokens = 0;
                    }
                }
                for (int32_t id : to_add) if (!(go = accept(id))) break;   // a stop inside the run: seq stays a prefix of the held state
            }
        } else {
            plain_decode();
        }
        if (budget_hit && go && !rsn_cancelled && !out.cancelled()) {
            // The thinking budget ran out: close the thought for the model and let it answer (plain decoding from here on).
            std::vector<int32_t> fseq = seq;   // the state holds all of seq except its last token
            std::vector<int32_t> closing;
            if (qwen) { const auto a = encode_plain("\n"), b = encode_plain("\n\n"); closing.insert(closing.end(), a.begin(), a.end()); closing.push_back(think_close); closing.insert(closing.end(), b.begin(), b.end()); }
            else closing.push_back(ch_close);
            fseq.insert(fseq.end(), closing.begin(), closing.end());
            if (fseq.size() + 1 >= cap) { r.finish_reason = "length"; go = false; }
            else {
                int32_t nx = -1; std::vector<float> lrow;
                feed(fseq, pos(), fseq.size(), true, greedy, &nx, &lrow, false, nullptr);
                seq = fseq;
                r.think_cut = true;
                end_thought();
                if (!greedy) nx = sample(lrow, sp, seq, rng);
                use_flash = false;
                go = accept(nx);
                plain_decode();
            }
        }
        const std::string tail = qd ? qd->flush() : gd->flush();
        if (!tail.empty() && !out.stopped() && !out.cancelled()) out.push(tail);
        out.finish();
        r.timings.decode_ms = generated > 0 ? ms_since(t_first) : 0;
        r.timings.decoded = generated;
        r.text = out.text();
        const bool was_cancelled = out.cancelled() || rsn_cancelled;
        if (r.finish_reason != "length" || was_cancelled || eog_hit || out.stopped())
            r.finish_reason = was_cancelled ? "cancelled" : (eog_hit || out.stopped()) ? "stop" : "length";
        r.usage.completion_tokens = generated;
        r.usage.reasoning_tokens = think_count;
        r.timings.thinking_ms = think_mode ? (think_done ? think_end_ms : r.timings.decode_ms) : 0;
        r.spec.passes = passes;
        // the state now holds every fed token: all of seq except the last generated one (never fed)
        const size_t p = pos();
        kv_tokens.assign(seq.begin(), seq.begin() + static_cast<long>(std::min(p, seq.size())));
        if (use_flash) flash_valid = std::min(flash->position(), p);
        r.context.capacity = cap;
        r.context.used = p;
    }

    // ---- images -------------------------------------------------------------------------------------------------------------------------
    // Runs `fn` (an image encode) with its device allocations carved from the unused tail of the K/V caches (positions the conversation has not
    // reached) first and cudaMalloc second, so the encoder takes no VRAM away from the context. `need` is the encoder's exact allocation
    // (scratch_bytes); the largest single buffers may not fit a tail region and then come from free VRAM.
    template <class Fn>
    auto with_scratch(size_t need, Fn&& fn) {
        dara::model::DeviceArena arena;
        if (qwen) qm->kv_tail_regions(pos(), arena.regions); else gm->kv_tail_regions(pos(), arena.regions);
        const size_t tail = arena.capacity();
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        last_image_need = need;
        try {
            dara::model::ArenaScope scope(arena);
            auto r = fn();
            last_image_borrowed = arena.peak;
            last_image_from_free = arena.fallback;
            return r;
        } catch (const std::exception& e) {
            cudaGetLastError();   // an out-of-memory error from cudaMalloc is not sticky, but clear it
            if (dynamic_cast<const std::length_error*>(&e) || dynamic_cast<const std::invalid_argument*>(&e)) throw;
            throw std::length_error("not enough GPU memory to encode this image right now: it needs " + std::to_string(need >> 20) + " MiB, of which " +
                                    std::to_string(tail >> 20) + " MiB can be borrowed from the unused part of the context and " + std::to_string(free_b >> 20) +
                                    " MiB are free (" + e.what() + "). Shorten the conversation, lower --image-max-tokens, or load the model with a smaller context.");
        }
    }
    size_t last_image_need = 0, last_image_borrowed = 0, last_image_from_free = 0;

    // Nano: the soft-token rows of `im` (from the cache, else encoded now).
    void ensure_rows(Img& im) {
        if (!im.rows.empty() || !im.pending) return;
        for (auto it = img_cache.begin(); it != img_cache.end(); ++it) {
            if (it->first != im.hash) continue;
            im.rows = it->second;
            auto e = std::move(*it); img_cache.erase(it); img_cache.push_back(std::move(e));
            im.pending.reset();
            return;
        }
        const auto t0 = Clock::now();
        im.rows = with_scratch(gvis->scratch_bytes(im.pending->n), [&] { return gvis->encode(*im.pending); });
        image_ms_acc += ms_since(t0);
        im.pending.reset();
        img_cache.emplace_back(im.hash, im.rows);
        if (img_cache.size() > kImgCacheMax) img_cache.erase(img_cache.begin());
    }

    std::vector<Img> encode_images(const std::vector<Message>& msgs, double* ms) {
        std::vector<Img> v;
        const auto t0 = Clock::now();
        for (const Message& m : msgs) {
            if (canon_role(m.role) != "user") continue;   // images are taken from user messages only (as the templates render them)
            for (const ContentPart& c : m.parts) {
                if (!c.is_image) continue;
                if (!qvis && !gvis) throw std::invalid_argument("this server has no image encoder (load the model with vision on: --vision / --mmproj)");
                const auto img = dara::io::load_image_rgb_from_memory(reinterpret_cast<const uint8_t*>(c.image.data()), c.image.size());
                Img im;
                if (qwen) {
                    const auto in = qvis->preprocess(img);
                    im.gh = in.merged_h(); im.gw = in.merged_w(); im.n = im.gh * im.gw;
                    im.rows = with_scratch(qvis->scratch_bytes(in.n), [&] { return qvis->encode(in); });
                } else {
                    im.pending = std::make_shared<dara::model::VisionInput>(gvis->preprocess(img));
                    im.n = im.pending->soft_tokens();
                    im.hash = fnv(1469598103934665603ull, c.image);   // the encode runs later, in generate(), only if the prompt cache cannot cover this image
                }
                v.push_back(std::move(im));
            }
        }
        *ms = v.empty() ? 0 : ms_since(t0);
        image_ms_acc = 0;
        last_image_need = 0;
        return v;
    }

    // ---- context maintainer -------------------------------------------------------------------------------------------------------------
    std::string transcript(const std::vector<Message>& msgs, size_t a, size_t b) const {
        std::string t;
        for (size_t i = a; i < b; ++i) {
            const Message& m = msgs[i];
            std::string body = strip_slots(m.text());
            int imgs = 0;
            for (const auto& c : m.parts) if (c.is_image) ++imgs;
            if (imgs) body = "[" + std::to_string(imgs) + " image" + (imgs > 1 ? "s" : "") + "] " + body;
            t += "[" + canon_role(m.role) + "]: " + trim(body) + "\n\n";
        }
        return t;
    }

    std::string run_summary(const std::string& previous, const std::string& text) {
        std::vector<Message> sm(2);
        sm[0].role = "system";
        sm[0].parts.push_back({false, "You condense conversations. Write a compact summary that preserves names, facts, numbers, decisions, code identifiers, "
                                      "file names, user preferences and open questions. Plain text, no preamble, at most 250 words.", ""});
        sm[1].role = "user";
        std::string u;
        if (!previous.empty()) u += "Summary so far:\n" + previous + "\n\n";
        u += "Conversation to summarize:\n" + text + "\nWrite the updated summary of everything above.";
        sm[1].parts.push_back({false, u, ""});
        const bool saved_think = think_on;
        think_on = false;   // a summary never thinks
        size_t gen_at = 0;
        const auto pieces = render(sm, true, &gen_at);
        Prompt pr = tokenize(pieces, gen_at, {});
        std::vector<Img> none;
        SampleParams sp;
        sp.temperature = 0;
        sp.max_tokens = 384;
        GenResult r;
        try { generate(pr, none, sp, nullptr, nullptr, r, Clock::now(), false, false); } catch (...) { think_on = saved_think; throw; }
        think_on = saved_think;
        return trim(r.text);
    }

    // Summarizes msgs[a, b) after `previous` (a summary of what came before a). Long input goes through in token-bounded chunks.
    std::string summarize(const std::string& previous, const std::vector<Message>& msgs, size_t a, size_t b) {
        const size_t cap = capacity();
        const size_t chunk_tokens = cap > 2048 ? std::min<size_t>(cap - 1536, 24576) : 512;
        std::string summary = previous, chunk;
        size_t chunk_n = 0;
        for (size_t i = a; i < b; ++i) {
            std::string t = transcript(msgs, i, i + 1);
            std::vector<int32_t> tt = encode_plain(t);
            if (tt.size() > chunk_tokens) {   // one huge message: keep its head and tail
                const size_t half = chunk_tokens / 2;
                std::vector<int32_t> h(tt.begin(), tt.begin() + static_cast<long>(half)), e(tt.end() - static_cast<long>(half), tt.end());
                t = (qwen ? qtok->decode(h, false) : gtok->decode(h, false)) + "\n[...]\n" + (qwen ? qtok->decode(e, false) : gtok->decode(e, false));
                tt.resize(chunk_tokens);
            }
            if (chunk_n + tt.size() > chunk_tokens && !chunk.empty()) { summary = run_summary(summary, chunk); chunk.clear(); chunk_n = 0; }
            chunk += t;
            chunk_n += tt.size();
        }
        if (!chunk.empty()) summary = run_summary(summary, chunk);
        return summary;
    }

    std::vector<Message> maintain(const std::vector<Message>& in, int max_tokens, ContextInfo* ci, double* ms) {
        const auto t0 = Clock::now();
        *ms = 0;
        const int cap = static_cast<int>(capacity());
        ci->capacity = static_cast<size_t>(cap);
        const int budget = cap - std::max(1, max_tokens) - opt.compact_reserve;
        if (in.empty()) return in;
        if (count(in) <= budget) return in;
        if (budget < 256) throw std::length_error("max_tokens leaves no room for the prompt in the context (" + std::to_string(cap) + " tokens)");

        const size_t s = canon_role(in[0].role) == "system" ? 1 : 0;   // the system message is always kept
        const size_t n = in.size();
        std::vector<uint64_t> key(n + 1, 1469598103934665603ull);       // key[k]: hash of messages [s, k)
        for (size_t i = s; i < n; ++i) key[i + 1] = message_hash(key[i], in[i]);
        auto assemble = [&](const std::string& summary, size_t k) {
            std::vector<Message> out;
            Message sys;
            sys.role = "system";
            std::string st = s ? strip_slots(in[0].text()) : std::string();
            if (!summary.empty()) st += (st.empty() ? "" : "\n\n") + std::string("Summary of the earlier part of this conversation:\n") + summary;
            if (!st.empty()) { sys.parts.push_back({false, st, ""}); out.push_back(sys); }
            for (size_t i = k; i < n; ++i) out.push_back(in[i]);
            return out;
        };
        // 1. a cached summary of a prefix that already makes everything fit (stable prefix across turns: the prompt cache keeps working)
        Summary* best = nullptr;
        for (auto& e : summaries)
            if (e.count > 0 && s + static_cast<size_t>(e.count) < n && key[s + static_cast<size_t>(e.count)] == e.key && (!best || e.count > best->count)) best = &e;
        if (best) {
            auto out = assemble(best->text, s + static_cast<size_t>(best->count));
            if (count(out) <= budget) {
                best->used = ++tick;
                ci->compacted = true; ci->summarized_messages = best->count; ci->summary_cached = true;
                *ms = ms_since(t0);
                return out;
            }
        }
        // 2. summarize more: keep the most recent messages that fill about half the budget (room for the next turns before compacting again)
        size_t k = n - 1;   // first kept message; the last one is always kept
        int recent = count(assemble("(summary)", k)) + 400;   // + room for a summary of up to 384 tokens
        while (k > s + 1) {
            std::vector<Message> one(in.begin() + static_cast<long>(k - 1), in.begin() + static_cast<long>(k));
            const int add = count(one, false);
            if (recent + add > budget / 2) break;
            recent += add;
            --k;
        }
        std::string prev;
        size_t from = s;
        if (best && s + static_cast<size_t>(best->count) <= k) { prev = best->text; from = s + static_cast<size_t>(best->count); }
        std::string summary = from < k ? summarize(prev, in, from, k) : prev;
        if (from < k) {
            if (summaries.size() >= 16) summaries.erase(std::min_element(summaries.begin(), summaries.end(), [](const Summary& x, const Summary& y) { return x.used < y.used; }));
            summaries.push_back({key[k], static_cast<int>(k - s), summary, ++tick});
        }
        ci->compacted = true;
        ci->summarized_messages = static_cast<int>(k - s);
        std::vector<Message> out = assemble(summary, k);
        // 3. still too long (one huge last message): cut the last message's text from the front
        if (count(out) > budget) {
            Message& last = out.back();
            std::string txt = strip_slots(last.text());
            const std::vector<int32_t> tt = encode_plain(txt);
            const int over = count(out) - budget;
            if (over >= static_cast<int>(tt.size())) throw std::length_error("the last message alone does not fit the context");
            std::vector<int32_t> keep(tt.begin() + over + 16, tt.end());
            const std::string cut = qwen ? qtok->decode(keep, false) : gtok->decode(keep, false);
            std::vector<ContentPart> parts;
            for (const auto& c : last.parts) if (c.is_image) parts.push_back(c);
            parts.push_back({false, "[...] " + cut, ""});
            last.parts = std::move(parts);
            ci->truncated_last = true;
        }
        *ms = ms_since(t0);
        return out;
    }

    void record(const GenResult& r, bool ok) {
        std::lock_guard<std::mutex> lk(smu);
        st.busy = false;
        st.live_tokens = 0;
        st.live_tps = 0;
        if (!ok) { ++st.errors; return; }
        ++st.requests;
        st.prompt_tokens += static_cast<uint64_t>(r.usage.prompt_tokens);
        st.cached_tokens += static_cast<uint64_t>(r.usage.cached_tokens);
        st.completion_tokens += static_cast<uint64_t>(r.usage.completion_tokens);
        if (r.context.compacted) ++st.compactions;
        st.prefill_ms += r.timings.prefill_ms;
        st.decode_ms += r.timings.decode_ms;
        st.prefilled += r.timings.prefilled;
        st.decoded += std::max(0, r.timings.decoded - 1);
        st.last = r;
        st.has_last = true;
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// ---- residency options shared by loading a model and by the dry run (ModelPlanner) ------------------------------------------------------------------------
namespace {

struct Residency {
    dara::model::ResidencyOptions ropt, ropt_plain;   // ropt: with the drafter's VRAM; ropt_plain: without it (the "auto" fallback)
    std::string flash_path, flash_note;
    size_t mmproj_bytes = 0;
};

dara::model::KvFormat parse_kv_format(const std::string& f) {
    using dara::model::KvFormat;
    if (f != "q8" && f != "int8" && f != "k8v4" && f != "q4" && f != "tq4" && f != "k16v8" && f != "f32") throw std::invalid_argument("unknown K/V format '" + f + "' (q8, k8v4, q4, tq4, k16v8)");
    return f == "k8v4" ? KvFormat::K8V4 : f == "q4" ? KvFormat::Q4 : f == "tq4" ? KvFormat::TQ4 : f == "k16v8" ? KvFormat::K16V8 : f == "f32" ? KvFormat::F32 : KvFormat::Q8;
}

Residency make_residency(const EngineOptions& o, const dara::io::GgufCheckpoint& gguf, bool qwen, const dara::model::QwenConfig* qcfg, bool verbose) {
    Residency r;
    dara::model::ResidencyOptions& ropt = r.ropt;
    ropt.kv_format = parse_kv_format(o.kv_format);
    ropt.auto_min_context = o.min_context;
    ropt.safety_bytes = o.vram_reserve_mb << 20;
    ropt.assume_free_bytes = o.assume_free_bytes;
    if (o.progress) ropt.progress = [cb = o.progress](const char* stage, int done, int total) { cb(stage, done, total); };
    std::error_code ec;
    if (!o.mmproj.empty()) {   // the image encoder's weights and activations get VRAM when the context is sized
        const auto sz = std::filesystem::file_size(std::filesystem::u8path(o.mmproj), ec);
        if (ec) throw std::runtime_error("cannot read --mmproj " + o.mmproj);
        r.mmproj_bytes = static_cast<size_t>(sz);
        // the encoder streams its weights from the file and borrows its working memory from the unused context while an image is encoded: between
        // images it keeps only its resident parts (patch embedding, biases, norms: a few MB)
        ropt.extra_vram_bytes += 24u << 20;
    }
    // Flash drafter: an explicit file, or the bundled drafter for this model when found next to dara.exe
    std::string flash_path = o.flash;
    if (o.flash == "auto") {
        flash_path.clear();
        std::string want;
        if (!qwen) want = "drafter_nano_v3.dflash";   // Nano: v3 (69% held-out next-token accuracy; v2 is no longer used)
        else if (qcfg && qcfg->hidden == 2560) want = "drafter_4b_v2.dflash";   // Core 4B (no drafter is shipped for Pro 9B)
        if (!want.empty()) {
            const std::string dirs[] = {exe_dir() + "/flash", exe_dir(), "build/flash", u8s(std::filesystem::u8path(o.gguf).parent_path())};
            for (const auto& d : dirs) {
                const auto p = std::filesystem::u8path(d) / want;
                if (std::filesystem::exists(p, ec)) { flash_path = u8s(p); break; }
            }
        }
    }
    if (!o.spec || o.flash_mode == "off") { flash_path.clear(); r.flash_note = "turned off when the server started"; }
    else if (flash_path.empty()) r.flash_note = qwen && qcfg && qcfg->hidden != 2560 ? "no drafter exists for this model" : "drafter file not installed";
    r.ropt_plain = ropt;   // without the drafter's VRAM (the "auto" fallback)
    if (!flash_path.empty()) {
        try {
            size_t fixed = 0, per = 0;
            dara::model::FlashDrafter::vram_estimate(flash_path, gguf, fixed, per);
            ropt.extra_vram_bytes += fixed;
            ropt.extra_bytes_per_token += per;
        } catch (const std::exception& e) {
            if (verbose) std::fprintf(stderr, "note: Flash drafter %s not used (%s)\n", flash_path.c_str(), e.what());
            r.flash_note = std::string("drafter unusable: ") + e.what();
            flash_path.clear();
        }
    }
    if (qwen && qcfg) {   // the prompt-cache checkpoint (a copy of the recurrent states); the speculative snapshot fits in the safety reserve, as in dara.exe
        ropt.extra_vram_bytes += dara::model::QwenModel::device_checkpoint_bytes(*qcfg);
        r.ropt_plain.extra_vram_bytes += dara::model::QwenModel::device_checkpoint_bytes(*qcfg);
    }
    r.flash_path = flash_path;
    return r;
}

}  // namespace

Engine::Engine(const EngineOptions& o) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.opt = o;
    m.gguf = std::make_unique<dara::io::GgufCheckpoint>(o.gguf);
    std::string arch;
    for (const auto& kv : m.gguf->header().metadata) if (kv.first == "general.architecture") { arch = kv.second; break; }
    if (arch != "qwen35" && arch != "gemma4") throw std::runtime_error("unsupported GGUF architecture '" + arch + "' (Dara runs qwen35 and gemma4)");
    m.qwen = arch == "qwen35";
    std::error_code ec;
    if (m.qwen) m.qcfg = dara::model::QwenConfig::from_gguf(*m.gguf);
    if (o.progress) o.progress("planning", 0, 1);
    Residency rs = make_residency(o, *m.gguf, m.qwen, m.qwen ? &m.qcfg : nullptr, true);
    rs.ropt_plain.progress = rs.ropt.progress;
    dara::model::ResidencyOptions& ropt = rs.ropt;
    dara::model::ResidencyOptions& ropt_plain = rs.ropt_plain;
    std::string flash_path = rs.flash_path;
    m.info.mmproj_bytes = rs.mmproj_bytes;
    m.info.flash_note = rs.flash_note;
    if (m.qwen) {
        m.qtok = std::make_unique<dara::tokenizer::Tokenizer>(dara::tokenizer::Tokenizer::from_gguf(m.gguf->header()));
        m.qweights = std::make_unique<dara::model::GgufWeights>(*m.gguf, m.qcfg);
        m.gguf->release_string_arrays();
        m.qm = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, ropt);
        // --flash auto: the drafter only pays off when it does not push layers out of VRAM (measured: 4B on 4 GB, 30/32 layers with it is slower
        // than 32/32 without); reload without it in that case
        if (!flash_path.empty() && o.flash_mode == "auto" && m.qm->placement().resident_layers < m.qcfg.layers) {
            m.qm.reset();
            m.qm = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, ropt_plain);
            if (m.qm->placement().resident_layers < m.qcfg.layers) std::fprintf(stderr, "note: the model does not fit in VRAM; Flash is off\n");
            else std::fprintf(stderr, "note: Flash is off (its VRAM would move layers out of VRAM, which costs more than it gains)\n");
            m.info.flash_note = "its VRAM would move model layers out of VRAM on this GPU (slower than without it); load with flash on to force it";
            flash_path.clear();
        }
        m.hidden = m.qcfg.hidden;
        m.vocab = m.qcfg.vocab;
        m.tok_vocab = static_cast<int>(std::min<size_t>(m.qtok->vocab_size(), static_cast<size_t>(m.vocab)));
        m.eog.push_back(m.qtok->eos_id());
        for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
            try { const int32_t id = m.special_id(s); if (std::find(m.eog.begin(), m.eog.end(), id) == m.eog.end()) m.eog.push_back(id); } catch (...) {}
        }
        m.img_open = m.special_id("<|vision_start|>");
        m.img_pad = m.special_id("<|image_pad|>");
        m.img_close = m.special_id("<|vision_end|>");
        try { m.think_close = m.special_id("</think>"); } catch (...) {}
        const auto& pr = m.qm->placement();
        m.info.family = "qwen35";
        m.info.layers = m.qcfg.layers;
        m.info.layers_gpu = pr.resident_layers;
        m.info.head_on_cpu = pr.head_bytes == 0;
        m.info.native_context = m.qcfg.context_length;
        m.info.load_seconds = m.qm->load_seconds();
        m.info.weights_vram_bytes = pr.resident_weight_bytes + pr.head_bytes;
        m.info.gpu = pr.gpu;
        m.info.placement = pr.plan;
        if (o.progress && !flash_path.empty()) o.progress("loading the Flash drafter", 0, 1);
        if (!flash_path.empty()) m.flash = std::make_unique<dara::model::FlashDrafter>(flash_path, *m.gguf, [w = m.qweights.get()](int32_t t) { return w->embed_row(t); }, m.qm->kv_capacity() + 16);
    } else {
        m.gtok = std::make_unique<dara::tokenizer::GemmaTokenizer>(dara::tokenizer::GemmaTokenizer::from_gguf(m.gguf->header()));
        m.gguf->release_string_arrays();
        m.gm = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, ropt);
        if (!flash_path.empty() && o.flash_mode == "auto" && m.gm->placement().resident_layers < m.gm->config().layers) {   // as for Qwen above
            m.gm.reset();
            m.gm = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, ropt_plain);
            std::fprintf(stderr, "note: Flash is off (its VRAM would move layers out of VRAM)\n");
            m.info.flash_note = "its VRAM would move model layers out of VRAM on this GPU (slower than without it); load with flash on to force it";
            flash_path.clear();
        }
        const auto& cfg = m.gm->config();
        m.hidden = cfg.hidden;
        m.vocab = cfg.vocab;
        m.tok_vocab = static_cast<int>(std::min<size_t>(m.gtok->vocab_size(), static_cast<size_t>(m.vocab)));
        m.eog.push_back(m.gtok->eos_id());
        if (m.gtok->eos_id() != 1) m.eog.push_back(1);   // <eos>
        m.img_open = m.special_id("<|image>");
        m.img_pad = m.special_id("<|image|>");
        m.img_close = m.special_id("<image|>");
        try { m.ch_open = m.special_id("<|channel>"); m.ch_close = m.special_id("<channel|>"); } catch (...) {}
        const auto& pr = m.gm->placement();
        m.info.family = "gemma4";
        m.info.layers = cfg.layers;
        m.info.layers_gpu = pr.resident_layers;
        m.info.head_on_cpu = false;
        m.info.native_context = cfg.context_length;
        m.info.load_seconds = m.gm->load_seconds();
        m.info.weights_vram_bytes = pr.resident_weight_bytes + pr.head_bytes;
        m.info.gpu = pr.gpu;
        m.info.placement = pr.plan;
        if (o.progress && !flash_path.empty()) o.progress("loading the Flash drafter", 0, 1);
        if (!flash_path.empty()) m.flash = std::make_unique<dara::model::FlashDrafter>(flash_path, *m.gguf, [g = m.gm.get()](int32_t t) { return g->embed_row_raw(t); }, m.gm->kv_capacity() + 16);
    }
    if (o.progress && !o.mmproj.empty()) o.progress("loading the image encoder", 0, 1);
    if (!o.mmproj.empty()) {   // after the model: the encoder's resident parts come out of the reservation above
        m.mm = std::make_unique<dara::io::GgufCheckpoint>(o.mmproj);
        if (m.qwen) {
            m.qvis = std::make_unique<dara::model::QwenVision>(*m.mm, o.image_max_tokens);
            if (m.qvis->out_dim() != m.hidden) throw std::runtime_error("the mmproj does not match this model (projection width differs)");
            m.info.image_scratch_bytes = m.qvis->scratch_bytes(4 * std::max(64, o.image_max_tokens));
            m.info.vision_resident_bytes = m.qvis->resident_bytes();
            m.info.vision_streamed = m.qvis->streaming();
        } else {
            m.gvis = std::make_unique<dara::model::GemmaVision>(*m.mm);
            m.info.image_scratch_bytes = m.gvis->scratch_bytes(280 * 9);
            m.info.vision_resident_bytes = m.gvis->resident_bytes();
            m.info.vision_streamed = m.gvis->streaming();
        }
    }
    m.info.path = o.gguf;
    m.info.id = o.id.empty() ? u8s(std::filesystem::u8path(o.gguf).stem()) : o.id;
    m.info.file_bytes = static_cast<size_t>(std::filesystem::file_size(std::filesystem::u8path(o.gguf), ec));
    m.info.mmproj = o.mmproj;
    m.info.quant = quant_from_name(o.gguf);
    m.info.hidden = m.hidden;
    m.info.vocab = m.vocab;
    m.info.context_capacity = m.capacity();
    m.info.kv_format = dara::model::to_string(ropt.kv_format);
    m.info.vision = !o.mmproj.empty();
    m.info.image_max_tokens = m.qwen ? o.image_max_tokens : 280;
    m.info.flash = m.flash ? flash_path : std::string();
    m.info.spec_k = o.spec ? o.spec_k : 0;
}

Engine::~Engine() = default;

const ModelInfo& Engine::info() const { return impl_->info; }

EngineStats Engine::stats() const {
    std::lock_guard<std::mutex> lk(impl_->smu);
    return impl_->st;
}

int Engine::count_tokens(const std::vector<Message>& messages) { return impl_->count(messages); }

std::vector<Message> Engine::compact(const std::vector<Message>& messages, int max_tokens, ContextInfo* info, double* ms) {
    ContextInfo ci;
    double t = 0;
    auto out = impl_->maintain(messages, max_tokens, &ci, &t);
    if (info) *info = ci;
    if (ms) *ms = t;
    return out;
}

GenResult Engine::chat(const std::vector<Message>& messages, const SampleParams& sp, const TextFn& on_text, const TextFn& on_reasoning) {
    Impl& m = *impl_;
    m.think_on = sp.think_budget != 0;
    const auto t0 = Clock::now();
    { std::lock_guard<std::mutex> lk(m.smu); m.st.busy = true; m.st.live_tokens = 0; m.st.live_tps = 0; }
    GenResult r;
    try {
        if (messages.empty()) throw std::invalid_argument("no messages");
        ContextInfo ci;
        double cms = 0;
        const std::vector<Message> msgs = m.maintain(messages, sp.max_tokens + (sp.think_budget == 0 ? 0 : static_cast<int>(std::min<size_t>(sp.think_budget > 0 ? static_cast<size_t>(sp.think_budget) : m.capacity() / 4, m.capacity() / 4))), &ci, &cms);
        double ims = 0;
        std::vector<Impl::Img> imgs = m.encode_images(msgs, &ims);
        std::vector<int> counts;
        for (const auto& im : imgs) counts.push_back(im.n);
        size_t gen_at = 0;
        const auto pieces = m.render(msgs, true, &gen_at);
        Impl::Prompt pr = m.tokenize(pieces, gen_at, counts);
        m.generate(pr, imgs, sp, on_text, on_reasoning, r, t0, true, true);
        r.context.compacted = ci.compacted;
        r.context.summarized_messages = ci.summarized_messages;
        r.context.dropped_messages = ci.dropped_messages;
        r.context.truncated_last = ci.truncated_last;
        r.context.summary_cached = ci.summary_cached;
        r.timings.compact_ms = cms;
        ims += m.image_ms_acc;   // Nano: encodes that ran inside generate()
        r.timings.image_ms = ims;
        if (ims > 0 && m.last_image_need > 0) {
            r.timings.image_scratch_mb = m.last_image_need / 1048576.0;
            r.timings.image_borrowed_mb = m.last_image_borrowed / 1048576.0;
            r.timings.image_free_vram_mb = m.last_image_from_free / 1048576.0;
        }
        for (const auto& im : imgs) r.usage.image_tokens += im.n;
        r.timings.total_ms = ms_since(t0);
    } catch (...) {
        m.invalidate();
        m.record(r, false);
        throw;
    }
    m.record(r, true);
    return r;
}

GenResult Engine::complete(const std::string& prompt, const SampleParams& sp, const TextFn& on_text) {
    Impl& m = *impl_;
    const auto t0 = Clock::now();
    { std::lock_guard<std::mutex> lk(m.smu); m.st.busy = true; m.st.live_tokens = 0; m.st.live_tps = 0; }
    GenResult r;
    try {
        Impl::Prompt pr;
        pr.ids = m.qwen ? m.qtok->encode(prompt, true) : m.gtok->encode(prompt, true, true);
        const size_t cap = m.capacity();
        const size_t room = cap > static_cast<size_t>(std::max(1, sp.max_tokens)) + 1 ? cap - static_cast<size_t>(std::max(1, sp.max_tokens)) - 1 : 1;
        if (pr.ids.size() > room) {   // keep the end of an over-long raw prompt
            pr.ids.erase(pr.ids.begin(), pr.ids.end() - static_cast<long>(room));
            r.context.truncated_last = true;
        }
        pr.conv_len = pr.ids.size();
        std::vector<Impl::Img> none;
        m.generate(pr, none, sp, on_text, nullptr, r, t0, true, true);
        r.timings.total_ms = ms_since(t0);
    } catch (...) {
        m.invalidate();
        m.record(r, false);
        throw;
    }
    m.record(r, true);
    return r;
}

// ---- ModelPlanner ------------------------------------------------------------------------------------------------------------------------------------

struct ModelPlanner::Impl {
    std::unique_ptr<dara::io::GgufCheckpoint> gguf;
    bool qwen = false;
    dara::model::QwenConfig qcfg;
    std::unique_ptr<dara::model::GgufWeights> qweights;
    std::string arch;
};

ModelPlanner::ModelPlanner(const std::string& path) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.gguf = std::make_unique<dara::io::GgufCheckpoint>(path);
    for (const auto& kv : m.gguf->header().metadata) if (kv.first == "general.architecture") { m.arch = kv.second; break; }
    if (m.arch != "qwen35" && m.arch != "gemma4") throw std::runtime_error("unsupported GGUF architecture '" + m.arch + "' (Dara runs qwen35 and gemma4)");
    m.qwen = m.arch == "qwen35";
    if (m.qwen) {
        m.qcfg = dara::model::QwenConfig::from_gguf(*m.gguf);
        m.qweights = std::make_unique<dara::model::GgufWeights>(*m.gguf, m.qcfg);
    }
}

ModelPlanner::~ModelPlanner() = default;
const std::string& ModelPlanner::family() const { return impl_->arch; }
int ModelPlanner::layers() const { return impl_->qwen ? impl_->qcfg.layers : 0; }

PlanResult ModelPlanner::plan(const EngineOptions& o) {
    Impl& m = *impl_;
    PlanResult res;
    try {
        Residency rs = make_residency(o, *m.gguf, m.qwen, m.qwen ? &m.qcfg : nullptr, false);
        rs.ropt.plan_only = rs.ropt_plain.plan_only = true;
        res.flash_active = !rs.flash_path.empty();
        res.flash_note = rs.flash_note;
        const char* drop = "its VRAM would move model layers out of VRAM on this GPU (slower than without it); load with flash on to force it";
        auto fill = [&](const dara::model::PlacementReport& pr, size_t context, size_t native, int layers) {
            res.ok = true;
            res.context = context;
            res.native_context = native;
            res.layers = layers;
            res.layers_gpu = pr.resident_layers;
            res.head_on_cpu = pr.head_on_cpu;
            res.placement = pr.plan;
            res.vram_total = pr.vram_total;
            res.vram_free_assumed = pr.vram_free_at_start;
            res.state_bytes = pr.state_bytes;
        };
        if (m.qwen) {
            auto q = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, rs.ropt);
            if (res.flash_active && o.flash_mode == "auto" && q->placement().resident_layers < m.qcfg.layers) {   // as Engine does: the drafter is dropped
                q = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, rs.ropt_plain);
                res.flash_active = false;
                res.flash_note = drop;
            }
            fill(q->placement(), q->kv_capacity(), m.qcfg.context_length, m.qcfg.layers);
        } else {
            auto g = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, rs.ropt);
            if (res.flash_active && o.flash_mode == "auto" && g->placement().resident_layers < g->config().layers) {
                g = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, rs.ropt_plain);
                res.flash_active = false;
                res.flash_note = drop;
            }
            fill(g->placement(), g->kv_capacity(), g->config().context_length, g->config().layers);
        }
    } catch (const std::exception& e) {
        res.ok = false;
        res.error = e.what();
    }
    return res;
}
}  // namespace dara::serve
