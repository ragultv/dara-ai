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
    // special ids
    std::vector<int32_t> eog;
    int32_t img_pad = -1, img_open = -1, img_close = -1;
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
            if (add_gen) { S("<|im_start|>"); T("assistant\n<think>\n\n</think>\n\n"); }
        } else {
            std::vector<dara::tokenizer::ChatMessage> cm;
            for (const Message& m : msgs) {
                std::string role = canon_role(m.role);
                if (role == "tool") role = "user";
                if (role != "system" && role != "user" && role != "assistant") throw std::invalid_argument("unsupported message role '" + m.role + "'");
                if (role == "system" && !cm.empty()) role = "user";   // the template supports one leading system turn
                cm.push_back({role, canon_role(m.role) == "user" ? content_with_slots(m) : strip_slots(m.text())});   // images: user messages only
            }
            const auto without = dara::tokenizer::render_gemma_chat_pieces(cm, false, false);
            const auto with = add_gen ? dara::tokenizer::render_gemma_chat_pieces(cm, true, false) : without;
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

    int32_t sample(std::vector<float>& lg, const SampleParams& sp, const std::vector<int32_t>& seq, std::mt19937_64& rng) {
        const int V = std::min(static_cast<int>(lg.size()), tok_vocab);
        if (sp.repeat_penalty != 1.0f && sp.repeat_penalty > 0) {
            std::unordered_set<int32_t> seen;
            const size_t n = std::min(seq.size(), static_cast<size_t>(std::max(0, sp.repeat_last_n)));
            for (size_t i = seq.size() - n; i < seq.size(); ++i) {
                const int32_t id = seq[i];
                if (id < 0 || id >= V || !seen.insert(id).second) continue;
                lg[static_cast<size_t>(id)] = lg[static_cast<size_t>(id)] > 0 ? lg[static_cast<size_t>(id)] / sp.repeat_penalty : lg[static_cast<size_t>(id)] * sp.repeat_penalty;
            }
        }
        if (sp.temperature <= 0 || sp.top_k == 1) return static_cast<int32_t>(std::max_element(lg.begin(), lg.begin() + V) - lg.begin());
        const float T = sp.temperature;
        const float mx = *std::max_element(lg.begin(), lg.begin() + V);
        cand.clear();
        // min_p as a logit threshold first (cheap pruning of the long tail): p_i >= min_p * p_max  <=>  l_i >= l_max + T ln(min_p)
        const float thr = sp.min_p > 0 && sp.min_p < 1 ? mx + T * std::log(sp.min_p) : -INFINITY;
        for (int i = 0; i < V; ++i) if (lg[static_cast<size_t>(i)] >= thr) cand.push_back(i);
        auto by_logit = [&](int32_t x, int32_t y) { return lg[static_cast<size_t>(x)] > lg[static_cast<size_t>(y)]; };
        if (sp.top_k > 0 && static_cast<int>(cand.size()) > sp.top_k) {
            std::nth_element(cand.begin(), cand.begin() + sp.top_k, cand.end(), by_logit);
            cand.resize(static_cast<size_t>(sp.top_k));
        }
        std::sort(cand.begin(), cand.end(), by_logit);
        probs.resize(cand.size());
        double sum = 0;
        for (size_t i = 0; i < cand.size(); ++i) { probs[i] = std::exp((lg[static_cast<size_t>(cand[i])] - mx) / T); sum += probs[i]; }
        size_t keep = cand.size();
        if (sp.top_p > 0 && sp.top_p < 1) {
            double c = 0;
            for (size_t i = 0; i < cand.size(); ++i) { c += probs[i] / sum; if (c >= sp.top_p) { keep = i + 1; break; } }
        }
        double tot = 0;
        for (size_t i = 0; i < keep; ++i) tot += probs[i];
        std::uniform_real_distribution<double> u(0.0, tot);
        double r = u(rng);
        for (size_t i = 0; i < keep; ++i) { r -= probs[i]; if (r <= 0) return cand[i]; }
        return cand[keep - 1];
    }

    // ---- one generation over a token prompt ---------------------------------------------------------------------------------------------
    // images: per image, the encoded rows and its merged grid (Qwen) / token count (Gemma)
    struct Img { std::vector<float> rows; int gh = 0, gw = 0, n = 0; };

    void generate(Prompt& pr, std::vector<Img>& imgs, const SampleParams& sp, const TextFn& on_text, GenResult& r, Clock::time_point t_start,
                  bool count_stats, bool allow_cache) {
        const size_t cap = capacity();
        const std::vector<int32_t>& ids = pr.ids;
        if (ids.empty()) throw std::invalid_argument("empty prompt");
        if (ids.size() >= cap) throw std::length_error("prompt (" + std::to_string(ids.size()) + " tokens) does not fit the context (" + std::to_string(cap) + ")");
        const int max_new = static_cast<int>(std::min<size_t>(static_cast<size_t>(std::max(1, sp.max_tokens)), cap - ids.size()));
        const bool greedy = sp.temperature <= 0 || sp.top_k == 1;
        const bool has_img = !imgs.empty();

        // ---- prompt cache: how much of the held state can be kept ----
        size_t reuse = 0;
        if (allow_cache && !has_img && !kv_images) {
            if (qwen) {
                if (qckpt.valid && !qckpt_tokens.empty() && qckpt_tokens.size() <= ids.size() && std::equal(qckpt_tokens.begin(), qckpt_tokens.end(), ids.begin()))
                    reuse = qckpt_tokens.size() - 1;   // the checkpoint holds all but the last of these tokens
            } else {
                size_t l = 0;
                while (l < kv_tokens.size() && l < ids.size() && kv_tokens[l] == ids[l]) ++l;
                reuse = l > 0 ? l - 1 : 0;              // re-feed one token: the drafter and the next-token logits need it
            }
        }
        bool use_flash = greedy && opt.spec && flash != nullptr;
        if (use_flash && std::min(flash_valid, flash->position()) < reuse) {
            if (!qwen) reuse = std::min(flash_valid, flash->position());   // Nano can rewind anywhere: recompute the part the drafter missed
            else use_flash = false;                 // the drafter fell behind (sampled requests do not feed it)
        }
        if (reuse == 0) reset_state();
        else if (qwen) { qm->restore_device(qckpt); qm->clear_images(); }
        else { gm->rewind(reuse); gm->clear_soft_tokens(); }
        kv_tokens.assign(ids.begin(), ids.begin() + static_cast<long>(reuse));
        if (use_flash) flash->rewind(reuse);
        flash_valid = std::min(flash_valid, reuse);

        // images: soft tokens at their placeholder positions
        for (const auto& [p0, k] : pr.images) {
            Img& im = imgs[static_cast<size_t>(k)];
            if (qwen) qm->add_image(p0, im.gh, im.gw, std::move(im.rows));
            else {
                for (int j = 0; j < im.n; ++j) pr.ids[p0 + static_cast<size_t>(j)] = 0;   // <pad>: the per-layer table reads it
                gm->add_soft_tokens(p0, im.n, std::move(im.rows));
            }
        }
        kv_images = has_img;

        // ---- prefill ----
        const auto tp = Clock::now();
        int32_t first = -1;
        std::vector<float> row, last_feat;
        const size_t split = qwen && allow_cache && !has_img && pr.conv_len >= 1 && pr.conv_len - 1 > reuse && pr.conv_len < ids.size() ? pr.conv_len - 1 : reuse;
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
            const std::string piece = qd ? qd->push(id) : gd->push(id);
            if (!piece.empty() && !out.push(piece)) return false;
            return generated < max_new;
        };
        int passes = 0;
        bool go = accept(first);
        if (go && greedy && opt.spec) {
            // speculative decoding, greedy-exact: drafts (prompt lookup first, else the Flash drafter) are verified in one forward; the longest prefix
            // the model itself would have produced is kept and the state rolled back past the rest
            const int kmax = qwen ? std::min(opt.spec_k > 0 ? opt.spec_k : 4, 7) : std::min(opt.spec_k > 0 ? opt.spec_k : 4, 15);
            std::vector<float> y; int32_t d1 = 0; float p1 = 1.0f;
            if (use_flash) { flash->extend(last_feat.data(), &first, 1, y, d1); p1 = flash->last_prob(); }
            r.spec.mode = use_flash ? "flash" : "ngram";
            std::vector<int32_t> draft, feed_t, vids;
            std::vector<float> feats;
            int ng_cool = 0, ng_back = 0, ng_len = qwen ? 4 : 7;
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
                        if (!draft.empty()) break;
                    }
                }
            };
            dara::model::RunStats rs;
            while (go) {
                const int budget = max_new - generated;   // tokens that may still be added after the verify step's own token
                const int32_t cur = seq.back();
                draft.clear();
                bool from_ngram = false;
                if (ng_cool == 0) { lookup(std::min(ng_len, budget - 1)); from_ngram = draft.size() >= 2; if (!from_ngram) draft.clear(); }
                else --ng_cool;
                const size_t dpos = use_flash ? flash->position() : 0;
                if (!from_ngram && use_flash && budget > 1) flash->draft(y, d1, p1, std::min(kmax, budget - 1), opt.flash_min_p, draft);
                if (static_cast<int>(draft.size()) > budget - 1) draft.resize(static_cast<size_t>(std::max(0, budget - 1)));
                const int kd = static_cast<int>(draft.size());
                feed_t.assign(1, cur);
                feed_t.insert(feed_t.end(), draft.begin(), draft.end());
                const size_t p0 = pos();
                if (qwen) {
                    if (kd > 0) qm->snapshot();
                    qm->forward(feed_t.data(), kd + 1, 0, nullptr, &rs, &vids, use_flash ? &feats : nullptr, 0);
                } else if (use_flash || kd > 0) {
                    vids = gm->forward_argmax(feed_t.data(), kd + 1, &rs, use_flash ? &feats : nullptr);
                } else {
                    vids.assign(1, gm->forward_greedy(cur, &rs));
                }
                ++passes;
                int m = 0;
                while (m < kd && vids[static_cast<size_t>(m)] == draft[static_cast<size_t>(m)]) ++m;
                if (from_ngram) {
                    ng_len = m == kd ? std::min(ng_len * 2, 7) : std::max(2, m + 1);
                    if (m == 0) { ng_back = ng_back ? std::min(ng_back * 2, 32) : 2; ng_cool = ng_back; } else ng_back = 0;
                }
                if (kd > 0) { ++r.spec.verify_steps; r.spec.drafted += kd; r.spec.accepted += m; }
                if (m < kd) { if (qwen) qm->rollback(m + 1); else gm->rewind(p0 + static_cast<size_t>(m) + 1); }
                std::vector<int32_t> to_add(draft.begin(), draft.begin() + m);
                to_add.push_back(vids[static_cast<size_t>(m)]);
                if (use_flash) {
                    flash->rewind(dpos);
                    flash->extend(feats.data(), to_add.data(), m + 1, y, d1);
                    p1 = flash->last_prob();
                }
                for (int32_t id : to_add) if (!(go = accept(id))) break;   // a stop inside the run: seq stays a prefix of the held state
            }
        } else {
            dara::model::RunStats rs;
            while (go) {
                const int32_t cur = seq.back();
                int32_t nx;
                if (greedy) {
                    if (qwen) { std::vector<int32_t> a1; qm->forward(&cur, 1, 0, nullptr, &rs, &a1); nx = a1.at(0); }
                    else nx = gm->forward_greedy(cur, &rs);
                } else {
                    if (qwen) row = qm->forward(&cur, 1, 0, nullptr, &rs);
                    else gm->forward_greedy(cur, &rs, &row);
                    nx = sample(row, sp, seq, rng);
                }
                ++passes;
                go = accept(nx);
            }
        }
        const std::string tail = qd ? qd->flush() : gd->flush();
        if (!tail.empty() && !out.stopped() && !out.cancelled()) out.push(tail);
        out.finish();
        r.timings.decode_ms = generated > 0 ? ms_since(t_first) : 0;
        r.timings.decoded = generated;
        r.text = out.text();
        r.finish_reason = out.cancelled() ? "cancelled" : (eog_hit || out.stopped()) ? "stop" : "length";
        r.usage.completion_tokens = generated;
        r.spec.passes = passes;
        // the state now holds every fed token: all of seq except the last generated one (never fed)
        const size_t p = pos();
        kv_tokens.assign(seq.begin(), seq.begin() + static_cast<long>(std::min(p, seq.size())));
        if (use_flash) flash_valid = std::min(flash->position(), p);
        r.context.capacity = cap;
        r.context.used = p;
    }

    // ---- images -------------------------------------------------------------------------------------------------------------------------
    std::vector<Img> encode_images(const std::vector<Message>& msgs, double* ms) {
        std::vector<Img> v;
        const auto t0 = Clock::now();
        for (const Message& m : msgs) {
            if (canon_role(m.role) != "user") continue;   // images are taken from user messages only (as the templates render them)
            for (const ContentPart& c : m.parts) {
                if (!c.is_image) continue;
                if (!mm) {
                    if (opt.mmproj.empty()) throw std::invalid_argument("this server has no image encoder (start it with --mmproj <vision GGUF>)");
                    mm = std::make_unique<dara::io::GgufCheckpoint>(opt.mmproj);
                    if (qwen) {
                        qvis = std::make_unique<dara::model::QwenVision>(*mm, opt.image_max_tokens);
                        if (qvis->out_dim() != hidden) throw std::runtime_error("the mmproj does not match this model (projection width differs)");
                    } else {
                        gvis = std::make_unique<dara::model::GemmaVision>(*mm);
                    }
                }
                const auto img = dara::io::load_image_rgb_from_memory(reinterpret_cast<const uint8_t*>(c.image.data()), c.image.size());
                Img im;
                if (qwen) {
                    const auto in = qvis->preprocess(img);
                    im.gh = in.merged_h(); im.gw = in.merged_w(); im.n = im.gh * im.gw;
                    im.rows = qvis->encode(in);
                } else {
                    const auto in = gvis->preprocess(img);
                    im.n = in.soft_tokens();
                    im.rows = gvis->encode(in);
                }
                v.push_back(std::move(im));
            }
        }
        *ms = v.empty() ? 0 : ms_since(t0);
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
        size_t gen_at = 0;
        const auto pieces = render(sm, true, &gen_at);
        Prompt pr = tokenize(pieces, gen_at, {});
        std::vector<Img> none;
        SampleParams sp;
        sp.temperature = 0;
        sp.max_tokens = 384;
        GenResult r;
        generate(pr, none, sp, nullptr, r, Clock::now(), false, false);
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
Engine::Engine(const EngineOptions& o) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    m.opt = o;
    m.gguf = std::make_unique<dara::io::GgufCheckpoint>(o.gguf);
    std::string arch;
    for (const auto& kv : m.gguf->header().metadata) if (kv.first == "general.architecture") { arch = kv.second; break; }
    if (arch != "qwen35" && arch != "gemma4") throw std::runtime_error("unsupported GGUF architecture '" + arch + "' (Dara runs qwen35 and gemma4)");
    m.qwen = arch == "qwen35";

    dara::model::ResidencyOptions ropt;
    ropt.kv_format = dara::model::KvFormat::Q8;
    ropt.auto_min_context = o.min_context;
    std::error_code ec;
    if (!o.mmproj.empty()) {   // the image encoder's weights and activations get VRAM when the context is sized
        const auto sz = std::filesystem::file_size(std::filesystem::u8path(o.mmproj), ec);
        if (ec) throw std::runtime_error("cannot read --mmproj " + o.mmproj);
        m.info.mmproj_bytes = static_cast<size_t>(sz);
        ropt.extra_vram_bytes += static_cast<size_t>(sz) + (256u << 20);
    }
    // Flash drafter: an explicit file, or the bundled drafter for this model when found next to dara.exe
    std::string flash_path = o.flash;
    if (o.flash == "auto") {
        flash_path.clear();
        std::string want;
        if (!m.qwen) want = "drafter_nano_v2.dflash";
        else {
            const auto qc = dara::model::QwenConfig::from_gguf(*m.gguf);
            if (qc.hidden == 2560) want = "drafter_4b_v2.dflash";   // Core 4B (no drafter is shipped for Pro 9B)
        }
        if (!want.empty()) {
            const std::string dirs[] = {exe_dir() + "/flash", exe_dir(), "build/flash", u8s(std::filesystem::u8path(o.gguf).parent_path())};
            for (const auto& d : dirs) {
                const auto p = std::filesystem::u8path(d) / want;
                if (std::filesystem::exists(p, ec)) { flash_path = u8s(p); break; }
            }
        }
    }
    if (!o.spec) flash_path.clear();
    dara::model::ResidencyOptions ropt_plain = ropt;   // without the drafter's VRAM (the "auto" fallback)
    if (!flash_path.empty()) {
        try {
            size_t fixed = 0, per = 0;
            dara::model::FlashDrafter::vram_estimate(flash_path, *m.gguf, fixed, per);
            ropt.extra_vram_bytes += fixed;
            ropt.extra_bytes_per_token += per;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "note: Flash drafter %s not used (%s)\n", flash_path.c_str(), e.what());
            flash_path.clear();
        }
    }

    if (m.qwen) {
        m.qcfg = dara::model::QwenConfig::from_gguf(*m.gguf);
        m.qtok = std::make_unique<dara::tokenizer::Tokenizer>(dara::tokenizer::Tokenizer::from_gguf(m.gguf->header()));
        m.qweights = std::make_unique<dara::model::GgufWeights>(*m.gguf, m.qcfg);
        // the prompt-cache checkpoint (a copy of the recurrent states); the speculative snapshot fits in the safety reserve, as in dara.exe
        ropt.extra_vram_bytes += dara::model::QwenModel::device_checkpoint_bytes(m.qcfg);
        ropt_plain.extra_vram_bytes += dara::model::QwenModel::device_checkpoint_bytes(m.qcfg);
        m.gguf->release_string_arrays();
        m.qm = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, ropt);
        // --flash auto: the drafter only pays off when it does not push layers out of VRAM (measured: 4B on 4 GB, 30/32 layers with it is slower
        // than 32/32 without); reload without it in that case
        if (!flash_path.empty() && o.flash == "auto" && m.qm->placement().resident_layers < m.qcfg.layers) {
            m.qm.reset();
            m.qm = std::make_unique<dara::model::QwenModel>(m.qcfg, *m.qweights, o.kv, ropt_plain);
            if (m.qm->placement().resident_layers < m.qcfg.layers) std::fprintf(stderr, "note: the model does not fit in VRAM; Flash is off\n");
            else std::fprintf(stderr, "note: Flash is off (its VRAM would move layers out of VRAM, which costs more than it gains)\n");
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
        if (!flash_path.empty()) m.flash = std::make_unique<dara::model::FlashDrafter>(flash_path, *m.gguf, [w = m.qweights.get()](int32_t t) { return w->embed_row(t); }, m.qm->kv_capacity() + 16);
    } else {
        m.gtok = std::make_unique<dara::tokenizer::GemmaTokenizer>(dara::tokenizer::GemmaTokenizer::from_gguf(m.gguf->header()));
        m.gguf->release_string_arrays();
        m.gm = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, ropt);
        if (!flash_path.empty() && o.flash == "auto" && m.gm->placement().resident_layers < m.gm->config().layers) {   // as for Qwen above
            m.gm.reset();
            m.gm = std::make_unique<dara::model::GemmaModel>(*m.gguf, o.kv, ropt_plain);
            std::fprintf(stderr, "note: Flash is off (its VRAM would move layers out of VRAM)\n");
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
        if (!flash_path.empty()) m.flash = std::make_unique<dara::model::FlashDrafter>(flash_path, *m.gguf, [g = m.gm.get()](int32_t t) { return g->embed_row_raw(t); }, m.gm->kv_capacity() + 16);
    }
    m.info.path = o.gguf;
    m.info.id = u8s(std::filesystem::u8path(o.gguf).stem());
    m.info.file_bytes = static_cast<size_t>(std::filesystem::file_size(std::filesystem::u8path(o.gguf), ec));
    m.info.mmproj = o.mmproj;
    m.info.quant = quant_from_name(o.gguf);
    m.info.hidden = m.hidden;
    m.info.vocab = m.vocab;
    m.info.context_capacity = m.capacity();
    m.info.kv_format = "q8";
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

GenResult Engine::chat(const std::vector<Message>& messages, const SampleParams& sp, const TextFn& on_text) {
    Impl& m = *impl_;
    const auto t0 = Clock::now();
    { std::lock_guard<std::mutex> lk(m.smu); m.st.busy = true; m.st.live_tokens = 0; m.st.live_tps = 0; }
    GenResult r;
    try {
        if (messages.empty()) throw std::invalid_argument("no messages");
        ContextInfo ci;
        double cms = 0;
        const std::vector<Message> msgs = m.maintain(messages, sp.max_tokens, &ci, &cms);
        double ims = 0;
        std::vector<Impl::Img> imgs = m.encode_images(msgs, &ims);
        std::vector<int> counts;
        for (const auto& im : imgs) counts.push_back(im.n);
        size_t gen_at = 0;
        const auto pieces = m.render(msgs, true, &gen_at);
        Impl::Prompt pr = m.tokenize(pieces, gen_at, counts);
        m.generate(pr, imgs, sp, on_text, r, t0, true, true);
        r.context.compacted = ci.compacted;
        r.context.summarized_messages = ci.summarized_messages;
        r.context.dropped_messages = ci.dropped_messages;
        r.context.truncated_last = ci.truncated_last;
        r.context.summary_cached = ci.summary_cached;
        r.timings.compact_ms = cms;
        r.timings.image_ms = ims;
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
        m.generate(pr, none, sp, on_text, r, t0, true, true);
        r.timings.total_ms = ms_since(t0);
    } catch (...) {
        m.invalidate();
        m.record(r, false);
        throw;
    }
    m.record(r, true);
    return r;
}

}  // namespace dara::serve
