// `dara serve`: an HTTP server around one loaded model (OpenAI-compatible chat / completions API) plus Dara's own development endpoints.
//
//   GET  /health                 liveness + model id
//   GET  /v1/models              the loaded model
//   POST /v1/chat/completions    OpenAI chat (stream or not); text + images (data: URIs / base64); usage + timings in every response
//   POST /v1/completions         raw prompt completion
//   POST /dara/tokenize          {"text"} or {"messages"} -> token count (and ids for text)
//   POST /dara/compact           {"messages", "max_tokens"} -> the messages as the context maintainer would send them
//   GET  /dara/stats             memory (process / RAM / VRAM / disk), GPU telemetry, model, context, throughput, last request
//   GET  /metrics                the same as Prometheus text
// Requests run one at a time on the engine (a queue); stats are readable while a request runs.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

#include "serve/catalog.h"
#include "serve/download.h"
#include "serve/engine.h"
#include "serve/metrics.h"

#ifdef _WIN32
#include <process.h>
#endif

using json = nlohmann::json;
using namespace dara::serve;

namespace {

using Clock = std::chrono::steady_clock;
std::string u8s(const std::filesystem::path& p) { const auto s = p.u8string(); return std::string(s.begin(), s.end()); }

std::string b64_decode(const std::string& in) {
    static int T[256];
    static bool init = false;
    if (!init) {
        for (int& t : T) t = -1;
        const char* a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; ++i) T[static_cast<unsigned char>(a[i])] = i;
        T['-'] = 62; T['_'] = 63;   // URL-safe alphabet too
        init = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=' ) break;
        if (T[c] < 0) { if (std::isspace(c)) continue; throw std::invalid_argument("invalid base64 image data"); }
        val = (val << 6) | T[c];
        bits += 6;
        if (bits >= 0) { out.push_back(static_cast<char>((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}

std::string image_bytes(const std::string& url_or_b64) {
    if (url_or_b64.rfind("data:", 0) == 0) {
        const size_t c = url_or_b64.find(',');
        if (c == std::string::npos || url_or_b64.substr(0, c).find(";base64") == std::string::npos) throw std::invalid_argument("image data URI must be base64");
        return b64_decode(url_or_b64.substr(c + 1));
    }
    if (url_or_b64.rfind("http://", 0) == 0 || url_or_b64.rfind("https://", 0) == 0)
        throw std::invalid_argument("remote image URLs are not fetched; send the image as a data: URI (base64)");
    return b64_decode(url_or_b64);   // bare base64 (Ollama style)
}

std::vector<Message> parse_messages(const json& arr) {
    if (!arr.is_array() || arr.empty()) throw std::invalid_argument("'messages' must be a non-empty array");
    std::vector<Message> out;
    for (const json& m : arr) {
        Message msg;
        msg.role = m.value("role", "");
        if (msg.role.empty()) throw std::invalid_argument("every message needs a role");
        if (m.contains("content")) {
            const json& c = m["content"];
            if (c.is_string()) msg.parts.push_back({false, c.get<std::string>(), ""});
            else if (c.is_array()) {
                for (const json& p : c) {
                    const std::string type = p.value("type", "text");
                    if (type == "text") msg.parts.push_back({false, p.value("text", ""), ""});
                    else if (type == "image_url") {
                        const json& iu = p["image_url"];
                        const std::string url = iu.is_string() ? iu.get<std::string>() : iu.value("url", "");
                        msg.parts.push_back({true, "", image_bytes(url)});
                    } else if (type == "image") {
                        msg.parts.push_back({true, "", image_bytes(p.value("image", p.value("data", "")))});
                    } else throw std::invalid_argument("unsupported content part type '" + type + "'");
                }
            } else if (!c.is_null()) throw std::invalid_argument("message content must be a string or an array of parts");
        }
        if (m.contains("images") && m["images"].is_array())   // Ollama style: base64 images next to the text
            for (const json& im : m["images"]) msg.parts.insert(msg.parts.begin(), ContentPart{true, "", image_bytes(im.get<std::string>())});
        out.push_back(std::move(msg));
    }
    return out;
}

SampleParams parse_sampling(const json& b, int default_max) {
    SampleParams sp;
    sp.max_tokens = b.value("max_completion_tokens", b.value("max_tokens", b.value("n_predict", default_max)));
    if (sp.max_tokens <= 0) sp.max_tokens = default_max;
    sp.temperature = b.value("temperature", sp.temperature);
    sp.top_k = b.value("top_k", sp.top_k);
    sp.top_p = b.value("top_p", sp.top_p);
    sp.min_p = b.value("min_p", sp.min_p);
    sp.repeat_penalty = b.value("repeat_penalty", sp.repeat_penalty);
    sp.repeat_last_n = b.value("repeat_last_n", sp.repeat_last_n);
    if (b.contains("seed") && b["seed"].is_number_integer()) sp.seed = b["seed"].get<uint64_t>();
    if (b.contains("stop")) {
        const json& s = b["stop"];
        if (s.is_string()) sp.stop.push_back(s.get<std::string>());
        else if (s.is_array()) for (const json& x : s) if (x.is_string()) sp.stop.push_back(x.get<std::string>());
    }
    return sp;
}

double r2(double v) { return std::round(v * 100.0) / 100.0; }

json usage_json(const GenResult& r) {
    return {{"prompt_tokens", r.usage.prompt_tokens},
            {"completion_tokens", r.usage.completion_tokens},
            {"total_tokens", r.usage.total_tokens()},
            {"prompt_tokens_details", {{"cached_tokens", r.usage.cached_tokens}, {"image_tokens", r.usage.image_tokens}}}};
}

json timings_json(const GenResult& r) {
    const Timings& t = r.timings;
    return {{"queue_ms", r2(t.queue_ms)},
            {"compact_ms", r2(t.compact_ms)},
            {"image_ms", r2(t.image_ms)},
            {"prefill_tokens", t.prefilled},
            {"prefill_ms", r2(t.prefill_ms)},
            {"prefill_tokens_per_second", r2(t.prefill_tps())},
            {"ttft_ms", r2(t.ttft_ms)},
            {"decode_tokens", t.decoded},
            {"decode_ms", r2(t.decode_ms)},
            {"decode_tokens_per_second", r2(t.decode_tps())},
            {"total_ms", r2(t.total_ms)}};
}

json context_json(const ContextInfo& c) {
    return {{"capacity", c.capacity}, {"used", c.used}, {"compacted", c.compacted}, {"summarized_messages", c.summarized_messages},
            {"dropped_messages", c.dropped_messages}, {"truncated_last_message", c.truncated_last}, {"summary_cached", c.summary_cached}};
}

json spec_json(const SpecInfo& s, int decoded) {
    return {{"mode", s.mode.empty() ? json(nullptr) : json(s.mode)}, {"verify_steps", s.verify_steps}, {"drafted", s.drafted}, {"accepted", s.accepted},
            {"acceptance", s.drafted ? r2(100.0 * s.accepted / s.drafted) : 0.0}, {"model_passes", s.passes},
            {"tokens_per_pass", s.passes ? r2(static_cast<double>(decoded) / s.passes) : 0.0}};
}

void add_dara_fields(json& j, const GenResult& r) {
    j["usage"] = usage_json(r);
    j["timings"] = timings_json(r);
    j["context"] = context_json(r.context);
    j["speculative"] = spec_json(r.spec, r.timings.decoded);
}

std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

json error_json(const std::string& msg, const char* type) { return {{"error", {{"message", msg}, {"type", type}}}}; }

double mb(uint64_t b) { return r2(b / 1048576.0); }

std::string find_mmproj(const std::string& model) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::vector<fs::path> found;
    for (const auto& e : fs::directory_iterator(fs::u8path(model).parent_path(), ec)) {
        std::string n = u8s(e.path().filename());
        for (char& c : n) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (e.is_regular_file() && n.find("mmproj") != std::string::npos && e.path().extension() == ".gguf") found.push_back(e.path());
    }
    return found.size() == 1 ? u8s(found[0]) : std::string();
}

}  // namespace

// ---- model registry: the loaded engine, catalog resolution, loading / switching -----------------------------------------------------------
namespace {

struct ModelNotFound : std::runtime_error { using std::runtime_error::runtime_error; };

struct LoadSpec {
    std::string model;            // catalog id (or short name) or a .gguf path
    std::string flash = "auto";   // auto | on | off | <drafter path>
    int vision = -1;              // -1: the server default, 0 off, 1 on
    std::string mmproj;           // explicit mmproj path (path models)
    long long ctx = -1;           // -1: the server default
};

struct Registry {
    std::string models_dir;
    EngineOptions base;
    bool default_vision = false;
    std::mutex engine_mu;         // serialises generation and loading
    std::mutex ptr_mu;
    std::shared_ptr<Engine> engine;
    std::string loaded_id;        // catalog id, or "" for a model loaded by path
    std::string loaded_flash_mode = "auto";

    std::shared_ptr<Engine> get() { std::lock_guard<std::mutex> lk(ptr_mu); return engine; }

    // engine_mu held
    void unload() {
        std::shared_ptr<Engine> old;
        { std::lock_guard<std::mutex> lk(ptr_mu); old = std::move(engine); loaded_id.clear(); }
        while (old && old.use_count() > 1) std::this_thread::sleep_for(std::chrono::milliseconds(5));   // a stats reader still holds it
        old.reset();   // frees the VRAM before anything else loads
    }

    // engine_mu held. Throws ModelNotFound for a catalog model that is not installed.
    void load(const LoadSpec& s) {
        EngineOptions eo = base;
        std::string id;
        const bool want_vision = s.vision < 0 ? default_vision : s.vision == 1;
        if (const CatalogEntry* e = catalog_find(s.model)) {
            id = e->id;
            eo.gguf = installed_path(models_dir, *e, *e->file("model"));
            if (eo.gguf.empty()) throw ModelNotFound("model '" + e->id + "' is not installed");
            eo.id = e->id;
            if (want_vision) {
                if (const CatalogFile* mf = e->file("mmproj")) eo.mmproj = installed_path(models_dir, *e, *mf);
                if (eo.mmproj.empty()) std::fprintf(stderr, "note: %s has no installed image encoder; image input is off\n", e->id.c_str());
            }
            const CatalogFile* ff = e->file("flash");
            eo.flash = ff ? installed_path(models_dir, *e, *ff) : std::string();
        } else {
            std::error_code ec;
            if (!std::filesystem::exists(std::filesystem::u8path(s.model), ec)) throw ModelNotFound("unknown model '" + s.model + "' (not a Dara model id and no such file)");
            eo.gguf = s.model;
            eo.flash = "auto";
            eo.mmproj = !s.mmproj.empty() ? s.mmproj : (want_vision ? find_mmproj(s.model) : std::string());
        }
        if (s.flash == "off" || s.flash == "false") eo.flash_mode = "off";
        else if (s.flash == "on" || s.flash == "true") eo.flash_mode = "on";
        else if (s.flash == "auto" || s.flash.empty()) eo.flash_mode = "auto";
        else { eo.flash = s.flash; eo.flash_mode = "on"; }   // an explicit drafter file
        if (s.ctx >= 0) eo.kv = static_cast<size_t>(s.ctx);
        unload();
        std::fprintf(stderr, "loading %s (%s) ...\n", id.empty() ? eo.gguf.c_str() : id.c_str(), eo.gguf.c_str());
        auto e = std::make_shared<Engine>(eo);
        const ModelInfo& mi = e->info();
        std::fprintf(stderr, "model %s (%s, %s) | %s\n  context %zu tokens | layers on GPU %d/%d%s | load %.1f s | flash %s | vision %s\n", mi.id.c_str(),
                     mi.family.c_str(), mi.quant.c_str(), mi.gpu.c_str(), mi.context_capacity, mi.layers_gpu, mi.layers, mi.head_on_cpu ? ", output head in RAM" : "",
                     mi.load_seconds, mi.flash.empty() ? ("off (" + mi.flash_note + ")").c_str() : mi.flash.c_str(), mi.vision ? mi.mmproj.c_str() : "off");
        std::lock_guard<std::mutex> lk(ptr_mu);
        engine = std::move(e);
        loaded_id = id;
        loaded_flash_mode = eo.flash_mode;
    }

    // engine_mu held: the engine a request for `model` runs on (loads / switches for an installed catalog model)
    std::shared_ptr<Engine> ensure(const std::string& model) {
        const CatalogEntry* e = model.empty() ? nullptr : catalog_find(model);
        if (e && e->id != loaded_id) {   // (no engine reference may be held here: unload() waits for the others to go)
            LoadSpec s;
            s.model = e->id;
            s.flash = loaded_flash_mode;
            load(s);
            return get();
        }
        auto cur = get();
        if (!cur) throw std::invalid_argument("no model is loaded: name one of " + ids() + " in \"model\" (or POST /dara/models/load)");
        return cur;   // the loaded model (also for names that are not Dara ids, as OpenAI clients send their own defaults)
    }

    static std::string ids() {
        std::string s;
        for (const auto& e : catalog()) s += (s.empty() ? "" : ", ") + e.id;
        return s;
    }
};

struct Downloads {
    std::mutex mu;
    std::map<std::string, std::shared_ptr<DownloadJob>> jobs;   // by job id
    uint64_t next = 0;
    std::shared_ptr<DownloadJob> running_for(const std::string& model) {
        for (auto& [k, j] : jobs) if (j->model() == model && !j->finished()) return j;
        return nullptr;
    }
};

json progress_json(const DownloadJob& j) {
    const DownloadProgress p = j.progress();
    const double rate = p.seconds > 0 ? p.bytes_done / p.seconds : 0;
    json o = {{"id", j.id()}, {"model", j.model()}, {"status", p.state}, {"file", p.current_file}, {"files_done", p.files_done}, {"files_total", p.files_total},
              {"completed_bytes", p.bytes_done}, {"total_bytes", p.bytes_total},
              {"percent", p.bytes_total ? r2(100.0 * p.bytes_done / p.bytes_total) : 100.0}, {"mb_per_second", r2(rate / 1048576.0)},
              {"eta_seconds", rate > 0 && p.bytes_total > p.bytes_done ? r2((p.bytes_total - p.bytes_done) / rate) : 0.0}, {"seconds", r2(p.seconds)}};
    if (!p.error.empty()) o["error"] = p.error;
    return o;
}

// Files of `e` to fetch for a pull of `kinds`; installed ones are skipped.
std::vector<std::pair<CatalogFile, std::string>> pull_list(const std::string& models_dir, const CatalogEntry& e, const std::vector<std::string>& kinds,
                                                          std::vector<std::string>* skipped) {
    std::vector<std::pair<CatalogFile, std::string>> v;
    for (const auto& k : {"model", "mmproj", "flash"}) {
        if (std::find(kinds.begin(), kinds.end(), k) == kinds.end()) continue;
        const CatalogFile* f = e.file(k);
        if (!f) continue;
        if (!installed_path(models_dir, e, *f).empty()) { if (skipped) skipped->push_back(f->name); continue; }
        v.push_back({*f, target_path(models_dir, e, *f)});
    }
    return v;
}

json model_json(Registry& reg, Downloads& dl, const CatalogEntry& e, int64_t created, uint64_t vram_total) {
    auto eng = reg.get();
    const bool loaded = eng && reg.loaded_id == e.id;
    json files = json::array();
    bool model_ok = false, any_part = false;
    std::string flash_path, mmproj_path;
    for (const auto& f : e.files) {
        const std::string p = installed_path(reg.models_dir, e, f);
        std::error_code ec;
        std::filesystem::path part = std::filesystem::u8path(target_path(reg.models_dir, e, f));
        part += ".part";
        if (std::filesystem::exists(part, ec)) any_part = true;
        if (f.kind == "model") model_ok = !p.empty();
        if (f.kind == "flash") flash_path = p;
        if (f.kind == "mmproj") mmproj_path = p;
        files.push_back({{"kind", f.kind}, {"name", f.name}, {"size_mb", mb(f.size)}, {"installed", !p.empty()}, {"url", f.url}});
    }
    std::string status = model_ok ? "installed" : any_part ? "partial" : "not_installed";
    json job = nullptr;
    {
        std::lock_guard<std::mutex> lk(dl.mu);
        if (auto j = dl.running_for(e.id)) { status = "downloading"; job = progress_json(*j); }
    }
    uint64_t total = 0;
    for (const auto& f : e.files) total += f.size;
    const uint64_t vmb = (vram_total >> 20) + (vram_total >> 24);   // + ~6%: a "4 GB" card reports 4095 MiB
    const char* fits = vmb >= static_cast<uint64_t>(e.full_vram_mb) ? "full" : vmb >= static_cast<uint64_t>(e.min_vram_mb) ? "partial" : "no";
    json flash = {{"available", e.file("flash") != nullptr}, {"installed", !flash_path.empty()}, {"active", loaded && !eng->info().flash.empty()}};
    if (!e.file("flash")) flash["note"] = "no drafter exists for this model yet";
    else if (loaded && eng->info().flash.empty()) flash["note"] = eng->info().flash_note;
    json vision = {{"available", e.file("mmproj") != nullptr}, {"installed", !mmproj_path.empty()}, {"active", loaded && eng->info().vision}};
    json d = {{"name", e.name}, {"description", e.description}, {"family", e.family}, {"base_model", e.base_model}, {"quant", e.quant},
              {"status", status}, {"loaded", loaded}, {"download_size_mb", mb(total)}, {"files", files}, {"flash", flash}, {"vision", vision},
              {"fits_gpu", fits}, {"min_vram_mb", e.min_vram_mb}, {"full_vram_mb", e.full_vram_mb}, {"download", job}};
    if (loaded) d["context"] = eng->info().context_capacity;
    if (!model_ok && job.is_null()) d["pull"] = {{"method", "POST"}, {"url", "/dara/models/pull"}, {"body", {{"model", e.id}}}};
    return {{"id", e.id}, {"object", "model"}, {"created", created}, {"owned_by", "dara"}, {"dara", d}};
}

std::string pull_usage() {
    return "usage: dara pull <model> [--no-vision] [--no-flash] [--models-dir DIR]\n"
           "       dara list [--models-dir DIR]\n"
           "       dara rm <model> [--models-dir DIR]\n"
           "models: " + Registry::ids() + " (short names: nano, 4b, 9b)\n";
}

}  // namespace

// dara pull / list / rm
int dara_models_main(int argc, char** argv) {
    if (argc < 1) { std::fprintf(stderr, "%s", pull_usage().c_str()); return 2; }
    const std::string cmd = argv[0];
    std::string models_dir = default_models_dir(), model;
    std::vector<std::string> kinds = {"model", "mmproj", "flash"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--models-dir" && i + 1 < argc) models_dir = argv[++i];
        else if (a == "--no-vision") kinds.erase(std::remove(kinds.begin(), kinds.end(), "mmproj"), kinds.end());
        else if (a == "--no-flash") kinds.erase(std::remove(kinds.begin(), kinds.end(), "flash"), kinds.end());
        else if (model.empty() && a[0] != '-') model = a;
        else { std::fprintf(stderr, "unknown argument %s\n%s", a.c_str(), pull_usage().c_str()); return 2; }
    }
    if (cmd == "list") {
        std::printf("models folder: %s\n", models_dir.c_str());
        for (const auto& e : catalog()) {
            std::printf("%-14s %-14s", e.id.c_str(), e.name.c_str());
            for (const auto& f : e.files) std::printf("  %s:%s", f.kind.c_str(), installed_path(models_dir, e, f).empty() ? "-" : "yes");
            std::printf("\n");
        }
        return 0;
    }
    const CatalogEntry* e = catalog_find(model);
    if (!e) { std::fprintf(stderr, "unknown model '%s'\n%s", model.c_str(), pull_usage().c_str()); return 2; }
    if (cmd == "rm") {
        int n = 0;
        std::error_code ec;
        for (const auto& f : e->files) {
            const auto t = std::filesystem::u8path(target_path(models_dir, *e, f));
            for (const char* suf : {"", ".part", ".part.chunks"}) {
                auto p = t; p += suf;
                if (std::filesystem::remove(p, ec)) ++n;
            }
        }
        std::printf("removed %d file(s) of %s from %s\n", n, e->id.c_str(), models_dir.c_str());
        return 0;
    }
    if (cmd != "pull") { std::fprintf(stderr, "%s", pull_usage().c_str()); return 2; }
    std::vector<std::string> skipped;
    auto files = pull_list(models_dir, *e, kinds, &skipped);
    for (const auto& s : skipped) std::printf("have %s\n", s.c_str());
    if (files.empty()) { std::printf("%s is installed\n", e->id.c_str()); return 0; }
    DownloadJob job("cli", e->id, files);
    job.start();
    while (!job.finished()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const auto p = job.progress();
        const double rate = p.seconds > 0 ? p.bytes_done / p.seconds : 0;
        std::printf("\r%s %-28s %6.1f%%  %7.0f / %.0f MB  %5.1f MB/s   ", p.state.c_str(), p.current_file.c_str(), p.bytes_total ? 100.0 * p.bytes_done / p.bytes_total : 0.0,
                    p.bytes_done / 1048576.0, p.bytes_total / 1048576.0, rate / 1048576.0);
        std::fflush(stdout);
    }
    job.wait();
    const auto p = job.progress();
    std::printf("\n%s: %s%s%s\n", e->id.c_str(), p.state.c_str(), p.error.empty() ? "" : " - ", p.error.c_str());
    return p.state == "done" ? 0 : 1;
}

int dara_serve_main(int argc, char** argv) {
    Registry reg;
    reg.models_dir = default_models_dir();
    LoadSpec initial;
    std::string host = "127.0.0.1", api_key;
    int port = 11435, default_max = 1024;
    std::vector<std::string> cors = {"http://localhost", "http://127.0.0.1", "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--model" || a == "--gguf" || a == "-m") initial.model = next();
        else if (a == "--models-dir") reg.models_dir = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--mmproj") { const std::string v = next(); if (v == "auto") reg.default_vision = true; else { initial.mmproj = v; reg.default_vision = true; } }
        else if (a == "--vision") reg.default_vision = true;
        else if (a == "--flash") initial.flash = next();
        else if (a == "--no-spec") reg.base.spec = false;
        else if (a == "--spec") reg.base.spec_k = std::atoi(next().c_str());
        else if (a == "--flash-min-p") reg.base.flash_min_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--kv" || a == "--ctx") { const std::string v = next(); reg.base.kv = v == "auto" ? 0 : static_cast<size_t>(std::atoll(v.c_str())); }
        else if (a == "--min-context") reg.base.min_context = static_cast<size_t>(std::atoll(next().c_str()));
        else if (a == "--vram-reserve") reg.base.vram_reserve_mb = static_cast<size_t>(std::atoll(next().c_str()));
        else if (a == "--image-max-tokens") reg.base.image_max_tokens = std::atoi(next().c_str());
        else if (a == "--compact-reserve") reg.base.compact_reserve = std::atoi(next().c_str());
        else if (a == "--max-tokens") default_max = std::atoi(next().c_str());
        else if (a == "--api-key") api_key = next();
        else if (a == "--cors-origin") cors.push_back(next());
        else if (initial.model.empty() && a[0] != '-') initial.model = a;
        else {
            std::fprintf(stderr, "unknown argument %s\n"
                         "usage: dara serve [--model <id or .gguf>] [--models-dir DIR] [--port 11435] [--host 127.0.0.1] [--vision | --mmproj <gguf>|auto]\n"
                         "                  [--flash auto|on|off|<drafter.dflash>] [--no-spec] [--ctx auto|N] [--min-context N] [--image-max-tokens N]\n"
                         "                  [--max-tokens N] [--api-key KEY] [--cors-origin ORIGIN]\n"
                         "model ids: %s\n", a.c_str(), Registry::ids().c_str());
            return 2;
        }
    }
    if (!initial.model.empty()) {
        try {
            std::lock_guard<std::mutex> lk(reg.engine_mu);
            reg.load(initial);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ERROR: %s%s\n", e.what(), dynamic_cast<const ModelNotFound*>(&e) ? (" (dara pull " + initial.model + ")").c_str() : "");
            return 1;
        }
    } else {
        std::fprintf(stderr, "no model loaded yet: requests naming a Dara model load it (installed: models folder %s)\n", reg.models_dir.c_str());
    }

    Downloads dl;
    std::atomic<int> queued{0};
    std::atomic<uint64_t> req_id{0};
    const auto t_start = Clock::now();
    const auto created = static_cast<int64_t>(std::time(nullptr));
    const uint64_t vram_total = gpu_telemetry().vram_total;

    httplib::Server svr;
    svr.set_payload_max_length(static_cast<size_t>(96) << 20);
    svr.set_read_timeout(600, 0);
    svr.set_write_timeout(600, 0);
    svr.set_keep_alive_timeout(30);

    auto allowed_origin = [&](const httplib::Request& req) -> std::string {
        if (!req.has_header("Origin")) return "";
        const std::string o = req.get_header_value("Origin");
        for (const auto& c : cors) {
            if (o == c) return o;
            if (o.rfind(c + ":", 0) == 0) return o;   // any port
        }
        return "";
    };
    svr.set_pre_routing_handler([&](const httplib::Request& req, httplib::Response& res) {
        const std::string o = allowed_origin(req);
        if (!o.empty()) {
            res.set_header("Access-Control-Allow-Origin", o);
            res.set_header("Vary", "Origin");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        }
        if (req.method == "OPTIONS") { res.status = 204; return httplib::Server::HandlerResponse::Handled; }
        if (!api_key.empty() && req.path != "/health") {
            const std::string auth = req.get_header_value("Authorization");
            if (auth != "Bearer " + api_key) {
                res.status = 401;
                res.set_content(dump(error_json("invalid or missing API key", "authentication_error")), "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });
    svr.set_logger([](const httplib::Request& req, const httplib::Response& res) {
        if (req.path == "/dara/stats" || req.path == "/metrics" || req.path == "/health" || req.path.rfind("/dara/models/pull/", 0) == 0) return;
        std::fprintf(stderr, "%s %s %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });

    auto bad = [](httplib::Response& res, int code, const std::string& msg, const char* type) {
        res.status = code;
        res.set_content(dump(error_json(msg, type)), "application/json");
    };
    auto not_found = [&](httplib::Response& res, const std::string& model, const std::string& msg) {
        json j = error_json(msg, "model_not_found");
        if (const CatalogEntry* e = catalog_find(model)) {
            j["error"]["code"] = "model_not_found";
            j["error"]["download"] = {{"method", "POST"}, {"url", "/dara/models/pull"}, {"body", {{"model", e->id}}}};
        }
        res.status = 404;
        res.set_content(dump(j), "application/json");
    };

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        auto e = reg.get();
        res.set_content("Dara is running" + (e ? " (model " + e->info().id + ")" : std::string(" (no model loaded)")) + "\n", "text/plain");
    });
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        auto e = reg.get();
        json j = {{"status", "ok"}, {"model", e ? json(e->info().id) : json(nullptr)}, {"queued", queued.load()}};
        if (e) j["busy"] = e->stats().busy;
        res.set_content(dump(j), "application/json");
    });

    // ---- models: catalog, downloads, loading ----
    auto list_models = [&](const httplib::Request&, httplib::Response& res) {
        json data = json::array();
        for (const auto& e : catalog()) data.push_back(model_json(reg, dl, e, created, vram_total));
        if (auto e = reg.get(); e && reg.loaded_id.empty()) {   // a model loaded from a path
            const ModelInfo& mi = e->info();
            data.push_back({{"id", mi.id}, {"object", "model"}, {"created", created}, {"owned_by", "local"},
                            {"dara", {{"name", mi.id}, {"family", mi.family}, {"quant", mi.quant}, {"status", "installed"}, {"loaded", true},
                                      {"context", mi.context_capacity}, {"flash", {{"available", !mi.flash.empty()}, {"active", !mi.flash.empty()}, {"note", mi.flash_note}}},
                                      {"vision", {{"available", mi.vision}, {"active", mi.vision}}}}}});
        }
        res.set_content(dump({{"object", "list"}, {"models_dir", reg.models_dir}, {"data", data}}), "application/json");
    };
    svr.Get("/v1/models", list_models);
    svr.Get("/dara/models", list_models);
    svr.Get(R"(/v1/models/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        const CatalogEntry* e = catalog_find(req.matches[1]);
        if (!e) { bad(res, 404, "unknown model '" + std::string(req.matches[1]) + "'", "model_not_found"); return; }
        res.set_content(dump(model_json(reg, dl, *e, created, vram_total)), "application/json");
    });

    svr.Post("/dara/models/pull", [&](const httplib::Request& req, httplib::Response& res) {
        json b;
        try { b = json::parse(req.body); } catch (const std::exception& ex) { bad(res, 400, std::string("invalid JSON: ") + ex.what(), "invalid_request_error"); return; }
        const std::string model = b.value("model", b.value("name", ""));
        const CatalogEntry* e = catalog_find(model);
        if (!e) { bad(res, 404, "unknown model '" + model + "'; available: " + Registry::ids(), "model_not_found"); return; }
        std::vector<std::string> kinds = {"model", "mmproj", "flash"};
        if (b.contains("files") && b["files"].is_array()) { kinds.clear(); for (const auto& k : b["files"]) kinds.push_back(k.get<std::string>()); }
        if (b.value("vision", true) == false) kinds.erase(std::remove(kinds.begin(), kinds.end(), "mmproj"), kinds.end());
        if (b.contains("flash") && b["flash"].is_boolean() && !b["flash"].get<bool>()) kinds.erase(std::remove(kinds.begin(), kinds.end(), "flash"), kinds.end());
        std::shared_ptr<DownloadJob> job;
        std::vector<std::string> skipped;
        {
            std::lock_guard<std::mutex> lk(dl.mu);
            job = dl.running_for(e->id);
            if (!job) {
                auto files = pull_list(reg.models_dir, *e, kinds, &skipped);
                if (files.empty()) {
                    res.set_content(dump({{"model", e->id}, {"status", "installed"}, {"skipped", skipped}}), "application/json");
                    return;
                }
                job = std::make_shared<DownloadJob>("pull-" + std::to_string(++dl.next), e->id, std::move(files));
                dl.jobs[job->id()] = job;
                job->start();
            }
        }
        if (!b.value("stream", false)) {
            res.status = 202;
            json j = progress_json(*job);
            j["skipped"] = skipped;
            j["progress_url"] = "/dara/models/pull/" + job->id();
            res.set_content(dump(j), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [job](size_t, httplib::DataSink& sink) {
            for (;;) {
                const bool fin = job->finished();
                const std::string s = "data: " + dump(progress_json(*job)) + "\n\n";
                if (!sink.write(s.data(), s.size())) return false;   // client went away; the download continues
                if (fin) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            const char* done = "data: [DONE]\n\n";
            sink.write(done, std::strlen(done));
            sink.done();
            return true;
        });
    });
    svr.Get(R"(/dara/models/pull/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(dl.mu);
        auto it = dl.jobs.find(req.matches[1]);
        if (it == dl.jobs.end()) { bad(res, 404, "no such download", "not_found"); return; }
        res.set_content(dump(progress_json(*it->second)), "application/json");
    });
    svr.Delete(R"(/dara/models/pull/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(dl.mu);
        auto it = dl.jobs.find(req.matches[1]);
        if (it == dl.jobs.end()) { bad(res, 404, "no such download", "not_found"); return; }
        it->second->cancel();
        res.set_content(dump({{"id", it->first}, {"status", "cancelling"}}), "application/json");
    });
    svr.Delete(R"(/dara/models/([^/]+))", [&](const httplib::Request& req, httplib::Response& res) {
        const CatalogEntry* e = catalog_find(req.matches[1]);
        if (!e) { bad(res, 404, "unknown model", "model_not_found"); return; }
        { std::lock_guard<std::mutex> lk(dl.mu); if (dl.running_for(e->id)) { bad(res, 409, "a download of this model is running; cancel it first", "conflict"); return; } }
        if (reg.get() && reg.loaded_id == e->id) { bad(res, 409, "the model is loaded; POST /dara/models/unload first", "conflict"); return; }
        int n = 0;
        std::error_code ec;
        for (const auto& f : e->files) {
            const auto t = std::filesystem::u8path(target_path(reg.models_dir, *e, f));
            for (const char* suf : {"", ".part", ".part.chunks"}) { auto p = t; p += suf; if (std::filesystem::remove(p, ec)) ++n; }
        }
        res.set_content(dump({{"model", e->id}, {"deleted_files", n}}), "application/json");
    });
    svr.Post("/dara/models/load", [&](const httplib::Request& req, httplib::Response& res) {
        LoadSpec s;
        try {
            const json b = json::parse(req.body);
            s.model = b.value("model", "");
            if (b.contains("flash")) { const json& f = b["flash"]; s.flash = f.is_boolean() ? (f.get<bool>() ? "on" : "off") : f.get<std::string>(); }
            if (b.contains("vision")) s.vision = b["vision"].get<bool>() ? 1 : 0;
            if (b.contains("ctx")) { const json& c = b["ctx"]; s.ctx = c.is_string() ? 0 : c.get<long long>(); }
        } catch (const std::exception& ex) { bad(res, 400, ex.what(), "invalid_request_error"); return; }
        if (s.model.empty()) { bad(res, 400, "\"model\" is required", "invalid_request_error"); return; }
        ++queued;
        std::lock_guard<std::mutex> lk(reg.engine_mu);
        --queued;
        try {
            reg.load(s);
            const ModelInfo& mi = reg.get()->info();
            res.set_content(dump({{"model", mi.id}, {"context", mi.context_capacity}, {"layers_on_gpu", mi.layers_gpu}, {"layers", mi.layers},
                                  {"load_seconds", r2(mi.load_seconds)}, {"flash", {{"active", !mi.flash.empty()}, {"note", mi.flash_note}}}, {"vision", mi.vision}}),
                            "application/json");
        } catch (const ModelNotFound& ex) { not_found(res, s.model, ex.what()); }
        catch (const std::exception& ex) { bad(res, 500, ex.what(), "server_error"); }
    });
    svr.Post("/dara/models/unload", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lk(reg.engine_mu);
        reg.unload();
        res.set_content(dump({{"status", "unloaded"}}), "application/json");
    });

    // ---- generation ----
    // Runs fn on the engine for `model` under the engine lock (loading / switching as needed).
    auto run_locked = [&](const std::string& model, const std::function<GenResult(Engine&)>& fn, GenResult& out, std::string& err, int& code) -> bool {
        const auto tq = Clock::now();
        ++queued;
        std::unique_lock<std::mutex> lk(reg.engine_mu);
        --queued;
        const double qms = std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
        try {
            auto e = reg.ensure(model);
            out = fn(*e);
            out.timings.queue_ms = qms;
            return true;
        } catch (const ModelNotFound& e) { err = e.what(); code = 404; }
        catch (const std::invalid_argument& e) { err = e.what(); code = 400; }
        catch (const std::length_error& e) { err = e.what(); code = 400; }
        catch (const std::exception& e) { err = e.what(); code = 500; }
        return false;
    };

    auto log_result = [](const char* what, const GenResult& r) {
        std::fprintf(stderr, "  %s: prompt %d (cached %d) + %d tokens | prefill %d tok %.0f ms = %.1f tok/s | ttft %.0f ms | decode %.2f tok/s%s%s | %s\n", what,
                     r.usage.prompt_tokens, r.usage.cached_tokens, r.usage.completion_tokens, r.timings.prefilled, r.timings.prefill_ms, r.timings.prefill_tps(),
                     r.timings.ttft_ms, r.timings.decode_tps(), r.spec.mode.empty() ? "" : (" (" + r.spec.mode + ")").c_str(),
                     r.context.compacted ? " | compacted" : "", r.finish_reason.c_str());
    };

    auto handle_gen = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); } catch (const std::exception& e) { bad(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error"); return; }
        std::vector<Message> msgs;
        std::string prompt;
        SampleParams sp;
        const std::string model = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : std::string();
        try {
            if (chat) msgs = parse_messages(body.value("messages", json()));
            else {
                const json& p = body.value("prompt", json());
                if (p.is_string()) prompt = p.get<std::string>();
                else if (p.is_array() && !p.empty() && p[0].is_string()) prompt = p[0].get<std::string>();
                else throw std::invalid_argument("'prompt' must be a string");
            }
            sp = parse_sampling(body, default_max);
            if (body.contains("flash")) {   // Dara extension: speculative decoding on / off for this request
                const json& f = body["flash"];
                if (f.is_boolean()) sp.flash = f.get<bool>() ? 1 : 0;
                else if (f.is_string()) { const std::string v = f.get<std::string>(); sp.flash = v == "on" || v == "true" ? 1 : v == "off" || v == "false" ? 0 : -1; }
            }
        } catch (const std::exception& e) { bad(res, 400, e.what(), "invalid_request_error"); return; }
        if (const CatalogEntry* ce = catalog_find(model)) {   // a Dara model that is not installed: 404 with the download hint, before streaming starts
            if (installed_path(reg.models_dir, *ce, *ce->file("model")).empty()) { not_found(res, model, "model '" + ce->id + "' is not installed"); return; }
        }
        const bool stream = body.value("stream", false);
        const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(created) + "-" + std::to_string(++req_id);
        const int64_t now = static_cast<int64_t>(std::time(nullptr));
        const char* obj = chat ? "chat.completion" : "text_completion";
        if (!stream) {
            GenResult r; std::string err; int code = 0;
            std::string used;
            if (!run_locked(model, [&](Engine& e) { used = e.info().id; return chat ? e.chat(msgs, sp, nullptr) : e.complete(prompt, sp, nullptr); }, r, err, code)) {
                if (code == 404) not_found(res, model, err); else bad(res, code, err, code == 400 ? "invalid_request_error" : "server_error");
                return;
            }
            log_result(chat ? "chat" : "completion", r);
            json choice = {{"index", 0}, {"finish_reason", r.finish_reason == "cancelled" ? "stop" : r.finish_reason}};
            if (chat) choice["message"] = {{"role", "assistant"}, {"content", r.text}};
            else choice["text"] = r.text;
            json j = {{"id", id}, {"object", obj}, {"created", now}, {"model", used}, {"choices", json::array({choice})}};
            add_dara_fields(j, r);
            res.set_content(dump(j), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&run_locked, &log_result, msgs, prompt, sp, id, now, chat, model](size_t, httplib::DataSink& sink) {
            const char* cobj = chat ? "chat.completion.chunk" : "text_completion";
            std::string used = model;
            auto send = [&](const json& j) { const std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            auto chunk = [&](const json& delta_or_text, const json& finish) {
                json c = {{"index", 0}, {"finish_reason", finish}};
                if (chat) c["delta"] = delta_or_text; else c["text"] = delta_or_text;
                return json{{"id", id}, {"object", cobj}, {"created", now}, {"model", used}, {"choices", json::array({c})}};
            };
            bool alive = true, started = false;
            Engine::TextFn fn = [&](const std::string& t) {
                if (!alive) return false;
                alive = send(chunk(chat ? json{{"content", t}} : json(t), nullptr));
                return alive;   // a closed connection cancels the generation
            };
            GenResult r; std::string err; int code = 0;
            const bool ok = run_locked(model, [&](Engine& e) {
                used = e.info().id;
                if (chat && !started) { started = true; alive = send(chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr)); }
                return chat ? e.chat(msgs, sp, fn) : e.complete(prompt, sp, fn);
            }, r, err, code);
            if (!ok) {
                send(error_json(err, code == 404 ? "model_not_found" : code == 400 ? "invalid_request_error" : "server_error"));
            } else {
                log_result(chat ? "chat (stream)" : "completion (stream)", r);
                if (alive) {
                    json last = chunk(chat ? json::object() : json(""), r.finish_reason == "cancelled" ? "stop" : r.finish_reason);
                    add_dara_fields(last, r);   // usage + timings on the final chunk (as stream_options.include_usage does)
                    send(last);
                }
            }
            if (alive) { const char* done = "data: [DONE]\n\n"; sink.write(done, std::strlen(done)); }
            sink.done();
            return true;
        });
    };
    svr.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle_gen(q, s, true); });
    svr.Post("/chat/completions", [&](const httplib::Request& q, httplib::Response& s) { handle_gen(q, s, true); });
    svr.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& s) { handle_gen(q, s, false); });

    svr.Post("/dara/tokenize", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json b = json::parse(req.body);
            std::lock_guard<std::mutex> lk(reg.engine_mu);
            auto e = reg.ensure(b.value("model", ""));
            if (b.contains("messages")) {
                const auto msgs = parse_messages(b["messages"]);
                res.set_content(dump({{"tokens", e->count_tokens(msgs)}, {"context", e->info().context_capacity}}), "application/json");
            } else {
                Message m; m.role = "user"; m.parts.push_back({false, b.value("text", ""), ""});
                res.set_content(dump({{"chat_tokens", e->count_tokens({m})}, {"context", e->info().context_capacity}}), "application/json");
            }
        } catch (const ModelNotFound& ex) { not_found(res, "", ex.what()); }
        catch (const std::exception& ex) { bad(res, 400, ex.what(), "invalid_request_error"); }
    });
    svr.Post("/dara/compact", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json b = json::parse(req.body);
            const auto msgs = parse_messages(b.value("messages", json()));
            const int max_tokens = b.value("max_tokens", default_max);
            std::lock_guard<std::mutex> lk(reg.engine_mu);
            auto e = reg.ensure(b.value("model", ""));
            ContextInfo ci; double ms = 0;
            const auto out = e->compact(msgs, max_tokens, &ci, &ms);
            json arr = json::array();
            for (const auto& m : out) {
                json parts = json::array();
                for (const auto& p : m.parts) parts.push_back(p.is_image ? json{{"type", "image_url"}, {"image_url", {{"url", "(image kept)"}}}} : json{{"type", "text"}, {"text", p.text}});
                arr.push_back({{"role", m.role}, {"content", parts}});
            }
            res.set_content(dump({{"messages", arr}, {"tokens_before", e->count_tokens(msgs)}, {"tokens_after", e->count_tokens(out)},
                                  {"context", context_json(ci)}, {"compact_ms", r2(ms)}}), "application/json");
        } catch (const ModelNotFound& ex) { not_found(res, "", ex.what()); }
        catch (const std::invalid_argument& ex) { bad(res, 400, ex.what(), "invalid_request_error"); }
        catch (const std::exception& ex) { bad(res, 500, ex.what(), "server_error"); }
    });

    auto stats_json = [&]() {
        auto eng = reg.get();
        const ProcessMemory pm = process_memory();
        const SystemMemory sm = system_memory();
        const GpuTelemetry g = gpu_telemetry();
        const DiskSpace d = disk_space(eng ? eng->info().path : reg.models_dir);
        const double up = std::chrono::duration<double>(Clock::now() - t_start).count();
        json j;
        j["uptime_seconds"] = r2(up);
#ifdef _WIN32
        j["pid"] = _getpid();
#endif
        j["process"] = {{"private_working_set_mb", mb(pm.private_working_set)}, {"working_set_mb", mb(pm.working_set)},
                        {"peak_working_set_mb", mb(pm.peak_working_set)}, {"committed_mb", mb(pm.committed)}};
        j["ram"] = {{"total_mb", mb(sm.total)}, {"available_mb", mb(sm.available)}, {"used_mb", mb(sm.total - sm.available)},
                    {"used_percent", sm.total ? r2(100.0 * (sm.total - sm.available) / sm.total) : 0.0}};
        j["gpu"] = {{"name", g.name}, {"compute_capability", std::to_string(g.cc_major) + "." + std::to_string(g.cc_minor)},
                    {"vram_total_mb", mb(g.vram_total)}, {"vram_free_mb", mb(g.vram_free)}, {"vram_used_mb", mb(g.vram_total - g.vram_free)},
                    {"vram_used_percent", g.vram_total ? r2(100.0 * (g.vram_total - g.vram_free) / g.vram_total) : 0.0},
                    {"vram_weights_mb", eng ? mb(eng->info().weights_vram_bytes) : 0.0}, {"nvml", g.nvml}};
        if (g.nvml) {
            j["gpu"]["utilization_percent"] = g.util_gpu;
            j["gpu"]["memory_utilization_percent"] = g.util_mem;
            j["gpu"]["temperature_c"] = g.temp_c;
            j["gpu"]["power_w"] = r2(g.power_w);
            j["gpu"]["power_limit_w"] = r2(g.power_limit_w);
            j["gpu"]["clock_sm_mhz"] = g.clock_sm_mhz;
            j["gpu"]["clock_mem_mhz"] = g.clock_mem_mhz;
            j["gpu"]["pstate"] = g.pstate;
        }
        uint64_t installed_bytes = 0;
        for (const auto& e : catalog()) for (const auto& f : e.files) if (!installed_path(reg.models_dir, e, f).empty()) installed_bytes += f.size;
        j["disk"] = {{"models_dir", reg.models_dir}, {"installed_models_mb", mb(installed_bytes)}, {"volume_total_gb", r2(d.total / 1073741824.0)},
                     {"volume_free_gb", r2(d.free / 1073741824.0)}};
        {
            std::lock_guard<std::mutex> lk(dl.mu);
            json jobs = json::array();
            for (auto& [k, job] : dl.jobs) if (!job->finished()) jobs.push_back(progress_json(*job));
            j["downloads"] = jobs;
        }
        j["queued"] = queued.load();
        if (!eng) { j["model"] = nullptr; j["context"] = nullptr; j["engine"] = nullptr; j["last_request"] = nullptr; return j; }
        const ModelInfo& mi = eng->info();
        const EngineStats s = eng->stats();
        std::error_code ec;
        const auto fsz = [&](const std::string& p) { return p.empty() ? 0.0 : mb(static_cast<uint64_t>(std::filesystem::file_size(std::filesystem::u8path(p), ec))); };
        j["disk"]["model_file_mb"] = mb(mi.file_bytes);
        j["disk"]["mmproj_file_mb"] = mb(mi.mmproj_bytes);
        j["disk"]["flash_file_mb"] = fsz(mi.flash);
        j["model"] = {{"id", mi.id}, {"family", mi.family}, {"path", mi.path}, {"quant", mi.quant}, {"layers", mi.layers}, {"layers_on_gpu", mi.layers_gpu},
                      {"output_head", mi.head_on_cpu ? "cpu" : "gpu"}, {"hidden", mi.hidden}, {"vocab", mi.vocab}, {"kv_format", mi.kv_format},
                      {"native_context", mi.native_context}, {"load_seconds", r2(mi.load_seconds)}, {"vision", mi.vision}, {"mmproj", mi.mmproj},
                      {"image_max_tokens", mi.image_max_tokens}, {"flash", mi.flash}, {"flash_note", mi.flash_note}, {"spec_k", mi.spec_k}, {"placement", mi.placement}};
        j["context"] = {{"capacity", mi.context_capacity}, {"held_tokens", s.has_last ? s.last.context.used : 0}};
        j["engine"] = {{"busy", s.busy}, {"live_tokens", s.live_tokens}, {"live_tokens_per_second", r2(s.live_tps)},
                       {"requests", s.requests}, {"errors", s.errors}, {"compactions", s.compactions}, {"prompt_tokens_total", s.prompt_tokens},
                       {"cached_tokens_total", s.cached_tokens}, {"completion_tokens_total", s.completion_tokens},
                       {"avg_prefill_tokens_per_second", s.prefill_ms > 0 ? r2(s.prefilled * 1000.0 / s.prefill_ms) : 0.0},
                       {"avg_decode_tokens_per_second", s.decode_ms > 0 ? r2(s.decoded * 1000.0 / s.decode_ms) : 0.0}};
        if (s.has_last) {
            json l;
            l["finish_reason"] = s.last.finish_reason;
            add_dara_fields(l, s.last);
            j["last_request"] = l;
        } else j["last_request"] = nullptr;
        return j;
    };
    svr.Get("/dara/stats", [&](const httplib::Request&, httplib::Response& res) { res.set_content(dump(stats_json()), "application/json"); });
    svr.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
        const json j = stats_json();
        std::ostringstream o;
        auto g = [&](const char* name, double v, const char* help) { o << "# HELP " << name << " " << help << "\n# TYPE " << name << " gauge\n" << name << " " << v << "\n"; };
        auto num = [](const json& x) { return x.is_number() ? x.get<double>() : x.is_boolean() ? (x.get<bool>() ? 1.0 : 0.0) : 0.0; };
        g("dara_uptime_seconds", num(j["uptime_seconds"]), "Seconds since the server started");
        g("dara_process_private_mb", num(j["process"]["private_working_set_mb"]), "Private working set of dara.exe (MB)");
        g("dara_process_working_set_mb", num(j["process"]["working_set_mb"]), "Working set incl. mapped model pages (MB)");
        g("dara_ram_available_mb", num(j["ram"]["available_mb"]), "Available system RAM (MB)");
        g("dara_vram_used_mb", num(j["gpu"]["vram_used_mb"]), "VRAM in use on the GPU (MB)");
        g("dara_vram_total_mb", num(j["gpu"]["vram_total_mb"]), "VRAM total (MB)");
        if (j["gpu"].contains("utilization_percent")) {
            g("dara_gpu_utilization_percent", num(j["gpu"]["utilization_percent"]), "GPU utilization");
            g("dara_gpu_temperature_c", num(j["gpu"]["temperature_c"]), "GPU temperature");
            g("dara_gpu_power_w", num(j["gpu"]["power_w"]), "GPU power draw");
        }
        g("dara_disk_free_gb", num(j["disk"]["volume_free_gb"]), "Free space on the models volume (GB)");
        g("dara_queued_requests", num(j["queued"]), "Requests waiting for the engine");
        g("dara_downloads_running", static_cast<double>(j["downloads"].size()), "Model downloads in progress");
        g("dara_model_loaded", j["model"].is_null() ? 0 : 1, "1 when a model is loaded");
        if (!j["model"].is_null()) {
            g("dara_context_capacity_tokens", num(j["context"]["capacity"]), "Context size");
            g("dara_context_held_tokens", num(j["context"]["held_tokens"]), "Tokens held in the model state");
            g("dara_busy", num(j["engine"]["busy"]), "1 while a request runs");
            g("dara_live_tokens_per_second", num(j["engine"]["live_tokens_per_second"]), "Decode speed of the running request");
            g("dara_requests_total", num(j["engine"]["requests"]), "Completed requests");
            g("dara_errors_total", num(j["engine"]["errors"]), "Failed requests");
            g("dara_prompt_tokens_total", num(j["engine"]["prompt_tokens_total"]), "Prompt tokens");
            g("dara_cached_tokens_total", num(j["engine"]["cached_tokens_total"]), "Prompt tokens served from the prompt cache");
            g("dara_completion_tokens_total", num(j["engine"]["completion_tokens_total"]), "Generated tokens");
            g("dara_avg_prefill_tokens_per_second", num(j["engine"]["avg_prefill_tokens_per_second"]), "Average prefill speed");
            g("dara_avg_decode_tokens_per_second", num(j["engine"]["avg_decode_tokens_per_second"]), "Average decode speed");
            if (!j["last_request"].is_null()) {
                g("dara_last_prefill_tokens_per_second", num(j["last_request"]["timings"]["prefill_tokens_per_second"]), "Last request prefill speed");
                g("dara_last_decode_tokens_per_second", num(j["last_request"]["timings"]["decode_tokens_per_second"]), "Last request decode speed");
                g("dara_last_ttft_ms", num(j["last_request"]["timings"]["ttft_ms"]), "Last request time to first token");
            }
        }
        res.set_content(o.str(), "text/plain; version=0.0.4");
    });

    std::fprintf(stderr, "Dara server listening on http://%s:%d  (OpenAI API at /v1, models at /v1/models, stats at /dara/stats)\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "ERROR: cannot listen on %s:%d (port in use?)\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
