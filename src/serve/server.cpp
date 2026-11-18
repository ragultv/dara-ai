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
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

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

int dara_serve_main(int argc, char** argv) {
    EngineOptions eo;
    std::string host = "127.0.0.1", api_key, mmproj = "";
    int port = 11435, default_max = 1024;
    std::vector<std::string> cors = {"http://localhost", "http://127.0.0.1", "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--model" || a == "--gguf" || a == "-m") eo.gguf = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--flash") { const std::string v = next(); eo.flash = v == "off" ? "" : v; }
        else if (a == "--no-spec") eo.spec = false;
        else if (a == "--spec") eo.spec_k = std::atoi(next().c_str());
        else if (a == "--flash-min-p") eo.flash_min_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--kv" || a == "--ctx") { const std::string v = next(); eo.kv = v == "auto" ? 0 : static_cast<size_t>(std::atoll(v.c_str())); }
        else if (a == "--min-context") eo.min_context = static_cast<size_t>(std::atoll(next().c_str()));
        else if (a == "--image-max-tokens") eo.image_max_tokens = std::atoi(next().c_str());
        else if (a == "--compact-reserve") eo.compact_reserve = std::atoi(next().c_str());
        else if (a == "--max-tokens") default_max = std::atoi(next().c_str());
        else if (a == "--api-key") api_key = next();
        else if (a == "--cors-origin") cors.push_back(next());
        else if (a.size() > 5 && a.substr(a.size() - 5) == ".gguf" && eo.gguf.empty()) eo.gguf = a;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (eo.gguf.empty()) {
        std::fprintf(stderr,
                     "usage: dara serve --model <model.gguf> [--port 11435] [--host 127.0.0.1] [--mmproj <vision.gguf>|auto]\n"
                     "                  [--flash auto|off|<drafter.dflash>] [--no-spec] [--ctx auto|N] [--min-context N] [--image-max-tokens N]\n"
                     "                  [--max-tokens N] [--api-key KEY] [--cors-origin ORIGIN]\n");
        return 2;
    }
    if (mmproj == "auto") {
        mmproj = find_mmproj(eo.gguf);
        if (mmproj.empty()) std::fprintf(stderr, "note: --mmproj auto found no single mmproj*.gguf next to the model; image input is off\n");
    }
    eo.mmproj = mmproj;

    std::unique_ptr<Engine> engine;
    try {
        std::fprintf(stderr, "loading %s ...\n", eo.gguf.c_str());
        engine = std::make_unique<Engine>(eo);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    const ModelInfo& mi = engine->info();
    std::fprintf(stderr, "model %s (%s, %s) | %s\n  context %zu tokens | layers on GPU %d/%d%s | load %.1f s | flash %s | vision %s\n", mi.id.c_str(), mi.family.c_str(),
                 mi.quant.c_str(), mi.gpu.c_str(), mi.context_capacity, mi.layers_gpu, mi.layers, mi.head_on_cpu ? ", output head in RAM" : "", mi.load_seconds,
                 mi.flash.empty() ? "off" : mi.flash.c_str(), mi.vision ? mi.mmproj.c_str() : "off");

    std::mutex engine_mu;
    std::atomic<int> queued{0};
    std::atomic<uint64_t> req_id{0};
    const auto t_start = Clock::now();
    const auto created = static_cast<int64_t>(std::time(nullptr));
    size_t flash_bytes = 0;
    {
        std::error_code ec;
        if (!mi.flash.empty()) flash_bytes = static_cast<size_t>(std::filesystem::file_size(std::filesystem::u8path(mi.flash), ec));
    }

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
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
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
        if (req.path == "/dara/stats" || req.path == "/metrics" || req.path == "/health") return;
        std::fprintf(stderr, "%s %s %d\n", req.method.c_str(), req.path.c_str(), res.status);
    });

    auto bad = [](httplib::Response& res, int code, const std::string& msg, const char* type) {
        res.status = code;
        res.set_content(dump(error_json(msg, type)), "application/json");
    };

    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) { res.set_content("Dara is running (model " + mi.id + ")\n", "text/plain"); });
    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        const EngineStats s = engine->stats();
        res.set_content(dump({{"status", "ok"}, {"model", mi.id}, {"busy", s.busy}, {"queued", queued.load()}}), "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        json m = {{"id", mi.id}, {"object", "model"}, {"created", created}, {"owned_by", "dara"},
                  {"meta", {{"family", mi.family}, {"quant", mi.quant}, {"context", mi.context_capacity}, {"vision", mi.vision}, {"flash", !mi.flash.empty()}}}};
        res.set_content(dump({{"object", "list"}, {"data", json::array({m})}}), "application/json");
    });

    // Runs one generation under the engine lock. `fn` does the call; returns false (with res filled) on failure.
    auto run_locked = [&](const std::function<GenResult()>& fn, GenResult& out, std::string& err, int& code) -> bool {
        const auto tq = Clock::now();
        ++queued;
        std::unique_lock<std::mutex> lk(engine_mu);
        --queued;
        const double qms = std::chrono::duration<double, std::milli>(Clock::now() - tq).count();
        try {
            out = fn();
            out.timings.queue_ms = qms;
            return true;
        } catch (const std::invalid_argument& e) { err = e.what(); code = 400; }
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

    // ---- chat / completions (shared streaming machinery) ----
    auto handle_gen = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
        json body;
        try { body = json::parse(req.body); } catch (const std::exception& e) { bad(res, 400, std::string("invalid JSON: ") + e.what(), "invalid_request_error"); return; }
        std::vector<Message> msgs;
        std::string prompt;
        SampleParams sp;
        try {
            if (chat) msgs = parse_messages(body.value("messages", json()));
            else {
                const json& p = body.value("prompt", json());
                if (p.is_string()) prompt = p.get<std::string>();
                else if (p.is_array() && !p.empty() && p[0].is_string()) prompt = p[0].get<std::string>();
                else throw std::invalid_argument("'prompt' must be a string");
            }
            sp = parse_sampling(body, default_max);
        } catch (const std::exception& e) { bad(res, 400, e.what(), "invalid_request_error"); return; }
        const bool stream = body.value("stream", false);
        const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(created) + "-" + std::to_string(++req_id);
        const int64_t now = static_cast<int64_t>(std::time(nullptr));
        const char* obj = chat ? "chat.completion" : "text_completion";
        if (!stream) {
            GenResult r; std::string err; int code = 0;
            if (!run_locked([&]() { return chat ? engine->chat(msgs, sp, nullptr) : engine->complete(prompt, sp, nullptr); }, r, err, code)) { bad(res, code, err, code == 400 ? "invalid_request_error" : "server_error"); return; }
            log_result(chat ? "chat" : "completion", r);
            json choice = {{"index", 0}, {"finish_reason", r.finish_reason == "cancelled" ? "stop" : r.finish_reason}};
            if (chat) choice["message"] = {{"role", "assistant"}, {"content", r.text}};
            else choice["text"] = r.text;
            json j = {{"id", id}, {"object", obj}, {"created", now}, {"model", mi.id}, {"choices", json::array({choice})}};
            add_dara_fields(j, r);
            res.set_content(dump(j), "application/json");
            return;
        }
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");
        res.set_chunked_content_provider("text/event-stream", [&engine, &mi, &run_locked, &log_result, msgs, prompt, sp, id, now, chat](size_t, httplib::DataSink& sink) {
            const char* cobj = chat ? "chat.completion.chunk" : "text_completion";
            auto send = [&](const json& j) { const std::string s = "data: " + dump(j) + "\n\n"; return sink.write(s.data(), s.size()); };
            auto chunk = [&](const json& delta_or_text, const json& finish) {
                json c = {{"index", 0}, {"finish_reason", finish}};
                if (chat) c["delta"] = delta_or_text; else c["text"] = delta_or_text;
                return json{{"id", id}, {"object", cobj}, {"created", now}, {"model", mi.id}, {"choices", json::array({c})}};
            };
            if (chat) send(chunk(json{{"role", "assistant"}, {"content", ""}}, nullptr));
            bool alive = true;
            Engine::TextFn fn = [&](const std::string& t) {
                if (!alive) return false;
                alive = send(chunk(chat ? json{{"content", t}} : json(t), nullptr));
                return alive;   // a closed connection cancels the generation
            };
            GenResult r; std::string err; int code = 0;
            const bool ok = run_locked([&]() { return chat ? engine->chat(msgs, sp, fn) : engine->complete(prompt, sp, fn); }, r, err, code);
            if (!ok) {
                send(error_json(err, code == 400 ? "invalid_request_error" : "server_error"));
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
            std::lock_guard<std::mutex> lk(engine_mu);
            if (b.contains("messages")) {
                const auto msgs = parse_messages(b["messages"]);
                res.set_content(dump({{"tokens", engine->count_tokens(msgs)}, {"context", mi.context_capacity}}), "application/json");
            } else {
                Message m; m.role = "user"; m.parts.push_back({false, b.value("text", ""), ""});
                res.set_content(dump({{"chat_tokens", engine->count_tokens({m})}, {"context", mi.context_capacity}}), "application/json");
            }
        } catch (const std::exception& e) { bad(res, 400, e.what(), "invalid_request_error"); }
    });
    svr.Post("/dara/compact", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            const json b = json::parse(req.body);
            const auto msgs = parse_messages(b.value("messages", json()));
            const int max_tokens = b.value("max_tokens", default_max);
            std::lock_guard<std::mutex> lk(engine_mu);
            ContextInfo ci; double ms = 0;
            const auto out = engine->compact(msgs, max_tokens, &ci, &ms);
            json arr = json::array();
            for (const auto& m : out) {
                json parts = json::array();
                for (const auto& p : m.parts) parts.push_back(p.is_image ? json{{"type", "image_url"}, {"image_url", {{"url", "(image kept)"}}}} : json{{"type", "text"}, {"text", p.text}});
                arr.push_back({{"role", m.role}, {"content", parts}});
            }
            res.set_content(dump({{"messages", arr}, {"tokens_before", engine->count_tokens(msgs)}, {"tokens_after", engine->count_tokens(out)},
                                  {"context", context_json(ci)}, {"compact_ms", r2(ms)}}), "application/json");
        } catch (const std::invalid_argument& e) { bad(res, 400, e.what(), "invalid_request_error"); }
        catch (const std::exception& e) { bad(res, 500, e.what(), "server_error"); }
    });

    auto stats_json = [&]() {
        const EngineStats s = engine->stats();
        const ProcessMemory pm = process_memory();
        const SystemMemory sm = system_memory();
        const GpuTelemetry g = gpu_telemetry();
        const DiskSpace d = disk_space(mi.path);
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
                    {"vram_weights_mb", mb(mi.weights_vram_bytes)}, {"nvml", g.nvml}};
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
        j["disk"] = {{"model_file_mb", mb(mi.file_bytes)}, {"mmproj_file_mb", mb(mi.mmproj_bytes)}, {"flash_file_mb", mb(flash_bytes)},
                     {"volume_total_gb", r2(d.total / 1073741824.0)}, {"volume_free_gb", r2(d.free / 1073741824.0)}};
        j["model"] = {{"id", mi.id}, {"family", mi.family}, {"path", mi.path}, {"quant", mi.quant}, {"layers", mi.layers}, {"layers_on_gpu", mi.layers_gpu},
                      {"output_head", mi.head_on_cpu ? "cpu" : "gpu"}, {"hidden", mi.hidden}, {"vocab", mi.vocab}, {"kv_format", mi.kv_format},
                      {"native_context", mi.native_context}, {"load_seconds", r2(mi.load_seconds)}, {"vision", mi.vision}, {"mmproj", mi.mmproj},
                      {"image_max_tokens", mi.image_max_tokens}, {"flash", mi.flash}, {"spec_k", mi.spec_k}, {"placement", mi.placement}};
        j["context"] = {{"capacity", mi.context_capacity}, {"held_tokens", s.has_last ? s.last.context.used : 0}};
        j["engine"] = {{"busy", s.busy}, {"queued", queued.load()}, {"live_tokens", s.live_tokens}, {"live_tokens_per_second", r2(s.live_tps)},
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
        g("dara_uptime_seconds", j["uptime_seconds"].get<double>(), "Seconds since the server started");
        g("dara_process_private_mb", j["process"]["private_working_set_mb"].get<double>(), "Private working set of dara.exe (MB)");
        g("dara_process_working_set_mb", j["process"]["working_set_mb"].get<double>(), "Working set incl. mapped model pages (MB)");
        g("dara_ram_available_mb", j["ram"]["available_mb"].get<double>(), "Available system RAM (MB)");
        g("dara_vram_used_mb", j["gpu"]["vram_used_mb"].get<double>(), "VRAM in use on the GPU (MB)");
        g("dara_vram_total_mb", j["gpu"]["vram_total_mb"].get<double>(), "VRAM total (MB)");
        if (j["gpu"].contains("utilization_percent")) {
            g("dara_gpu_utilization_percent", j["gpu"]["utilization_percent"].get<double>(), "GPU utilization");
            g("dara_gpu_temperature_c", j["gpu"]["temperature_c"].get<double>(), "GPU temperature");
            g("dara_gpu_power_w", j["gpu"]["power_w"].get<double>(), "GPU power draw");
        }
        g("dara_disk_free_gb", j["disk"]["volume_free_gb"].get<double>(), "Free space on the model's volume (GB)");
        g("dara_context_capacity_tokens", static_cast<double>(mi.context_capacity), "Context size");
        g("dara_context_held_tokens", j["context"]["held_tokens"].get<double>(), "Tokens held in the model state");
        g("dara_busy", j["engine"]["busy"].get<bool>() ? 1 : 0, "1 while a request runs");
        g("dara_queued_requests", j["engine"]["queued"].get<double>(), "Requests waiting for the engine");
        g("dara_live_tokens_per_second", j["engine"]["live_tokens_per_second"].get<double>(), "Decode speed of the running request");
        g("dara_requests_total", j["engine"]["requests"].get<double>(), "Completed requests");
        g("dara_errors_total", j["engine"]["errors"].get<double>(), "Failed requests");
        g("dara_prompt_tokens_total", j["engine"]["prompt_tokens_total"].get<double>(), "Prompt tokens");
        g("dara_cached_tokens_total", j["engine"]["cached_tokens_total"].get<double>(), "Prompt tokens served from the prompt cache");
        g("dara_completion_tokens_total", j["engine"]["completion_tokens_total"].get<double>(), "Generated tokens");
        g("dara_avg_prefill_tokens_per_second", j["engine"]["avg_prefill_tokens_per_second"].get<double>(), "Average prefill speed");
        g("dara_avg_decode_tokens_per_second", j["engine"]["avg_decode_tokens_per_second"].get<double>(), "Average decode speed");
        if (!j["last_request"].is_null()) {
            g("dara_last_prefill_tokens_per_second", j["last_request"]["timings"]["prefill_tokens_per_second"].get<double>(), "Last request prefill speed");
            g("dara_last_decode_tokens_per_second", j["last_request"]["timings"]["decode_tokens_per_second"].get<double>(), "Last request decode speed");
            g("dara_last_ttft_ms", j["last_request"]["timings"]["ttft_ms"].get<double>(), "Last request time to first token");
        }
        res.set_content(o.str(), "text/plain; version=0.0.4");
    });

    std::fprintf(stderr, "Dara server listening on http://%s:%d  (OpenAI API at /v1, stats at /dara/stats)\n", host.c_str(), port);
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "ERROR: cannot listen on %s:%d (port in use?)\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
