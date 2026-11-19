// Inference engine behind `dara serve`: one loaded model (Qwen3.5 Core 4B / Pro 9B or Gemma 4 Nano, detected from the GGUF), its tokenizer and chat
// template, optional image encoder (mmproj, loaded only while an image is encoded), sampling, a prompt cache (the next request reuses the K/V and
// recurrent state of the shared prefix), a context maintainer that compacts conversations that do not fit, and per-request usage / timings.
// Not thread-safe: the server serialises requests.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace dara::serve {

struct ContentPart {
    bool is_image = false;
    std::string text;    // text part
    std::string image;   // encoded image bytes (PNG / JPEG / ...)
};

struct Message {
    std::string role;                // "system", "user", "assistant" ("developer" = system, "tool" = user)
    std::vector<ContentPart> parts;  // in order
    std::string text() const;        // the text parts joined
    bool has_image() const;
};

struct SampleParams {
    int max_tokens = 512;
    float temperature = 0.7f;        // 0: greedy (argmax on the GPU)
    int top_k = 40;                  // 0: off
    float top_p = 0.95f;             // 1: off
    float min_p = 0.05f;             // 0: off
    float repeat_penalty = 1.0f;     // 1: off
    int repeat_last_n = 64;
    uint64_t seed = 0;               // 0: random
    std::vector<std::string> stop;   // stop strings (not included in the output)
    int flash = -1;                  // speculative decoding for this request: -1 auto (on when available), 0 off, 1 on (reported if unavailable)
    bool cache_write = true;         // false: read the prompt cache but keep the saved prefix (a side question does not displace the conversation)
};

struct Usage {
    int prompt_tokens = 0;           // tokens of the prompt as sent to the model (after compaction)
    int cached_tokens = 0;           // of those, reused from the previous request (no prefill)
    int completion_tokens = 0;
    int image_tokens = 0;
    int total_tokens() const { return prompt_tokens + completion_tokens; }
};

struct Timings {
    double queue_ms = 0;             // filled by the server: waiting for the engine
    double compact_ms = 0;           // context maintenance (summary generation)
    double image_ms = 0;             // image decode + encode
    double image_scratch_mb = 0;     // device memory the image encoding used: ...
    double image_borrowed_mb = 0;    // ... of which carved from the unused part of the context (no extra VRAM)
    double image_free_vram_mb = 0;   // ... and of which taken from free VRAM
    double prefill_ms = 0;           // prompt tokens through the model (excluding cached ones)
    double ttft_ms = 0;              // request start in the engine -> first generated token
    double decode_ms = 0;            // first token -> last token
    double total_ms = 0;
    double prefill_tps() const { return prefill_ms > 0 ? (prefilled * 1000.0 / prefill_ms) : 0; }
    double decode_tps() const { return decode_ms > 0 && decoded > 1 ? ((decoded - 1) * 1000.0 / decode_ms) : 0; }
    int prefilled = 0, decoded = 0;
};

struct ContextInfo {
    size_t capacity = 0;             // positions the model holds
    size_t used = 0;                 // after this request (prompt + completion)
    bool compacted = false;
    int summarized_messages = 0;     // old messages replaced by a summary
    int dropped_messages = 0;        // old messages dropped (no room even for a summary)
    bool truncated_last = false;     // the last message had to be cut from the front
    bool summary_cached = false;
};

struct SpecInfo {                    // speculative decoding (greedy: output identical to plain greedy; sampled: same distribution as plain sampling)
    std::string mode;                // "": not used, "flash" (trained drafter + prompt lookup), "ngram" (prompt lookup)
    std::string note;                // why Flash was not used when it was asked for
    int stood_down_at = -1;          // model pass at which Flash stood down for the conversation (-1: it did not)
    int verify_steps = 0, drafted = 0, accepted = 0;
    int passes = 0;                  // model forward passes during decode (tokens per pass = decoded / passes)
};

struct GenResult {
    std::string text;
    std::string finish_reason;       // "stop" (end of turn / stop string), "length" (max_tokens or context full), "cancelled"
    Usage usage;
    Timings timings;
    ContextInfo context;
    SpecInfo spec;
};

struct ModelInfo {
    std::string id, family, path, mmproj, quant;
    size_t file_bytes = 0, mmproj_bytes = 0;
    int layers = 0, layers_gpu = 0, hidden = 0, vocab = 0;
    bool head_on_cpu = false;
    size_t context_capacity = 0, native_context = 0;
    std::string kv_format;           // K/V cache format in use
    double load_seconds = 0;
    bool vision = false;
    int image_max_tokens = 0;
    size_t weights_vram_bytes = 0;
    size_t image_scratch_bytes = 0;  // device memory encoding one image at the largest size takes (borrowed from the unused context, else free VRAM)
    size_t vision_resident_bytes = 0;   // VRAM the image encoder keeps between images
    bool vision_streamed = false;       // the encoder streams its weights from the mmproj file
    std::string flash;               // drafter file in use ("" = none)
    std::string flash_note;          // why no drafter is loaded
    int spec_k = 0;
    std::string gpu;
    std::string placement;
};

// Running totals and the latest request, safe to read while a request runs.
struct EngineStats {
    uint64_t requests = 0, prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0, compactions = 0, errors = 0;
    double prefill_ms = 0, decode_ms = 0;   // sums (for averages)
    int prefilled = 0, decoded = 0;
    bool busy = false;
    int live_tokens = 0;                    // tokens generated so far by the running request
    double live_tps = 0;
    GenResult last;
    bool has_last = false;
};

struct EngineOptions {
    std::string gguf, mmproj;               // mmproj optional (image input)
    std::string id;                         // model id reported by the API (default: the file name)
    size_t kv = 0;                          // 0: auto
    size_t min_context = 32768;
    size_t assume_free_bytes = 0;           // dry runs (ModelPlanner): treat this much VRAM as free instead of asking the driver (0 = ask)
    size_t vram_reserve_mb = 0;
    std::string kv_format = "q8";              // K/V cache: q8 (int8), k8v4, q4, tq4, k16v8             // VRAM the planner leaves free (0: the model's default; DARA_VRAM_SAFETY_MB overrides)
    int image_max_tokens = 400;             // Qwen image token cap (Gemma: at most 280 by the model)
    int compact_reserve = 256;              // tokens kept free after max_tokens before compacting
    std::string flash = "auto";             // drafter file, "auto" (the bundled drafter for this model, if found) or "" (off)
    int spec_k = 4;                         // draft tokens per verify step
    float flash_min_p = 0.6f;
    std::string flash_mode = "auto";        // "auto": keep the drafter only if the model stays fully in VRAM with it; "on": always; "off": never
    bool spec = true;                       // speculative decoding (prompt lookup + Flash) for requests that do not turn it off
};

class Engine {
public:
    explicit Engine(const EngineOptions& opt);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // on_text receives the generated text as it streams (complete UTF-8 characters, stop strings held back); returning false cancels.
    using TextFn = std::function<bool(const std::string&)>;
    GenResult chat(const std::vector<Message>& messages, const SampleParams& sp, const TextFn& on_text);
    GenResult complete(const std::string& prompt, const SampleParams& sp, const TextFn& on_text);   // raw prompt (no chat template)
    // Context maintenance on its own: the messages as they would be sent to the model for a reply of max_tokens.
    std::vector<Message> compact(const std::vector<Message>& messages, int max_tokens, ContextInfo* info, double* ms);
    int count_tokens(const std::vector<Message>& messages);

    const ModelInfo& info() const;
    EngineStats stats() const;   // copy, thread-safe

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// What loading a configuration would do on this GPU, from the real placement planner and without loading anything.
struct PlanResult {
    bool ok = false;
    std::string error;               // why it cannot be placed
    size_t context = 0;              // positions the K/V caches hold (the auto choice when the query asked for 0)
    size_t native_context = 0;
    int layers = 0, layers_gpu = 0;
    bool head_on_cpu = false;        // the output head stays in host RAM
    bool flash_active = false;       // the Flash drafter would be loaded (it is dropped when its VRAM would push layers out)
    std::string flash_note;
    std::string placement;           // the planner's description
    size_t vram_total = 0, vram_free_assumed = 0;
    size_t state_bytes = 0;          // K/V caches + recurrent state
};

// Parses a model's GGUF once, then answers plan() queries. A query is an EngineOptions: kv_format, kv (context, 0 = auto), min_context, flash / flash_mode /
// spec, mmproj (reserves the image encoder's resident part), vram_reserve_mb and assume_free_bytes (VRAM to treat as free; 0 = ask the driver).
class ModelPlanner {
public:
    explicit ModelPlanner(const std::string& gguf);
    ~ModelPlanner();
    ModelPlanner(const ModelPlanner&) = delete;
    ModelPlanner& operator=(const ModelPlanner&) = delete;
    PlanResult plan(const EngineOptions& query);
    const std::string& family() const;   // "qwen35" or "gemma4"
    int layers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace dara::serve
