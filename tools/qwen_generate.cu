// Dara full-model driver (GGUF only): prefill a prompt, generate greedily, and dump everything needed to compare against
// the reference: the token sequence, logits at EVERY position, per-layer outputs for the prefill, and memory / timing
// statistics. Also checks model-level checkpoint + restore.
//
// Usage: qwen_generate.exe --gguf <model.gguf> --tokens <file of ids> --new N --out <prefix> [--kv 256] [--no-restore-test]
// Output files: <prefix>.tokens.txt (prompt + generated ids), <prefix>.logits.bin (float32 [P][vocab] for positions 0..P-1
// where P = prompt + N - 1), <prefix>.layers.bin (float32 [layers][prompt][hidden]), <prefix>.stats.json
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/system.h"
#include "io/gguf_checkpoint.h"
#include "model/flash_drafter.h"
#include "io/image.h"
#include "model/qwen_model.h"
#include "model/qwen_vision.h"
#include "tokenizer/tokenizer.h"

using namespace dara::model;

static int argmax(const float* v, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i) if (v[i] > v[b]) b = i;
    return b;
}

int main(int argc, char** argv) {
    std::string gguf_path, tokens_file, prompt_text, prompt_file, out = "build/run";
    int n_new = 8;
    size_t kv = 256;
    bool restore_test = true, parse_special = false, stop_at_eos = false;
    int chunk_tokens = 0;
    KvFormat kv_format = KvFormat::Q8;   // default INT8 K/V (near-lossless; --kv-mode f32 for the reference path)  // --kv-mode f32|tq4
    int logits_last = 0;  // --logits-last N: logits only for the last N prompt positions (long-context quality runs), no layer dump
    bool bench = false;  // --bench: throughput measurement mode (last-position logits only, no layer dumps, no logits file, no restore test)
    std::string teacher_file;
    std::string chat_user, mmproj_path;      // --chat <text>: Qwen chat template (no thinking); --mmproj <vision GGUF> + --image <file> (repeatable)
    std::vector<std::string> image_paths;
    int image_max_tokens = 1024;             // --image-max-tokens N: images are resized to at most N image tokens (N * 32 * 32 pixels)
    int spec_k = 0, ngram_max = 4, ngram_min = 3;  // --spec K: prompt-lookup speculative decoding with up to K draft tokens
    bool flash_ngram = true;    // --flash-no-ngram: Flash drafts only (default: prompt-lookup drafts first when the context repeats)
    float flash_min_p = 0.0f;   // --flash-min-p P: draft only while the drafter's cumulative probability stays >= P
    std::string flash_path;   // --flash <drafter.bin>: Dara Flash speculative decoding (trained drafter), up to --spec K draft tokens (default 4)
    int force_resident = -1;  // -1: the planner decides how many layers live in VRAM
    size_t min_context = 32768;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--gguf") gguf_path = next();
        else if (a == "--tokens") tokens_file = next();
        else if (a == "--prompt") prompt_text = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--chat") chat_user = next();
        else if (a == "--mmproj") mmproj_path = next();
        else if (a == "--image") image_paths.push_back(next());
        else if (a == "--image-max-tokens") image_max_tokens = std::atoi(next().c_str());
        else if (a == "--special") parse_special = true;
        else if (a == "--stop-at-eos") stop_at_eos = true;
        else if (a == "--resident") force_resident = std::atoi(next().c_str());
        else if (a == "--teacher") teacher_file = next();   // teacher forcing: feed the continuation of this reference tokens file instead of our own argmax
        else if (a == "--spec") spec_k = std::atoi(next().c_str());
        else if (a == "--flash") flash_path = next();
        else if (a == "--flash-no-ngram") flash_ngram = false;
        else if (a == "--flash-min-p") flash_min_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--ngram") { ngram_max = std::atoi(next().c_str()); }
        else if (a == "--ngram-min") { ngram_min = std::atoi(next().c_str()); }
        else if (a == "--new") n_new = std::atoi(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--kv") { const std::string v = next(); kv = v == "auto" ? 0 : static_cast<size_t>(std::atoll(v.c_str())); }   // auto (0): the largest context that keeps the model in VRAM
        else if (a == "--no-restore-test") restore_test = false;
        else if (a == "--bench") { bench = true; restore_test = false; }
        else if (a == "--logits-last") { logits_last = std::atoi(next().c_str()); restore_test = false; }
        else if (a == "--chunk") chunk_tokens = std::atoi(next().c_str());   // prefill chunk size (default: the model's, 64)
        else if (a == "--min-context") min_context = static_cast<size_t>(std::atoll(next().c_str()));   // auto context floor (default 32768; 0 = fully-resident size)
        else if (a == "--kv-mode") { const std::string m = next(); if (m == "tq4") kv_format = KvFormat::TQ4; else if (m == "q8") kv_format = KvFormat::Q8; else if (m == "k16v8") kv_format = KvFormat::K16V8; else if (m == "f32") kv_format = KvFormat::F32; else if (m == "k8v4") kv_format = KvFormat::K8V4; else if (m == "q4") kv_format = KvFormat::Q4; else { std::fprintf(stderr, "--kv-mode must be q8, k8v4, q4, tq4, k16v8 or f32\n"); return 2; } }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const bool chat_mode = !chat_user.empty();
    const bool text_mode = !prompt_text.empty() || !prompt_file.empty() || chat_mode;
    if (gguf_path.empty() || (tokens_file.empty() && !text_mode)) {
        std::fprintf(stderr, "usage: qwen_generate --gguf <model.gguf> (--prompt <text> | --prompt-file <utf8 file> | --tokens <ids file>) [--special] [--stop-at-eos] --new N --out <prefix>\n");
        return 2;
    }

    try {
        // GGUF is the only supported model format: weights, configuration AND tokenizer all come from the .gguf file.
        dara::core::memtrace("start");
        dara::io::GgufCheckpoint gguf(gguf_path);
        dara::core::memtrace("gguf header parsed");
        const QwenConfig cfg = QwenConfig::from_gguf(gguf);
        std::unique_ptr<dara::tokenizer::Tokenizer> tokenizer;
        try { tokenizer = std::make_unique<dara::tokenizer::Tokenizer>(dara::tokenizer::Tokenizer::from_gguf(gguf.header())); }
        catch (const std::exception& e) { if (text_mode) throw; std::fprintf(stderr, "note: no text output (%s)\n", e.what()); }

        std::vector<int32_t> prompt;
        struct EncodedImage { int mh, mw; std::vector<float> rows; };
        std::vector<EncodedImage> images;
        if (!image_paths.empty()) {   // encode before the text model loads: the encoder's VRAM is free again when the text model sizes its context
            if (!chat_mode) throw std::runtime_error("--image needs --chat <text> (the image goes into the user turn)");
            if (mmproj_path.empty()) throw std::runtime_error("--image needs --mmproj <vision GGUF>");
            dara::io::GgufCheckpoint mm(mmproj_path);
            const auto tv0 = std::chrono::steady_clock::now();
            dara::model::QwenVision vision(mm, image_max_tokens);
            const double vload = std::chrono::duration<double>(std::chrono::steady_clock::now() - tv0).count();
            if (vision.out_dim() != cfg.hidden) throw std::runtime_error("the mmproj does not match this model (projection width differs)");
            for (const std::string& ip : image_paths) {
                const auto tp0 = std::chrono::steady_clock::now();
                const auto img = dara::io::load_image_rgb(ip);
                const auto in = vision.preprocess(img);
                const double prep = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
                images.push_back({in.merged_h(), in.merged_w(), vision.encode(in)});
                std::printf("image %s: %dx%d -> %dx%d px, %dx%d patches -> %d tokens, preprocessed in %.3f s, encoded in %.3f s (encoder load %.1f s)\n",
                            ip.c_str(), img.width, img.height, in.grid_w * 16, in.grid_h * 16, in.grid_w, in.grid_h, in.tokens(), prep, vision.last_seconds(), vload);
            }
        }
        if (chat_mode) {   // Qwen3.5 chat template, thinking off; images (if any) before the text: <|vision_start|> n x <|image_pad|> <|vision_end|>
            std::string content;
            for (const auto& im : images) {
                content += "<|vision_start|>";
                for (int i = 0; i < im.mh * im.mw; ++i) content += "<|image_pad|>";
                content += "<|vision_end|>";
            }
            content += chat_user;
            prompt_text = "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            prompt = tokenizer->encode(prompt_text, /*parse_special=*/true);
            std::printf("chat prompt%s -> %zu tokens\n", images.empty() ? "" : " with image(s)", prompt.size());
        } else if (text_mode) {
            if (!prompt_file.empty()) { std::ifstream f(prompt_file, std::ios::binary); prompt_text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }
            prompt = tokenizer->encode(prompt_text, parse_special);
            std::printf("prompt text (%zu bytes) -> %zu tokens by Dara's own tokenizer:", prompt_text.size(), prompt.size());
            for (int32_t t : prompt) std::printf(" %d", t);
            std::printf("\n");
        } else {
            std::ifstream f(tokens_file);
            int t;
            while (f >> t) prompt.push_back(t);
        }
        if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }
        std::unique_ptr<IModelWeights> weights = std::make_unique<GgufWeights>(gguf, cfg);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);
        std::printf("%s | weights: %s | layers %d, hidden %d, vocab %d | prompt %zu tokens, generate %d\n", prop.name, weights->format(), cfg.layers,
                    cfg.hidden, cfg.vocab, prompt.size(), n_new);
        std::vector<int32_t> teacher;   // --teacher: forced continuation (the reference run's tokens after the prompt)
        if (!teacher_file.empty()) {
            std::ifstream f(teacher_file);
            std::vector<int32_t> all; int v;
            while (f >> v) all.push_back(v);
            if (all.size() <= prompt.size() || !std::equal(prompt.begin(), prompt.end(), all.begin())) throw std::runtime_error("--teacher file does not start with this prompt");
            teacher.assign(all.begin() + static_cast<std::ptrdiff_t>(prompt.size()), all.end());
            n_new = static_cast<int>(teacher.size());
            stop_at_eos = false;
        }
        ResidencyOptions ropt;
        ropt.force_resident_layers = force_resident;
        ropt.kv_format = kv_format;
        ropt.auto_min_context = min_context;
        if (chunk_tokens > 0) ropt.max_chunk_tokens = chunk_tokens;
        if (!flash_path.empty()) {   // leave room for the drafter (loaded after the model): its weights and its K/V per context position
            FlashDrafter::vram_estimate(flash_path, gguf, ropt.extra_vram_bytes, ropt.extra_bytes_per_token);
        }
        gguf.release_string_arrays();   // the tokenizer holds its own copy of the vocabulary and merges
        dara::core::memtrace("tokenizer + prompt ready");
        QwenModel model(cfg, *weights, kv, ropt);  // cold load: disk -> host images -> VRAM slabs, once
        dara::core::memtrace("model loaded");
        if (kv == 0) std::printf("context: %zu tokens (auto: the most that keeps the model in VRAM)\n", model.kv_capacity());
        kv = model.kv_capacity();
        if (!images.empty()) {   // each run of <|image_pad|> takes one image's tokens (embeddings + 3D rotary positions)
            const int32_t pad_id = tokenizer->encode("<|image_pad|>", true).at(0);
            size_t k = 0;
            for (size_t p = 0; p < prompt.size() && k < images.size();) {
                if (prompt[p] != pad_id) { ++p; continue; }
                const size_t n = static_cast<size_t>(images[k].mh) * images[k].mw;
                for (size_t j = 0; j < n; ++j)
                    if (p + j >= prompt.size() || prompt[p + j] != pad_id) throw std::runtime_error("image placeholder run shorter than the image's tokens");
                model.add_image(p, images[k].mh, images[k].mw, std::move(images[k].rows));
                p += n;
                ++k;
            }
            if (k != images.size()) throw std::runtime_error("image placeholders missing from the prompt");
        }
        {
            const PlacementReport& pr = model.placement();
            std::printf("GPU: %s\n", pr.gpu.c_str());
            std::printf("cold load: %.1f s | placement: %s\n", model.load_seconds(), pr.plan.c_str());
            std::printf("  VRAM-resident layers %d (%.0f MiB), host-resident layers %d (%.0f MiB, %s), output head %.0f MiB in VRAM, embedding table %.0f MiB in host RAM\n",
                        pr.resident_layers, pr.resident_weight_bytes / 1048576.0, pr.hosted_layers, pr.hosted_weight_bytes / 1048576.0,
                        pr.hosted_layers ? (pr.hosted_pinned ? "pinned" : "pageable") : "n/a", pr.head_bytes / 1048576.0, pr.embedding_host_bytes / 1048576.0);
            std::printf("  VRAM free at start %.0f of %.0f MiB; state+KV %.0f MiB, workspace %.0f MiB, staging %.0f MiB\n", pr.vram_free_at_start / 1048576.0,
                        pr.vram_total / 1048576.0, pr.state_bytes / 1048576.0, pr.workspace_bytes / 1048576.0, pr.staging_bytes / 1048576.0);
        }

        const int T0 = static_cast<int>(prompt.size()), V = cfg.vocab, H = cfg.hidden;
        std::vector<float> all_logits;  // [positions][V]
        std::vector<int32_t> seq = prompt;
        std::vector<double> step_seconds;
        RunStats st;
        size_t peak_device = 0, min_free = ~size_t(0), max_layer = model.placement().head_bytes;
        RunStats decode_sum;  // warm decode only (prefill excluded)
        int decode_steps = 0;
        auto note = [&](const RunStats& s) { peak_device = std::max(peak_device, s.device_peak_bytes); min_free = std::min(min_free, s.device_free_min_bytes); };

        // ---- prefill: logits at every prompt position, plus every layer's output ----------------
        std::vector<std::vector<float>> layer_out;
        std::vector<float> prompt_feats;   // --flash: the target's final normed states of the prompt (the drafter's input)
        std::vector<float> lg = model.forward(prompt.data(), T0, bench ? T0 - 1 : (logits_last > 0 ? std::max(0, T0 - logits_last) : 0), (bench || logits_last > 0) ? nullptr : &layer_out, &st,
                                              nullptr, flash_path.empty() ? nullptr : &prompt_feats, 0);
        note(st); step_seconds.push_back(st.seconds);
        all_logits.insert(all_logits.end(), lg.begin(), lg.end());
        std::printf("prefill %d tokens: %.3f s\n", T0, st.seconds);
        if (!bench && logits_last == 0) { std::ofstream f(out + ".layers.bin", std::ios::binary); for (auto& l : layer_out) f.write(reinterpret_cast<const char*>(l.data()), static_cast<std::streamsize>(l.size() * 4)); }
        const QwenModel::Checkpoint after_prefill = model.checkpoint();

        // ---- greedy decode ---------------------------------------------------------------------
        // Generated text streams through the incremental detokenizer: only complete UTF-8 characters are ever printed.
        std::unique_ptr<dara::tokenizer::Tokenizer::StreamDecoder> stream;
        if (tokenizer) stream = std::make_unique<dara::tokenizer::Tokenizer::StreamDecoder>(*tokenizer);
        std::string generated_text;
        auto emit = [&](int32_t id) {
            if (!stream) return;
            const std::string chunk = stream->push(id);
            generated_text += chunk;
            if (text_mode && !chunk.empty()) { std::fwrite(chunk.data(), 1, chunk.size(), stdout); std::fflush(stdout); }
        };
        if (text_mode) std::printf("--- generated ---\n");
        int tok = argmax(&all_logits[all_logits.size() - static_cast<size_t>(V)], V);  // the last prompt position's row
        if (!teacher.empty()) { if (spec_k) throw std::runtime_error("--teacher needs plain decoding (no --spec)"); tok = teacher[0]; }
        seq.push_back(tok);
        emit(tok);
        bool hit_eos = stop_at_eos && tokenizer && tok == tokenizer->eos_id();
        std::unique_ptr<FlashDrafter> flash;   // loaded once, like the model (not part of the decode time)
        if (!flash_path.empty()) {
            const auto tl = std::chrono::steady_clock::now();
            flash = std::make_unique<FlashDrafter>(flash_path, gguf, [&](int32_t t) { return weights->embed_row(t); }, kv + 16);
            std::printf("flash drafter load: %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
        }
        const auto t_dec0 = std::chrono::steady_clock::now();
        long passes = 0, spec_steps = 0, drafted = 0, accepted = 0, rollbacks = 0;  // one pass = one read of all model weights
        if (!flash_path.empty()) {
            if (!teacher.empty()) throw std::runtime_error("--teacher needs plain decoding (no --flash)");
            if (spec_k <= 0) spec_k = 4;
            if (spec_k > 7) spec_k = 7;
            restore_test = false;
            const auto tf0 = std::chrono::steady_clock::now();
            std::vector<int32_t> nxt(prompt.begin() + 1, prompt.end());
            nxt.push_back(tok);
            std::vector<float> y; int32_t d1 = 0;
            flash->extend(prompt_feats.data(), nxt.data(), T0, y, d1);
            float p1 = flash->last_prob();
            std::printf("flash drafter: prompt pass %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count());
            std::vector<int32_t> draft, feed, ids;
            std::vector<float> feats;
            auto ngram_lookup = [&](std::vector<int32_t>& out, int limit) {   // prompt lookup (copy-heavy text: edits, transforms)
                out.clear();
                const int n_sz = static_cast<int>(seq.size());
                for (int n = ngram_max; n >= ngram_min && out.empty(); --n) {
                    if (n_sz <= n) continue;
                    for (int j = n_sz - n - 1; j >= 0; --j) {
                        bool eq = true;
                        for (int q = 0; q < n && eq; ++q) eq = seq[j + q] == seq[n_sz - n + q];
                        if (!eq) continue;
                        for (int q = j + n; q < n_sz && static_cast<int>(out.size()) < limit; ++q) out.push_back(seq[q]);
                        if (static_cast<int>(out.size()) >= std::min(2, limit)) break; out.clear();   // too close to the end to copy enough: look further back
                    }
                }
            };
            int ng_cool = 0, ng_back = 0, ng_len = 4;   // adaptive prompt-lookup draft length (long failed drafts are expensive verifies)
            long ng_steps = 0, ng_acc = 0;
            double fp_draft = 0, fp_verify = 0, fp_roll = 0, fp_ext = 0, fp_head = 0, fp_comp = 0, fp_emb = 0;
            while (static_cast<int>(seq.size() - prompt.size()) < n_new && !hit_eos) {
                const int budget = n_new - static_cast<int>(seq.size() - prompt.size()) - 1;
                const int32_t cur = seq.back();
                const size_t dpos = flash->position();
                draft.clear();
                auto ft0 = std::chrono::steady_clock::now();
                bool from_ngram = false;
                if (flash_ngram && ng_cool == 0) { ngram_lookup(draft, std::min(ng_len, budget)); from_ngram = draft.size() >= 2; if (!from_ngram) draft.clear(); }
                else if (ng_cool > 0) --ng_cool;
                if (!from_ngram) flash->draft(y, d1, p1, std::min(spec_k, budget), flash_min_p, draft);
                auto ft1 = std::chrono::steady_clock::now(); fp_draft += std::chrono::duration<double>(ft1 - ft0).count();
                const int kd = static_cast<int>(draft.size());
                feed.assign(1, cur); feed.insert(feed.end(), draft.begin(), draft.end());
                if (kd > 0) model.snapshot();
                model.forward(feed.data(), kd + 1, 0, nullptr, &st, &ids, &feats, 0); note(st); ++passes;
                auto ft2 = std::chrono::steady_clock::now(); fp_verify += std::chrono::duration<double>(ft2 - ft1).count(); fp_head += st.head_seconds; fp_comp += st.compute_seconds; fp_emb += st.embed_seconds;
                int m = 0;
                while (m < kd && ids[m] == draft[m]) ++m;
                if (from_ngram) { ++ng_steps; ng_acc += m; ng_len = m == kd ? std::min(ng_len * 2, 7) : std::max(2, m + 1); if (m == 0) { ng_back = ng_back ? std::min(ng_back * 2, 32) : 2; ng_cool = ng_back; } else ng_back = 0; }
                if (kd > 0) { ++spec_steps; drafted += kd; accepted += m; }
                if (m < kd) { model.rollback(m + 1); ++rollbacks; }
                std::vector<int32_t> to_add(draft.begin(), draft.begin() + m);
                to_add.push_back(ids[m]);
                flash->rewind(dpos);   // drop the drafter's own guesses; re-feed it the target's true states of the accepted positions
                auto ft3 = std::chrono::steady_clock::now(); fp_roll += std::chrono::duration<double>(ft3 - ft2).count();
                flash->extend(feats.data(), to_add.data(), m + 1, y, d1);
                p1 = flash->last_prob();
                fp_ext += std::chrono::duration<double>(std::chrono::steady_clock::now() - ft3).count();
                for (int32_t id : to_add) {
                    seq.push_back(id); emit(id);
                    if (stop_at_eos && tokenizer && id == tokenizer->eos_id()) { hit_eos = true; break; }
                }
            }
            std::printf("flash time: draft %.3f s, verify %.3f s, accept/rollback %.3f s, drafter extend %.3f s (verify: layers %.3f s, head %.3f s, embed %.3f s)\n", fp_draft, fp_verify, fp_roll, fp_ext, fp_comp, fp_head, fp_emb);
            if (flash_ngram) std::printf("flash n-gram: %ld steps, %ld accepted\n", ng_steps, ng_acc);
            spec_k = std::max(spec_k, 1);   // the plain loop below must not run
        } else if (spec_k > 0) {
            // Speculative decoding, greedy-exact: prompt lookup proposes up to spec_k tokens; one multi-token forward scores them
            // all; the longest prefix the model itself would have produced is kept (rolling the state back if any was rejected).
            restore_test = false;
            auto lookup = [&](std::vector<int32_t>& draft, int limit) {
                draft.clear();
                const int n_sz = static_cast<int>(seq.size());
                for (int n = ngram_max; n >= ngram_min && draft.empty(); --n) {
                    if (n_sz <= n) continue;
                    for (int j = n_sz - n - 1; j >= 0; --j) {
                        bool eq = true;
                        for (int q = 0; q < n && eq; ++q) eq = seq[j + q] == seq[n_sz - n + q];
                        if (!eq) continue;
                        for (int q = j + n; q < n_sz && static_cast<int>(draft.size()) < limit; ++q) draft.push_back(seq[q]);
                        if (static_cast<int>(draft.size()) >= std::min(2, limit)) break; draft.clear();   // too close to the end to copy enough: look further back
                    }
                }
            };
            std::vector<int32_t> draft, feed;
            int cooldown = 0, backoff = 0;
            if (spec_k > 7) { std::printf("note: --spec limited to 7 draft tokens (a verify step of 8 tokens is the multi-vector kernel's group size)\n"); spec_k = 7; }
            while (static_cast<int>(seq.size() - prompt.size()) < n_new && !hit_eos) {
                const int budget = n_new - static_cast<int>(seq.size() - prompt.size()) - 1;  // tokens that may still be added after `cur`
                if (cooldown > 0) { --cooldown; draft.clear(); }  // back off after steps where no draft token was accepted
                else lookup(draft, std::min(spec_k, budget));
                const int32_t cur = seq.back();
                std::vector<int32_t> to_add;
                if (draft.empty()) {
                    std::vector<int32_t> ids;
                    model.forward(&cur, 1, 0, nullptr, &st, &ids); note(st); ++passes;
                    to_add.push_back(ids[0]);
                } else {
                    const int kd = static_cast<int>(draft.size());
                    feed.assign(1, cur); feed.insert(feed.end(), draft.begin(), draft.end());
                    model.snapshot();
                    std::vector<int32_t> ids;   // greedy token after each fed position, chosen on the GPU
                    model.forward(feed.data(), kd + 1, 0, nullptr, &st, &ids); note(st); ++passes; ++spec_steps; drafted += kd;
                    int m = 0;
                    while (m < kd && ids[m] == draft[m]) ++m;
                    accepted += m;
                    if (m == 0) { backoff = backoff ? std::min(backoff * 2, 32) : 2; cooldown = backoff; } else backoff = 0;   // a failed check costs several steps: back off hard
                    if (m < kd) {  // a candidate was rejected: rebuild the state after the accepted prefix (cur + m drafts), no weight pass
                        model.rollback(m + 1); ++rollbacks;
                    }
                    for (int q = 0; q < m; ++q) to_add.push_back(draft[q]);
                    to_add.push_back(ids[m]);
                }
                for (int32_t id : to_add) {
                    seq.push_back(id); emit(id);
                    if (stop_at_eos && tokenizer && id == tokenizer->eos_id()) { hit_eos = true; break; }
                }
            }
        }
        for (int k = 1; spec_k == 0 && k < n_new && !hit_eos; ++k) {
            ++passes;
            const int32_t feed = tok;
            std::vector<float> row = model.forward(&feed, 1, 0, nullptr, &st);
            note(st); step_seconds.push_back(st.seconds);
            decode_sum.seconds += st.seconds; decode_sum.embed_seconds += st.embed_seconds; decode_sum.compute_seconds += st.compute_seconds;
            decode_sum.copy_wait_seconds += st.copy_wait_seconds; decode_sum.head_seconds += st.head_seconds; decode_sum.streamed_bytes += st.streamed_bytes;
            ++decode_steps;
            if (!bench) all_logits.insert(all_logits.end(), row.begin(), row.end());
            tok = argmax(row.data(), V);
            if (!teacher.empty()) tok = teacher[static_cast<size_t>(k)];
            seq.push_back(tok);
            emit(tok);
            if (!text_mode) std::printf("decode step %d: fed %d -> next %d (%.1f s)\n", k, feed, tok, st.seconds);
            hit_eos = stop_at_eos && tokenizer && tok == tokenizer->eos_id();
        }
        const double dec_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count();
        const long dec_tokens = static_cast<long>(seq.size() - prompt.size()) - 1;  // the first token came from the prefill
        if (stream) { const std::string tail = stream->flush(); generated_text += tail; if (text_mode && !tail.empty()) std::fwrite(tail.data(), 1, tail.size(), stdout); }
        if (text_mode) std::printf("\n--- end (%s) ---\n", hit_eos ? "stopped at end-of-sequence" : "token limit");
        dara::core::memtrace("after generation");
        n_new = static_cast<int>(seq.size() - prompt.size());  // tokens actually generated
        if (tokenizer) { std::ofstream f(out + ".text.txt", std::ios::binary); f << prompt_text << generated_text; }

        // ---- model-level checkpoint / restore --------------------------------------------------
        bool restore_ok = true;
        int restore_steps = std::min(n_new - 1, 3);
        if (restore_test && restore_steps > 0) {
            model.restore(after_prefill);
            for (int k = 0; k < restore_steps; ++k) {
                const int32_t feed = seq[T0 + k];
                std::vector<float> row = model.forward(&feed, 1, 0);
                const float* want = &all_logits[static_cast<size_t>(T0 + k) * V];
                restore_ok &= std::memcmp(row.data(), want, static_cast<size_t>(V) * 4) == 0;
            }
            std::printf("checkpoint after prefill, restore, re-run %d decode steps: logits %s\n", restore_steps, restore_ok ? "bitwise identical" : "DIFFER");
        }

        // ---- outputs ---------------------------------------------------------------------------
        { std::ofstream f(out + ".tokens.txt"); for (size_t i = 0; i < seq.size(); ++i) f << seq[i] << (i + 1 < seq.size() ? " " : "\n"); }
        if (!bench) { std::ofstream f(out + ".logits.bin", std::ios::binary); f.write(reinterpret_cast<const char*>(all_logits.data()), static_cast<std::streamsize>(all_logits.size() * 4)); }
        const auto sys = dara::core::query_system_memory();
        double total_s = 0; for (double s : step_seconds) total_s += s;
        {
            std::ofstream f(out + ".stats.json");
            f << "{\n  \"gpu\": \"" << prop.name << "\",\n  \"prompt_tokens\": " << T0 << ",\n  \"generated\": " << n_new
              << ",\n  \"positions_with_logits\": " << (all_logits.size() / V)
              << ",\n  \"restore_ok\": " << (restore_ok ? "true" : "false")
              << ",\n  \"device_peak_counted_bytes\": " << peak_device
              << ",\n  \"device_free_min_bytes\": " << min_free
              << ",\n  \"output_head_bytes\": " << max_layer
              << ",\n  \"cold_load_seconds\": " << model.load_seconds()
              << ",\n  \"decode_steps\": " << decode_steps
              << ",\n  \"decode_seconds\": " << decode_sum.seconds
              << ",\n  \"decode_compute_seconds\": " << decode_sum.compute_seconds
              << ",\n  \"decode_copy_wait_seconds\": " << decode_sum.copy_wait_seconds
              << ",\n  \"decode_head_seconds\": " << decode_sum.head_seconds
              << ",\n  \"decode_embed_seconds\": " << decode_sum.embed_seconds
              << ",\n  \"decode_streamed_bytes\": " << decode_sum.streamed_bytes
              << ",\n  \"resident_layers\": " << model.placement().resident_layers
              << ",\n  \"hosted_layers\": " << model.placement().hosted_layers
              << ",\n  \"host_peak_working_set_bytes\": " << sys.process_peak_working_set_bytes
              << ",\n  \"total_pass_seconds\": " << total_s << "\n}\n";
        }
        std::printf("generated ids:"); for (size_t i = T0; i < seq.size(); ++i) std::printf(" %d", seq[i]); std::printf("\n");
        std::printf("memory: device peak counted %.0f MiB, lowest free VRAM %.0f MiB, host peak working set %.0f MiB\n",
                    peak_device / 1048576.0, min_free / 1048576.0, sys.process_peak_working_set_bytes / 1048576.0);
        if (dec_tokens > 0 && passes > 0)
            std::printf("decode wall clock: %ld tokens in %.3f s = %.2f tok/s | weight passes %ld => %.2f tokens per weight pass%s\n", dec_tokens, dec_wall,
                        dec_tokens / dec_wall, passes, static_cast<double>(dec_tokens) / passes, spec_k > 0 ? " (speculative)" : "");
        if (spec_k > 0 && spec_steps > 0)
            std::printf("speculation: %ld verify steps, %ld drafted, %ld accepted (%.0f%%, %.2f per step), %ld rollbacks\n", spec_steps, drafted, accepted,
                        drafted ? 100.0 * accepted / drafted : 0.0, static_cast<double>(accepted) / spec_steps, rollbacks);
        if (decode_steps > 0 && decode_sum.seconds > 0)
            std::printf("warm decode: %d steps, %.3f s/token = %.2f tok/s | compute %.3f s, copy-wait %.3f s, head %.3f s, embed %.3f s, streamed %.1f MiB/token\n",
                        decode_steps, decode_sum.seconds / decode_steps, decode_steps / decode_sum.seconds, decode_sum.compute_seconds / decode_steps,
                        decode_sum.copy_wait_seconds / decode_steps, decode_sum.head_seconds / decode_steps, decode_sum.embed_seconds / decode_steps,
                        decode_sum.streamed_bytes / 1048576.0 / decode_steps);
        (void)H;
        if (const char* lp = std::getenv("DARA_LAYER_PROF")) if (lp[0] == '1' || lp[0] == '2') std::printf("%s", layer_prof_report().c_str());
        return restore_ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
