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
#include "model/qwen_model.h"
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
    KvFormat kv_format = KvFormat::F32;  // --kv-mode f32|tq4
    int logits_last = 0;  // --logits-last N: logits only for the last N prompt positions (long-context quality runs), no layer dump
    bool bench = false;  // --bench: throughput measurement mode (last-position logits only, no layer dumps, no logits file, no restore test)
    std::string teacher_file;
    int spec_k = 0, ngram_max = 3, ngram_min = 2;  // --spec K: prompt-lookup speculative decoding with up to K draft tokens
    int force_resident = -1;  // -1: the planner decides how many layers live in VRAM
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--gguf") gguf_path = next();
        else if (a == "--tokens") tokens_file = next();
        else if (a == "--prompt") prompt_text = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--special") parse_special = true;
        else if (a == "--stop-at-eos") stop_at_eos = true;
        else if (a == "--resident") force_resident = std::atoi(next().c_str());
        else if (a == "--teacher") teacher_file = next();   // teacher forcing: feed the continuation of this reference tokens file instead of our own argmax
        else if (a == "--spec") spec_k = std::atoi(next().c_str());
        else if (a == "--ngram") { ngram_max = std::atoi(next().c_str()); }
        else if (a == "--ngram-min") { ngram_min = std::atoi(next().c_str()); }
        else if (a == "--new") n_new = std::atoi(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--kv") kv = static_cast<size_t>(std::atoll(next().c_str()));
        else if (a == "--no-restore-test") restore_test = false;
        else if (a == "--bench") { bench = true; restore_test = false; }
        else if (a == "--logits-last") { logits_last = std::atoi(next().c_str()); restore_test = false; }
        else if (a == "--chunk") chunk_tokens = std::atoi(next().c_str());   // prefill chunk size (default: the model's, 64)
        else if (a == "--kv-mode") { const std::string m = next(); if (m == "tq4") kv_format = KvFormat::TQ4; else if (m == "q8") kv_format = KvFormat::Q8; else if (m == "k16v8") kv_format = KvFormat::K16V8; else if (m == "f32") kv_format = KvFormat::F32; else { std::fprintf(stderr, "--kv-mode must be f32, q8 or tq4\n"); return 2; } }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const bool text_mode = !prompt_text.empty() || !prompt_file.empty();
    if (gguf_path.empty() || (tokens_file.empty() && !text_mode)) {
        std::fprintf(stderr, "usage: qwen_generate --gguf <model.gguf> (--prompt <text> | --prompt-file <utf8 file> | --tokens <ids file>) [--special] [--stop-at-eos] --new N --out <prefix>\n");
        return 2;
    }

    try {
        // GGUF is the only supported model format: weights, configuration AND tokenizer all come from the .gguf file.
        dara::io::GgufCheckpoint gguf(gguf_path);
        const QwenConfig cfg = QwenConfig::from_gguf(gguf);
        std::unique_ptr<dara::tokenizer::Tokenizer> tokenizer;
        try { tokenizer = std::make_unique<dara::tokenizer::Tokenizer>(dara::tokenizer::Tokenizer::from_gguf(gguf.header())); }
        catch (const std::exception& e) { if (text_mode) throw; std::fprintf(stderr, "note: no text output (%s)\n", e.what()); }

        std::vector<int32_t> prompt;
        if (text_mode) {
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
        if (chunk_tokens > 0) ropt.max_chunk_tokens = chunk_tokens;
        QwenModel model(cfg, *weights, kv, ropt);  // cold load: disk -> host images -> VRAM slabs, once
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
        std::vector<float> lg = model.forward(prompt.data(), T0, bench ? T0 - 1 : (logits_last > 0 ? std::max(0, T0 - logits_last) : 0), (bench || logits_last > 0) ? nullptr : &layer_out, &st);
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
        const auto t_dec0 = std::chrono::steady_clock::now();
        long passes = 0, spec_steps = 0, drafted = 0, accepted = 0, rollbacks = 0;  // one pass = one read of all model weights
        if (spec_k > 0) {
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
                        if (!draft.empty()) break;
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
                    std::vector<float> row = model.forward(&cur, 1, 0, nullptr, &st); note(st); ++passes;
                    to_add.push_back(argmax(row.data(), V));
                } else {
                    const int kd = static_cast<int>(draft.size());
                    feed.assign(1, cur); feed.insert(feed.end(), draft.begin(), draft.end());
                    model.snapshot();
                    std::vector<float> rows = model.forward(feed.data(), kd + 1, 0, nullptr, &st); note(st); ++passes; ++spec_steps; drafted += kd;
                    int m = 0;
                    while (m < kd && argmax(&rows[static_cast<size_t>(m) * V], V) == draft[m]) ++m;
                    accepted += m;
                    if (m == 0) { backoff = backoff ? std::min(backoff * 2, 8) : 1; cooldown = backoff; } else backoff = 0;
                    if (m < kd) {  // a candidate was rejected: rebuild the state after the accepted prefix (cur + m drafts), no weight pass
                        model.rollback(m + 1); ++rollbacks;
                    }
                    for (int q = 0; q < m; ++q) to_add.push_back(draft[q]);
                    to_add.push_back(argmax(&rows[static_cast<size_t>(m) * V], V));
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
