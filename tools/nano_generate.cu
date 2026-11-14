// Nano tier (Gemma 4 E2B) driver: GGUF in, text or token ids in, greedy text / token ids out. Mirrors tools/qwen_generate.cu so the same
// reference comparison (tools/compare_logits.py) applies. Text goes through Dara's own GGUF-native Gemma tokenizer (<bos> is prepended).
//
// Usage: nano_generate --gguf <model.gguf> (--prompt <text> | --prompt-file <utf8 file> | --tokens <ids file>) --new N --out <prefix>
//                      [--special] [--stop-at-eos] [--kv N] [--resident N] [--no-layers] [--logits-last N]
// Outputs: <prefix>.tokens.txt, <prefix>.logits.bin (float32 [positions][vocab]), <prefix>.layers.bin (prefill layer outputs),
//          <prefix>.text.txt (text mode). Prints cold-load time separately from warm decode tokens/s.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "core/system.h"
#include <stdexcept>

#include "io/gguf_checkpoint.h"
#include "model/gemma_model.h"
#include "tokenizer/gemma_chat.h"
#include "tokenizer/gemma_tokenizer.h"

using namespace dara::model;
using dara::tokenizer::GemmaTokenizer;

static int argmax(const float* p, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) if (p[i] > p[best]) best = i;
    return best;
}

int main(int argc, char** argv) {
    std::string gguf_path, tokens_file, prompt_text, prompt_file, out = "build/nano";
    int n_new = 8, force_resident = -1, logits_last = 0;  // logits_last: only the last N prompt positions get logits (0 = all)
    size_t kv = 2048;
    bool dump_layers = true, parse_special = false, stop_at_eos = false, think = false, profile = false, bench = false;
    int chunk_tokens = 0;
    KvFormat kv_format = KvFormat::Q8;   // default INT8 K/V (near-lossless; --kv-mode f32 for the reference path)  // --kv-mode f32|tq4
    std::string chat_user, chat_system, teacher_file;  // --chat renders the model's chat template around the user message
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--gguf") gguf_path = next();
        else if (a == "--tokens") tokens_file = next();
        else if (a == "--prompt") prompt_text = next();
        else if (a == "--prompt-file") prompt_file = next();
        else if (a == "--chat") chat_user = next();
        else if (a == "--system") chat_system = next();
        else if (a == "--think") think = true;
        else if (a == "--profile") profile = true;
        else if (a == "--special") parse_special = true;
        else if (a == "--stop-at-eos") stop_at_eos = true;
        else if (a == "--new") n_new = std::atoi(next().c_str());
        else if (a == "--out") out = next();
        else if (a == "--kv") kv = static_cast<size_t>(std::atoll(next().c_str()));
        else if (a == "--resident") force_resident = std::atoi(next().c_str());
        else if (a == "--no-layers") dump_layers = false;
        else if (a == "--bench") { bench = true; dump_layers = false; }   // throughput mode: decode logits are not copied to the host or saved
        else if (a == "--chunk") chunk_tokens = std::atoi(next().c_str());   // prefill chunk size (default: the model's, 64)
        else if (a == "--kv-mode") { const std::string m = next(); if (m == "tq4") kv_format = KvFormat::TQ4; else if (m == "q8") kv_format = KvFormat::Q8; else if (m == "k16v8") kv_format = KvFormat::K16V8; else if (m == "f32") kv_format = KvFormat::F32; else { std::fprintf(stderr, "--kv-mode must be f32, q8 or tq4\n"); return 2; } }
        else if (a == "--teacher") teacher_file = next();   // teacher forcing: feed the continuation of this reference tokens file instead of our own argmax
        else if (a == "--logits-last") logits_last = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    const bool chat_mode = !chat_user.empty();
    if (chat_mode) stop_at_eos = true;  // a chat answer ends at <turn|>
    const bool text_mode = chat_mode || !prompt_text.empty() || !prompt_file.empty();
    if (gguf_path.empty() || (tokens_file.empty() && !text_mode)) {
        std::fprintf(stderr, "usage: nano_generate --gguf <model.gguf> (--prompt <text> | --prompt-file <file> | --tokens <ids file>) --new N --out <prefix> "
                             "[--special] [--stop-at-eos] [--kv N] [--resident N] [--no-layers] [--logits-last N]\n");
        return 2;
    }
    try {
        dara::io::GgufCheckpoint gguf(gguf_path);
        std::unique_ptr<GemmaTokenizer> tokenizer;
        try { tokenizer = std::make_unique<GemmaTokenizer>(GemmaTokenizer::from_gguf(gguf.header())); }
        catch (const std::exception& e) { if (text_mode) throw; std::fprintf(stderr, "note: no text output (%s)\n", e.what()); }

        std::vector<int32_t> prompt;
        if (chat_mode) {
            std::vector<dara::tokenizer::ChatMessage> msgs;
            if (!chat_system.empty()) msgs.push_back({"system", chat_system});
            msgs.push_back({"user", chat_user});
            prompt_text = dara::tokenizer::render_gemma_chat(msgs, true, think);
            prompt = dara::tokenizer::tokenize_gemma_chat(*tokenizer, msgs, true, think);
            std::printf("chat prompt -> %zu tokens (template rendered by Dara, validated against the HF renderer)\n", prompt.size());
        } else if (text_mode) {
            if (!prompt_file.empty()) { std::ifstream f(prompt_file, std::ios::binary); prompt_text.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }
            prompt = tokenizer->encode(prompt_text, parse_special);
            std::printf("prompt text (%zu bytes) -> %zu tokens (with <bos>) by Dara's own tokenizer:", prompt_text.size(), prompt.size());
            for (size_t i = 0; i < prompt.size() && i < 40; ++i) std::printf(" %d", prompt[i]);
            std::printf(prompt.size() > 40 ? " ...\n" : "\n");
        } else {
            std::ifstream f(tokens_file);
            int t;
            while (f >> t) prompt.push_back(t);
        }
        if (prompt.empty()) { std::fprintf(stderr, "empty prompt\n"); return 2; }

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
        GemmaModel model(gguf, kv, ropt);
        const GemmaConfig& cfg = model.config();
        const PlacementReport& pr = model.placement();
        const int V = cfg.vocab, T0 = static_cast<int>(prompt.size());
        std::printf("GPU: %s\n", pr.gpu.c_str());
        std::printf("Nano (gemma4): layers %d, hidden %d, vocab %d, heads %d/%d kv, head_dim %d (sliding) / %d (global), window %d | prompt %d tokens, generate %d\n",
                    cfg.layers, cfg.hidden, V, cfg.heads, cfg.kv_heads, cfg.head_dim_swa, cfg.head_dim_full, cfg.window, T0, n_new);
        std::printf("cold load: %.1f s | placement: %s\n", model.load_seconds(), pr.plan.c_str());
        std::printf("  VRAM-resident layers %d (%.0f MiB), host-resident layers %d (%.0f MiB), output head %.0f MiB in VRAM, embedding tables %.0f MiB in host RAM\n",
                    pr.resident_layers, pr.resident_weight_bytes / 1048576.0, pr.hosted_layers, pr.hosted_weight_bytes / 1048576.0, pr.head_bytes / 1048576.0,
                    pr.embedding_host_bytes / 1048576.0);
        std::printf("  VRAM free at start %.0f of %.0f MiB; K/V %.0f MiB, workspace %.0f MiB, staging %.0f MiB\n", pr.vram_free_at_start / 1048576.0,
                    pr.vram_total / 1048576.0, pr.state_bytes / 1048576.0, pr.workspace_bytes / 1048576.0, pr.staging_bytes / 1048576.0);

        model.profile_enable(profile);  // --profile: per-operation GPU timing of every decode step
        std::vector<int32_t> seq = prompt;
        std::vector<float> all_logits;
        RunStats st;
        std::vector<std::vector<float>> layer_out;
        const int from = bench ? T0 - 1 : (logits_last > 0 ? std::max(0, T0 - logits_last) : 0);   // --bench: last position only (like the Qwen tool)
        std::vector<float> lg = model.forward(prompt.data(), T0, from, dump_layers ? &layer_out : nullptr, &st);
        all_logits.insert(all_logits.end(), lg.begin(), lg.end());
        std::printf("prefill %d tokens: %.2f s | embed (host) %.3f s, layers %.3f s, head %.3f s\n", T0, st.seconds, st.embed_seconds, st.compute_seconds,
                    st.head_seconds);
        if (dump_layers) {
            std::ofstream f(out + ".layers.bin", std::ios::binary);
            for (auto& l : layer_out) f.write(reinterpret_cast<const char*>(l.data()), static_cast<std::streamsize>(l.size() * 4));
        }

        std::unique_ptr<GemmaTokenizer::StreamDecoder> stream;
        if (tokenizer) stream = std::make_unique<GemmaTokenizer::StreamDecoder>(*tokenizer);
        std::string generated_text;
        auto emit = [&](int32_t id) {
            if (!stream) return;
            const std::string chunk = stream->push(id);
            generated_text += chunk;
            if (text_mode && !chunk.empty()) { std::fwrite(chunk.data(), 1, chunk.size(), stdout); std::fflush(stdout); }
        };
        auto is_eos = [&](int32_t id) { return stop_at_eos && tokenizer && (id == tokenizer->eos_id() || id == 1); };

        if (text_mode) std::printf("--- generated ---\n");
        int tok = argmax(&all_logits[all_logits.size() - static_cast<size_t>(V)], V);
        if (!teacher.empty()) tok = teacher[0];
        seq.push_back(tok);
        emit(tok);
        bool hit_eos = is_eos(tok);
        RunStats sum;
        int steps = 0;
        const auto t_dec0 = std::chrono::steady_clock::now();
        for (int k = 1; k < n_new && !hit_eos; ++k) {
            const int32_t feed = tok;
            std::vector<float> row;
            tok = model.forward_greedy(feed, &st, bench ? nullptr : &row);   // next token chosen on the GPU (same as the host argmax)
            sum.seconds += st.seconds; sum.compute_seconds += st.compute_seconds; sum.copy_wait_seconds += st.copy_wait_seconds;
            sum.head_seconds += st.head_seconds; sum.embed_seconds += st.embed_seconds; sum.streamed_bytes += st.streamed_bytes;
            ++steps;
            all_logits.insert(all_logits.end(), row.begin(), row.end());
            if (!bench && tok != argmax(row.data(), V)) throw std::runtime_error("GPU argmax disagrees with the host argmax");
            if (!teacher.empty()) tok = teacher[static_cast<size_t>(k)];
            seq.push_back(tok);
            emit(tok);
            hit_eos = is_eos(tok);
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count();
        if (stream) { const std::string tail = stream->flush(); generated_text += tail; if (text_mode && !tail.empty()) std::fwrite(tail.data(), 1, tail.size(), stdout); }
        if (text_mode) std::printf("\n--- end (%s) ---\n", hit_eos ? "stopped at end-of-sequence" : "token limit");

        if (tokenizer && text_mode) { std::ofstream f(out + ".text.txt", std::ios::binary); f << prompt_text << generated_text; }
        { std::ofstream f(out + ".tokens.txt"); for (size_t i = 0; i < seq.size(); ++i) f << seq[i] << (i + 1 < seq.size() ? " " : "\n"); }
        { std::ofstream f(out + ".logits.bin", std::ios::binary); f.write(reinterpret_cast<const char*>(all_logits.data()), static_cast<std::streamsize>(all_logits.size() * 4)); }
        std::printf("generated ids:"); for (size_t i = T0; i < seq.size(); ++i) std::printf(" %d", seq[i]); std::printf("\n");
        if (steps > 0)
            std::printf("warm decode: %d steps, %.4f s/token = %.2f tok/s (wall %.2f tok/s) | compute %.4f s, copy-wait %.4f s, head %.4f s, embed %.4f s, streamed %.1f MiB/token\n",
                        steps, sum.seconds / steps, steps / sum.seconds, steps / wall, sum.compute_seconds / steps, sum.copy_wait_seconds / steps,
                        sum.head_seconds / steps, sum.embed_seconds / steps, sum.streamed_bytes / 1048576.0 / steps);
        if (profile) std::printf("%s", model.profile_report().c_str());
        const auto sys = dara::core::query_system_memory();
        std::printf("memory: Dara's counted device peak %.0f MiB (all other VRAM users on this GPU excluded), lowest free VRAM seen %.0f MiB, host peak working set %.0f MiB\n",
                    st.device_peak_bytes / 1048576.0, st.device_free_min_bytes / 1048576.0, sys.process_peak_working_set_bytes / 1048576.0);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
