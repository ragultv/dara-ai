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
#include "model/flash_drafter.h"
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
    bool flash_ngram = true;    // --flash-no-ngram: Flash drafts only (default: prompt-lookup drafts first when the context repeats)
    float flash_min_p = 0.0f;   // --flash-min-p P: draft only while the drafter's cumulative probability stays >= P
    std::string flash_path;   // --flash <drafter.dflash>: Dara Flash speculative decoding, up to --spec K draft tokens (default 4)
    int spec_k = 0, ngram_max = 4, ngram_min = 3;   // --spec K: prompt-lookup speculative decoding (greedy-exact)
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
        else if (a == "--kv") { const std::string v = next(); kv = v == "auto" ? 0 : static_cast<size_t>(std::atoll(v.c_str())); }   // auto (0): the largest context that keeps the model in VRAM
        else if (a == "--resident") force_resident = std::atoi(next().c_str());
        else if (a == "--no-layers") dump_layers = false;
        else if (a == "--bench") { bench = true; dump_layers = false; }   // throughput mode: decode logits are not copied to the host or saved
        else if (a == "--spec") spec_k = std::atoi(next().c_str());
        else if (a == "--flash") flash_path = next();
        else if (a == "--flash-no-ngram") flash_ngram = false;
        else if (a == "--flash-min-p") flash_min_p = static_cast<float>(std::atof(next().c_str()));
        else if (a == "--ngram") ngram_max = std::atoi(next().c_str());
        else if (a == "--ngram-min") ngram_min = std::atoi(next().c_str());
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
        if (!flash_path.empty()) {   // leave room for the drafter (loaded after the model): its weights and its K/V per context position
            FlashDrafter::vram_estimate(flash_path, gguf, ropt.extra_vram_bytes, ropt.extra_bytes_per_token);
        }
        GemmaModel model(gguf, kv, ropt);
        if (kv == 0) std::printf("context: %zu tokens (auto: the most that keeps the model in VRAM)\n", model.kv_capacity());
        kv = model.kv_capacity();
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
        std::vector<float> prompt_feats;   // --flash: final normed states of the prompt (the drafter's input)
        std::vector<float> lg = model.forward(prompt.data(), T0, from, dump_layers ? &layer_out : nullptr, &st, flash_path.empty() ? nullptr : &prompt_feats);
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
        std::unique_ptr<FlashDrafter> flash;   // loaded once, like the model (not part of the decode time)
        if (!flash_path.empty()) {
            const auto tl = std::chrono::steady_clock::now();
            flash = std::make_unique<FlashDrafter>(flash_path, gguf, [&](int32_t t) { return model.embed_row_raw(t); }, kv + 16);
            std::printf("flash drafter load: %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count());
        }
        const auto t_dec0 = std::chrono::steady_clock::now();
        long spec_steps = 0, drafted = 0, accepted = 0, passes = 0;
        if (!flash_path.empty() && teacher.empty()) {
            if (spec_k <= 0) spec_k = 4;
            if (spec_k > 15) spec_k = 15;
            const auto tf0 = std::chrono::steady_clock::now();
            std::vector<int32_t> nxt(prompt.begin() + 1, prompt.end());
            nxt.push_back(tok);
            std::vector<float> y; int32_t d1 = 0;
            flash->extend(prompt_feats.data(), nxt.data(), T0, y, d1);
            float p1 = flash->last_prob();
            std::printf("flash drafter: prompt pass %.3f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count());
            std::vector<int32_t> draft, feed;
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
                        if (!out.empty()) break;
                    }
                }
            };
            int ng_cool = 0, ng_back = 0, ng_len = 7;   // adaptive prompt-lookup draft length (long failed drafts are expensive verifies)
            long ng_steps = 0, ng_acc = 0;
            double tp_draft = 0, tp_ext = 0, tp_ver[17] = {0};
            long np_ver[17] = {0};
            using SC = std::chrono::steady_clock;
            while (static_cast<int>(seq.size()) - T0 < n_new && !hit_eos) {
                const int budget = n_new - (static_cast<int>(seq.size()) - T0) - 1;
                const int32_t cur = seq.back();
                const size_t dpos = flash->position(), p0 = model.position();
                draft.clear();
                bool from_ngram = false;
                if (flash_ngram && ng_cool == 0) { ngram_lookup(draft, std::min(ng_len, budget)); from_ngram = draft.size() >= 2; if (!from_ngram) draft.clear(); }
                else if (ng_cool > 0) --ng_cool;
                const auto q0 = SC::now();
                if (!from_ngram) flash->draft(y, d1, p1, std::min(spec_k, budget), flash_min_p, draft);
                const int kd = static_cast<int>(draft.size());
                feed.assign(1, cur); feed.insert(feed.end(), draft.begin(), draft.end());
                const auto q1 = SC::now();
                const std::vector<int32_t> ids = model.forward_argmax(feed.data(), kd + 1, &st, &feats); ++passes;
                const auto q2 = SC::now();
                tp_draft += std::chrono::duration<double>(q1 - q0).count();
                tp_ver[std::min(kd + 1, 16)] += std::chrono::duration<double>(q2 - q1).count(); ++np_ver[std::min(kd + 1, 16)];
                int m = 0;
                while (m < kd && ids[m] == draft[m]) ++m;
                if (from_ngram) { ++ng_steps; ng_acc += m; ng_len = m == kd ? std::min(ng_len * 2, 7) : std::max(2, m + 1); if (m == 0) { ng_back = ng_back ? std::min(ng_back * 2, 32) : 2; ng_cool = ng_back; } else ng_back = 0; }
                if (kd > 0) { ++spec_steps; drafted += kd; accepted += m; }
                if (m < kd) model.rewind(p0 + static_cast<size_t>(m) + 1);
                std::vector<int32_t> to_add(draft.begin(), draft.begin() + m);
                to_add.push_back(ids[m]);
                const auto q3 = SC::now();
                flash->rewind(dpos);
                flash->extend(feats.data(), to_add.data(), m + 1, y, d1);
                p1 = flash->last_prob();
                tp_ext += std::chrono::duration<double>(SC::now() - q3).count();
                sum.seconds += st.seconds; sum.compute_seconds += st.compute_seconds; sum.head_seconds += st.head_seconds; sum.embed_seconds += st.embed_seconds;
                for (int32_t id : to_add) {
                    seq.push_back(id); emit(id); ++steps;
                    if (is_eos(id)) { hit_eos = true; break; }
                }
            }
            if (flash_ngram) std::printf("flash n-gram: %ld steps, %ld accepted\n", ng_steps, ng_acc);
            std::printf("flash time: draft %.3f s, drafter extend %.3f s; verify ms per call by width:", tp_draft, tp_ext);
            for (int w = 1; w <= 16; ++w) if (np_ver[w]) std::printf(" T=%d %.2f (x%ld)", w, 1000.0 * tp_ver[w] / np_ver[w], np_ver[w]);
            std::printf("\n");
            spec_k = std::max(spec_k, 1);
        } else if (spec_k > 0 && teacher.empty()) {
            // Speculative decoding, greedy-exact (same scheme as qwen_generate): prompt lookup proposes up to spec_k tokens, ONE forward of cur + drafts
            // gives the model's own greedy token after each, the longest agreeing prefix is kept and the KV length rewound past the rest (Gemma's only
            // state is its K/V cache). A step with no accepted draft backs off for a few tokens, so non-repetitive text is never slower.
            if (spec_k > 15) spec_k = 15;
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
            while (static_cast<int>(seq.size()) - T0 < n_new && !hit_eos) {
                const int budget = n_new - (static_cast<int>(seq.size()) - T0) - 1;
                if (cooldown > 0) { --cooldown; draft.clear(); } else lookup(draft, std::min(spec_k, budget));
                const int32_t cur = seq.back();
                std::vector<int32_t> to_add;
                if (draft.empty()) {
                    to_add.push_back(model.forward_greedy(cur, &st)); ++passes;
                } else {
                    const int kd = static_cast<int>(draft.size());
                    feed.assign(1, cur); feed.insert(feed.end(), draft.begin(), draft.end());
                    const size_t p0 = model.position();
                    const std::vector<int32_t> ids = model.forward_argmax(feed.data(), kd + 1, &st); ++passes; ++spec_steps; drafted += kd;
                    int m = 0;
                    while (m < kd && ids[m] == draft[m]) ++m;
                    accepted += m;
                    if (m == 0) { backoff = backoff ? std::min(backoff * 2, 32) : 2; cooldown = backoff; } else backoff = 0;   // a failed check costs several steps: back off hard
                    if (m < kd) model.rewind(p0 + static_cast<size_t>(m) + 1);   // keep cur + the accepted drafts
                    for (int q = 0; q < m; ++q) to_add.push_back(draft[q]);
                    to_add.push_back(ids[m]);
                }
                sum.seconds += st.seconds; sum.compute_seconds += st.compute_seconds; sum.head_seconds += st.head_seconds; sum.embed_seconds += st.embed_seconds;
                for (int32_t id : to_add) {
                    seq.push_back(id); emit(id); ++steps;
                    if (is_eos(id)) { hit_eos = true; break; }
                }
            }
        }
        for (int k = 1; (spec_k == 0 || !teacher.empty()) && k < n_new && !hit_eos; ++k) {
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
        if (spec_k > 0 && spec_steps > 0)
            std::printf("speculation: %ld verify steps, %ld drafted, %ld accepted (%.0f%%, %.2f per step), %ld model passes for %d tokens\n", spec_steps, drafted, accepted,
                        drafted ? 100.0 * accepted / drafted : 0.0, static_cast<double>(accepted) / spec_steps, passes, steps);
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
