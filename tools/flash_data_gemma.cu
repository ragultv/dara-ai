// Dara Flash training data (Gemma family / Nano): self-distillation, same outputs as tools/flash_data_qwen.cu (seqs.bin, feats.bin, index.jsonl).
// Each prompt is rendered with the Gemma 4 chat template (no thinking), the model's own greedy answer is generated, then the whole sequence is run once
// more and the final normed hidden state of every position is recorded.
// Usage: flash_data_gemma --gguf <model.gguf> --prompts <prompts.jsonl> --out-dir <dir> [--new 256] [--limit N] [--start K]
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#ifdef _WIN32
#include <windows.h>   // SetThreadExecutionState: long runs must not let the laptop idle into Modern Standby (the dGPU does not come back)
#endif

#include "io/gguf_checkpoint.h"
#include "kernels/half.h"
#include "model/gemma_model.h"
#include "tokenizer/gemma_chat.h"
#include "tokenizer/gemma_tokenizer.h"

using namespace dara::model;
using dara::tokenizer::GemmaTokenizer;

int main(int argc, char** argv) {
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED);   // display too: on Modern Standby laptops the display timeout alone triggers standby
#endif
    std::string gguf_path, prompts_path, out_dir;
    int n_new = 256, limit = 1 << 30, start = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--gguf") gguf_path = next();
        else if (a == "--prompts") prompts_path = next();
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--new") n_new = std::atoi(next().c_str());
        else if (a == "--limit") limit = std::atoi(next().c_str());
        else if (a == "--start") start = std::atoi(next().c_str());
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (gguf_path.empty() || prompts_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "usage: flash_data_gemma --gguf <model.gguf> --prompts <prompts.jsonl> --out-dir <dir> [--new 256] [--limit N] [--start K]\n");
        return 2;
    }
    try {
        dara::io::GgufCheckpoint gguf(gguf_path);
        const GemmaTokenizer tok = GemmaTokenizer::from_gguf(gguf.header());
        GemmaModel model(gguf, 2048);
        const int H = model.config().hidden;
        const int32_t eos = tok.eos_id();

        std::ifstream pin(prompts_path, std::ios::binary);
        const bool append = start > 0;
        std::ofstream seqs(out_dir + "/seqs.bin", std::ios::binary | (append ? std::ios::app : std::ios::trunc));
        std::ofstream feats(out_dir + "/feats.bin", std::ios::binary | (append ? std::ios::app : std::ios::trunc));
        std::ofstream index(out_dir + "/index.jsonl", append ? std::ios::app : std::ios::trunc);
        size_t feat_offset = 0;
        if (append) {
            std::ifstream f(out_dir + "/feats.bin", std::ios::binary | std::ios::ate);
            feat_offset = static_cast<size_t>(f.tellg()) / 2;
        }
        std::string line;
        int done = 0, n_line = 0;
        long total_tokens = 0;
        while (std::getline(pin, line) && done < limit) {
            if (line.empty()) continue;
            if (n_line++ < start) continue;
            const auto j = nlohmann::json::parse(line);
            std::vector<int32_t> seq = dara::tokenizer::tokenize_gemma_chat(tok, {{"user", j["prompt"].get<std::string>()}}, true, false);
            const int n_prompt = static_cast<int>(seq.size());
            if (n_prompt + n_new + 1 > 2048) continue;
            // 1. the model's own greedy answer
            model.rewind(0);
            const std::vector<float> lg = model.forward(seq.data(), n_prompt, n_prompt - 1);
            int32_t t = 0;
            for (int i = 1; i < model.config().vocab; ++i) if (lg[i] > lg[t]) t = i;
            for (int k = 0; k < n_new; ++k) {
                seq.push_back(t);
                if (t == eos || t == 1) break;
                t = model.forward_greedy(t);
            }
            // 2. final normed hidden states of the whole sequence
            model.rewind(0);
            std::vector<float> f;
            model.forward(seq.data(), static_cast<int>(seq.size()), static_cast<int>(seq.size()), nullptr, nullptr, &f);
            std::vector<uint16_t> f16(f.size());
            for (size_t i = 0; i < f.size(); ++i) f16[i] = dara::kernels::float_to_half(f[i]);
            const int32_t hdr[2] = {static_cast<int32_t>(seq.size()), n_prompt};
            seqs.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
            seqs.write(reinterpret_cast<const char*>(seq.data()), static_cast<std::streamsize>(seq.size() * sizeof(int32_t)));
            feats.write(reinterpret_cast<const char*>(f16.data()), static_cast<std::streamsize>(f16.size() * sizeof(uint16_t)));
            index << nlohmann::json{{"id", j["id"]}, {"n_total", seq.size()}, {"n_prompt", n_prompt}, {"feat_offset", feat_offset}}.dump() << "\n";
            feat_offset += f16.size();
            total_tokens += static_cast<long>(seq.size());
            if (++done % 10 == 0) {
                seqs.flush(); feats.flush(); index.flush();
                std::printf("%d sequences, %ld tokens\n", done, total_tokens);
                std::fflush(stdout);
            }
            (void)H;
        }
        std::printf("done: %d sequences, %ld tokens -> %s\n", done, total_tokens, out_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
