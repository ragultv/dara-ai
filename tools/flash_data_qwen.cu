// Dara Flash training data (Qwen family): self-distillation from the target model.
// For every prompt of a JSONL file ({"id":..,"prompt":"..."}): wrap it in the chat template, generate the model's own greedy answer (the default fast mode,
// i.e. exactly the model the drafter will run against), then run the whole sequence once more and record the FINAL hidden state of every position (after
// the final norm: the vector the output head reads), which the drafter learns to predict.
//
// Outputs in --out-dir:  seqs.bin   per sequence: int32 n_total, int32 n_prompt, int32 tokens[n_total]
//                        feats.bin  per sequence: fp16 [n_total][hidden]
//                        index.jsonl {"id", "n_total", "n_prompt", "feat_offset"} (feat_offset in fp16 elements)
// Usage: flash_data_qwen --gguf <model.gguf> --prompts <prompts.jsonl> --out-dir <dir> [--new 256] [--limit N] [--start K]
//        flash_data_qwen --gguf <model.gguf> --from-seqs <dir> --out-dir <dir> [--limit N]
//   --from-seqs: features only. Re-uses the token sequences of another model's data directory (e.g. the 4B's answers for a 9B drafter: prefill-speed
//   instead of decode-speed data generation). Train with --self-targets: the token target is this model's own argmax, derived from the next feature.
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
#include "model/ops.h"
#include "model/qwen_model.h"
#include "tokenizer/tokenizer.h"

using namespace dara::model;

int main(int argc, char** argv) {
#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
#endif
    std::string gguf_path, prompts_path, out_dir, from_seqs;
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
        else if (a == "--from-seqs") from_seqs = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (gguf_path.empty() || (prompts_path.empty() && from_seqs.empty()) || out_dir.empty()) {
        std::fprintf(stderr, "usage: flash_data_qwen --gguf <model.gguf> --prompts <prompts.jsonl> --out-dir <dir> [--new 256] [--limit N] [--start K]\n");
        return 2;
    }
    try {
        dara::io::GgufCheckpoint gguf(gguf_path);
        const QwenConfig cfg = QwenConfig::from_gguf(gguf);
        auto tokenizer = std::make_unique<dara::tokenizer::Tokenizer>(dara::tokenizer::Tokenizer::from_gguf(gguf.header()));
        std::unique_ptr<IModelWeights> weights = std::make_unique<GgufWeights>(gguf, cfg);
        QwenModel model(cfg, *weights, 2048);
        const QwenModel::Checkpoint empty = model.checkpoint();
        const std::vector<float> final_scale = weights->final_norm_scale();
        const int H = cfg.hidden;
        const std::vector<int32_t> im_end = tokenizer->encode("<|im_end|>", true);
        const int32_t end_turn = im_end.size() == 1 ? im_end[0] : -1;
        const int32_t eos = tokenizer->eos_id();

        if (!from_seqs.empty()) {   // features-only mode
            std::ifstream sin(from_seqs + "/seqs.bin", std::ios::binary);
            std::ifstream iin(from_seqs + "/index.jsonl");
            std::ofstream seqs(out_dir + "/seqs.bin", std::ios::binary), feats(out_dir + "/feats.bin", std::ios::binary), index(out_dir + "/index.jsonl");
            size_t feat_offset = 0;
            int done = 0;
            long total = 0;
            std::string iline;
            std::vector<int32_t> ids;
            std::vector<float> f;
            while (done < limit && std::getline(iin, iline)) {
                int32_t hdr[2];
                if (!sin.read(reinterpret_cast<char*>(hdr), 8)) break;
                std::vector<int32_t> seq(static_cast<size_t>(hdr[0]));
                sin.read(reinterpret_cast<char*>(seq.data()), static_cast<std::streamsize>(seq.size() * 4));
                const auto j = nlohmann::json::parse(iline);
                if (seq.size() > 2000) continue;
                model.restore(empty);
                model.forward(seq.data(), static_cast<int>(seq.size()), static_cast<int>(seq.size()), nullptr, nullptr, nullptr, &f, 0);   // features only: no head pass (the trainer derives targets from the next features)
                std::vector<uint16_t> f16(f.size());
                for (size_t i = 0; i < f.size(); ++i) f16[i] = dara::kernels::float_to_half(f[i]);
                seqs.write(reinterpret_cast<const char*>(hdr), 8);
                seqs.write(reinterpret_cast<const char*>(seq.data()), static_cast<std::streamsize>(seq.size() * 4));
                feats.write(reinterpret_cast<const char*>(f16.data()), static_cast<std::streamsize>(f16.size() * 2));
                index << nlohmann::json{{"id", j["id"]}, {"n_total", seq.size()}, {"n_prompt", hdr[1]}, {"feat_offset", feat_offset}}.dump() << "\n";
                feat_offset += f16.size();
                total += static_cast<long>(seq.size());
                if (++done % 10 == 0) { seqs.flush(); feats.flush(); index.flush(); std::printf("%d sequences, %ld tokens\n", done, total); std::fflush(stdout); }
            }
            std::printf("done: %d sequences, %ld tokens -> %s\n", done, total, out_dir.c_str());
            return 0;
        }
        std::ifstream pin(prompts_path, std::ios::binary);
        const bool append = start > 0;
        std::ofstream seqs(out_dir + "/seqs.bin", std::ios::binary | (append ? std::ios::app : std::ios::trunc));
        std::ofstream feats(out_dir + "/feats.bin", std::ios::binary | (append ? std::ios::app : std::ios::trunc));
        std::ofstream index(out_dir + "/index.jsonl", append ? std::ios::app : std::ios::trunc);
        size_t feat_offset = 0;
        if (append) {   // continue after an interrupted run: the next offset is the current size of feats.bin
            std::ifstream f(out_dir + "/feats.bin", std::ios::binary | std::ios::ate);
            feat_offset = static_cast<size_t>(f.tellg()) / 2;
        }
        std::string line;
        int done = 0, n_line = 0;
        long total_tokens = 0;
        std::vector<int32_t> ids;
        while (std::getline(pin, line) && done < limit) {
            if (line.empty()) continue;
            if (n_line++ < start) continue;
            const auto j = nlohmann::json::parse(line);
            const std::string text = "<|im_start|>user\n" + j["prompt"].get<std::string>() + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            std::vector<int32_t> seq = tokenizer->encode(text, true);
            const int n_prompt = static_cast<int>(seq.size());
            // 1. the model's own greedy answer
            model.restore(empty);
            model.forward(seq.data(), n_prompt, n_prompt - 1, nullptr, nullptr, &ids);
            int32_t tok = ids[0];
            for (int k = 0; k < n_new; ++k) {
                seq.push_back(tok);
                if (tok == eos || tok == end_turn) break;
                model.forward(&tok, 1, 0, nullptr, nullptr, &ids);
                tok = ids[0];
            }
            // 2. final hidden states of the whole sequence (one prompt-style pass)
            model.restore(empty);
            std::vector<std::vector<float>> layer_out;
            model.forward(seq.data(), static_cast<int>(seq.size()), static_cast<int>(seq.size()), &layer_out, nullptr);
            const std::vector<float>& last = layer_out.back();
            std::vector<uint16_t> f16(seq.size() * static_cast<size_t>(H));
            std::vector<float> normed(H);
            for (size_t t = 0; t < seq.size(); ++t) {
                rmsnorm(&last[t * H], final_scale.data(), H, cfg.eps, normed.data());
                for (int i = 0; i < H; ++i) f16[t * H + i] = dara::kernels::float_to_half(normed[i]);
            }
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
        }
        std::printf("done: %d sequences, %ld tokens -> %s\n", done, total_tokens, out_dir.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
