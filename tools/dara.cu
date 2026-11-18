// dara.exe: single front end for every Dara tier. Reads general.architecture from the GGUF header and hands the command line,
// unchanged, to the matching driver:
//   qwen35  -> Core 4B / Pro 9B   (tools/qwen_generate.cu)
//   gemma4  -> Nano               (tools/nano_generate.cu)
// The drivers are compiled into this executable with their main() renamed (-Dmain=...), so every option they accept works here.
//
// Context: --kv auto is the default here (the largest context that keeps every layer and the output head in VRAM, after room for the
// Flash drafter; a model that does not fit fully, e.g. 9B on 4 GB, gets 32K and the planner places what fits). --kv N overrides.
//
// Usage: dara --gguf <model.gguf> [driver options...]
//        dara --info --gguf <model.gguf>      print the detected architecture and exit
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "io/gguf.h"

int dara_qwen_main(int argc, char** argv);
int dara_nano_main(int argc, char** argv);
int dara_probe_main();   // tests/bw_probe.cu: GPU memory-bandwidth probe (dara --probe)
int dara_serve_main(int argc, char** argv);   // src/serve/server.cpp: dara serve --model <gguf> (HTTP API)

int main(int argc, char** argv) {
    if (argc >= 2 && std::strcmp(argv[1], "--probe") == 0) return dara_probe_main();   // memory-bandwidth ceiling of this GPU
    if (argc >= 2 && std::strcmp(argv[1], "serve") == 0) return dara_serve_main(argc - 1, argv + 1);
    std::string gguf;
    bool info = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gguf") == 0 && i + 1 < argc) gguf = argv[i + 1];
        else if (std::strcmp(argv[i], "--info") == 0) info = true;
    }
    if (gguf.empty()) {
        std::fprintf(stderr,
                     "usage: dara --gguf <model.gguf> [options]\n"
                     "  the model family is detected from the GGUF (qwen35: Core 4B / Pro 9B, gemma4: Nano);\n"
                     "  all options of the matching driver apply (--prompt, --prompt-file, --new, --flash, --spec, ...)\n"
                     "  dara --info --gguf <model.gguf> prints the detected architecture\n"
                     "  dara --probe measures this GPU's memory bandwidth (the ceiling for decode speed)\n"
                     "  dara serve --model <model.gguf> [--port 11435] runs the HTTP server (OpenAI-compatible API, /dara/stats)\n");
        return 2;
    }
    std::string arch;
    try {
        for (const auto& kv : dara::io::read_gguf_header(gguf).metadata)
            if (kv.first == "general.architecture") { arch = kv.second; break; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: cannot read %s: %s\n", gguf.c_str(), e.what());
        return 1;
    }
    const char* tier = arch == "qwen35" ? "qwen (Core 4B / Pro 9B)" : arch == "gemma4" ? "nano (Gemma 4 E2B)" : nullptr;
    if (!tier) {
        std::fprintf(stderr, "ERROR: unsupported GGUF architecture '%s' (Dara runs qwen35 and gemma4)\n", arch.c_str());
        return 1;
    }
    if (info) { std::printf("architecture: %s -> %s\n", arch.c_str(), tier); return 0; }
    // default context: auto (the largest that keeps the model in VRAM; see ResidencyOptions) unless --kv is given
    std::vector<char*> args(argv, argv + argc);
    bool has_kv = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--kv") == 0) has_kv = true;
    static char kv_flag[] = "--kv", kv_auto[] = "auto";
    if (!has_kv) { args.push_back(kv_flag); args.push_back(kv_auto); }
    const int n = static_cast<int>(args.size());
    args.push_back(nullptr);
    return arch == "qwen35" ? dara_qwen_main(n, args.data()) : dara_nano_main(n, args.data());
}
