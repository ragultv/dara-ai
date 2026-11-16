// dara.exe: single front end for every Dara tier. Reads general.architecture from the GGUF header and hands the command line,
// unchanged, to the matching driver:
//   qwen35  -> Core 4B / Pro 9B   (tools/qwen_generate.cu)
//   gemma4  -> Nano               (tools/nano_generate.cu)
// The drivers are compiled into this executable with their main() renamed (-Dmain=...), so every option they accept works here.
//
// Usage: dara --gguf <model.gguf> [driver options...]
//        dara --info --gguf <model.gguf>      print the detected architecture and exit
#include <cstdio>
#include <cstring>
#include <string>

#include "io/gguf.h"

int dara_qwen_main(int argc, char** argv);
int dara_nano_main(int argc, char** argv);

int main(int argc, char** argv) {
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
                     "  dara --info --gguf <model.gguf> prints the detected architecture\n");
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
    return arch == "qwen35" ? dara_qwen_main(argc, argv) : dara_nano_main(argc, argv);
}
