// dara_tokenize: encode / decode text with the GGUF-native tokenizer.
//   dara_tokenize <model.gguf> encode [--special] <utf8 text file>     prints the ids, space separated
//   dara_tokenize <model.gguf> decode <id> <id> ...                    prints the text
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "io/gguf.h"
#include "tokenizer/tokenizer.h"

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: dara_tokenize <model.gguf> encode [--special] <file> | decode <ids...>\n"); return 2; }
    try {
        const auto tok = dara::tokenizer::Tokenizer::from_gguf(dara::io::read_gguf_header(argv[1]));
        const std::string cmd = argv[2];
        if (cmd == "encode") {
            bool special = false;
            int a = 3;
            if (std::string(argv[a]) == "--special") { special = true; ++a; }
            std::ifstream f(argv[a], std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            const auto ids = tok.encode(text, special);
            for (size_t i = 0; i < ids.size(); ++i) std::printf("%d%c", ids[i], i + 1 < ids.size() ? ' ' : '\n');
        } else if (cmd == "decode") {
            std::vector<int32_t> ids;
            for (int a = 3; a < argc; ++a) ids.push_back(std::atoi(argv[a]));
            std::fwrite(tok.decode(ids).data(), 1, tok.decode(ids).size(), stdout);
        } else { std::fprintf(stderr, "unknown command %s\n", cmd.c_str()); return 2; }
    } catch (const std::exception& e) { std::fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
    return 0;
}
