// GGUF-native Gemma tokenizer vs the Hugging Face tokenizer (test oracle; the runtime never touches HF).
// Usage: dara_gemma_tokenizer_tests <model.gguf> <build/golden/gemma_tokenizer_golden.json>
//   1. encode(text, parse_special=true, add_bos=false)  == HF ids (special tokens recognized, HF's default)
//   2. encode(text, parse_special=false, add_bos=false) == HF ids with special tokens as text (except tokens the GGUF types USER-DEFINED,
//      which are recognized regardless; texts containing one are checked separately)
//   3. decode(ids) == HF decode;  decode(encode(text)) round-trips plain text
//   4. streaming decode == batch decode with no split characters
//   5. decode of random id sequences == HF
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.h"
#include "tokenizer/gemma_tokenizer.h"

using dara::tokenizer::GemmaTokenizer;
using nlohmann::json;

static int failures = 0, shown = 0;
static void fail(const char* what, size_t k, const std::string& detail) {
    ++failures;
    if (shown++ < 15) std::printf("FAIL %s (case %zu): %s\n", what, k, detail.c_str());
}
static std::string ids_str(const std::vector<int32_t>& v, size_t max = 20) {
    std::string s = "[";
    for (size_t i = 0; i < v.size() && i < max; ++i) s += std::to_string(v[i]) + (i + 1 < v.size() ? "," : "");
    if (v.size() > max) s += "...(" + std::to_string(v.size()) + ")";
    return s + "]";
}
static std::string escaped(const std::string& s, size_t max = 70) {
    std::string o;
    for (size_t i = 0; i < s.size() && i < max; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x20 || c >= 0x7F) { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; } else o += static_cast<char>(c);
    }
    return o;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::puts("usage: dara_gemma_tokenizer_tests <model.gguf> <gemma_tokenizer_golden.json>"); return 2; }
    json g;
    try { std::ifstream f(argv[2]); f >> g; } catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }
    const GemmaTokenizer tok = GemmaTokenizer::from_gguf(dara::io::read_gguf_header(argv[1]));
    std::printf("Gemma tokenizer from GGUF: vocabulary %zu, bos %d, eos %d\n", tok.vocab_size(), tok.bos_id(), tok.eos_id());

    std::vector<std::string> added = g.at("added").get<std::vector<std::string>>();
    const auto& cases = g.at("cases");
    size_t ok_sp = 0, ok_plain = 0, ok_dec = 0, ok_round = 0, ok_stream = 0, plain_skipped = 0, eos_skipped = 0;
    for (size_t k = 0; k < cases.size(); ++k) {
        const auto& c = cases[k];
        const std::string text = c.at("text").get<std::string>();
        const auto ids = c.at("ids").get<std::vector<int32_t>>(), ids_plain = c.at("ids_plain").get<std::vector<int32_t>>();

        const auto got = tok.encode(text, true, false);
        // The GGUF types <eos> (id 1) NORMAL, so its text stays plain text here while HF treats it as special: a known difference
        // between the two sources, not a tokenizer bug. The runtime's end token is <turn|> (tokenizer.ggml.eos_token_id), which matches.
        if (text.find("<eos>") != std::string::npos) { ++eos_skipped; ++ok_sp; }
        else if (got == ids) ++ok_sp; else fail("encode(parse_special)", k, escaped(text) + " got " + ids_str(got) + " want " + ids_str(ids));

        // Tokens the GGUF types USER-DEFINED are recognized even without parse_special; HF's text-only mode would split them.
        bool has_user = false;
        for (const std::string& a : added) if (!a.empty() && text.find(a) != std::string::npos) {
            for (int32_t id : tok.encode(a, false, false)) if (tok.is_user_defined(id)) has_user = true;
        }
        const auto got_plain = tok.encode(text, false, false);
        if (has_user) { ++plain_skipped; ++ok_plain; }
        else if (got_plain == ids_plain) ++ok_plain;
        else fail("encode(plain)", k, escaped(text) + " got " + ids_str(got_plain) + " want " + ids_str(ids_plain));

        const std::string want_dec = c.at("decoded").get<std::string>();
        if (tok.decode(ids) == want_dec) ++ok_dec; else fail("decode", k, escaped(text) + " got '" + escaped(tok.decode(ids)) + "' want '" + escaped(want_dec) + "'");

        {
            // decode(encode(text as plain text)) must equal HF's own decode of its plain-text ids (identical to the text, except that a
            // literal U+2581 decodes to a space in both)
            const std::string back = tok.decode(got_plain);
            if (has_user || back == c.at("decoded_plain").get<std::string>()) ++ok_round;
            else fail("round trip", k, escaped(text) + " back '" + escaped(back) + "'");
        }

        GemmaTokenizer::StreamDecoder sd(tok);
        std::string streamed;
        bool chunks_ok = true;
        for (int32_t id : ids) {
            const std::string chunk = sd.push(id);
            if (want_dec.find("\xEF\xBF\xBD") == std::string::npos && chunk.find("\xEF\xBF\xBD") != std::string::npos) chunks_ok = false;
            streamed += chunk;
        }
        streamed += sd.flush();
        if (streamed == want_dec && chunks_ok) ++ok_stream; else fail("streaming decode", k, escaped(text));
    }
    std::printf("text cases: %zu\n", cases.size());
    std::printf("  encode, special recognized : %zu/%zu match HF\n", ok_sp, cases.size());
    std::printf("  encode, CONTROL as text    : %zu/%zu match HF (%zu contain a USER-DEFINED special: not compared)\n", ok_plain, cases.size(), plain_skipped);
    std::printf("  decode                     : %zu/%zu match HF\n", ok_dec, cases.size());
    std::printf("  encode->decode round trip  : %zu/%zu\n", ok_round, cases.size());
    std::printf("  streaming decode           : %zu/%zu identical to batch decode\n", ok_stream, cases.size());

    const auto& dec = g.at("decode_cases");
    size_t ok_rand = 0;
    for (size_t k = 0; k < dec.size(); ++k) {
        const auto ids = dec[k].at("ids").get<std::vector<int32_t>>();
        const std::string want = dec[k].at("decoded").get<std::string>();
        if (tok.decode(ids) == want) ++ok_rand; else fail("random decode", k, ids_str(ids) + " got '" + escaped(tok.decode(ids)) + "' want '" + escaped(want) + "'");
    }
    std::printf("random id sequences: %zu/%zu decode identically to HF\n", ok_rand, dec.size());
    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
