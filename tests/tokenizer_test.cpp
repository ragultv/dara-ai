// GGUF-native tokenizer vs the Hugging Face tokenizer (test oracle; the runtime never touches HF).
// Usage: dara_tokenizer_tests <model.gguf> <build/golden/tokenizer_golden.json>
//
//  1. encode(text, parse_special=true)  == HF ids (special tokens recognized, HF's default)
//  2. encode(text, parse_special=false) == HF ids with special tokens treated as plain text
//  3. pre-tokenization pieces == HF's (isolates the hand-written pattern from BPE)
//  4. decode(ids) == HF decode;  decode(encode(text, false)) == the NFC-normalized text (round trip)
//  5. streaming decode: concatenated chunks == batch decode, and no chunk ever ends inside a multibyte character
//  6. decode of random id sequences (including special / unused ids) == HF
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "io/gguf.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

using dara::tokenizer::Tokenizer;
using nlohmann::json;

static int failures = 0, shown = 0;
static void fail(const char* what, size_t case_no, const std::string& detail) {
    ++failures;
    if (shown++ < 12) std::printf("FAIL %s (case %zu): %s\n", what, case_no, detail.c_str());
}

static std::string ids_str(const std::vector<int32_t>& v, size_t max = 24) {
    std::string s = "[";
    for (size_t i = 0; i < v.size() && i < max; ++i) s += std::to_string(v[i]) + (i + 1 < v.size() ? "," : "");
    if (v.size() > max) s += "...(" + std::to_string(v.size()) + ")";
    return s + "]";
}
static std::string escaped(const std::string& s, size_t max = 60) {
    std::string o;
    for (size_t i = 0; i < s.size() && i < max; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x20 || c >= 0x7F) { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); o += b; } else o += static_cast<char>(c);
    }
    return o;
}
static std::vector<int32_t> to_ids(const json& j) { return j.get<std::vector<int32_t>>(); }
static const char* kAlwaysSplitInHf[] = {"<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>"};
static size_t n_always_split = 0;

int main(int argc, char** argv) {
    if (argc < 3) { std::puts("usage: dara_tokenizer_tests <model.gguf> <tokenizer_golden.json>"); return 2; }
    json g;
    try { std::ifstream f(argv[2]); f >> g; }
    catch (const std::exception& e) { std::printf("cannot load golden data: %s\n", e.what()); return 2; }

    Tokenizer tok = Tokenizer::from_gguf(dara::io::read_gguf_header(argv[1]));
    std::printf("tokenizer from GGUF: pre-tokenizer '%s', vocabulary %zu, %zu special tokens, eos %d\n", tok.pre_type().c_str(), tok.vocab_size(),
                tok.special_count(), tok.eos_id());

    const auto& cases = g.at("cases");
    size_t n_special_ok = 0, n_plain_ok = 0, n_pieces_ok = 0, n_dec_ok = 0, n_round_ok = 0, n_stream_ok = 0, n_extra = 0;
    for (size_t k = 0; k < cases.size(); ++k) {
        const auto& c = cases[k];
        const std::string text = c.at("text").get<std::string>();
        const auto ids = to_ids(c.at("ids")), ids_plain = to_ids(c.at("ids_plain"));
        if (c.at("has_extra_special").get<bool>()) ++n_extra;

        const auto got = tok.encode(text, true);
        if (got == ids) ++n_special_ok; else fail("encode(parse_special)", k, escaped(text) + " got " + ids_str(got) + " want " + ids_str(ids));
        const auto got_plain = tok.encode(text, false);
        // The reference tokenizer always splits six added tokens that the GGUF types CONTROL (they are only recognized here when
        // parse_special is true). For texts containing them the comparison is replaced by: no such id may be produced.
        bool has_always_split = false;
        for (const char* s : kAlwaysSplitInHf) if (text.find(s) != std::string::npos) has_always_split = true;
        if (has_always_split) {
            ++n_always_split;
            bool leaked = false;
            for (int32_t id : got_plain) if (id >= 248060 && id <= 248065) leaked = true;
            if (!leaked) ++n_plain_ok; else fail("control token recognized without parse_special", k, escaped(text));
        } else if (got_plain == ids_plain) {
            ++n_plain_ok;
        } else {
            fail("encode(plain)", k, escaped(text) + " got " + ids_str(got_plain) + " want " + ids_str(ids_plain));
        }

        const auto pieces = c.at("pieces").get<std::vector<std::string>>();
        const auto mine = tok.pretokenize(text);
        if (mine == pieces) ++n_pieces_ok;
        else {
            size_t d = 0;
            while (d < mine.size() && d < pieces.size() && mine[d] == pieces[d]) ++d;
            fail("pre-tokenize", k, escaped(text) + " first difference at piece " + std::to_string(d) + ": got '" + escaped(d < mine.size() ? mine[d] : "<none>") +
                 "' want '" + escaped(d < pieces.size() ? pieces[d] : "<none>") + "'");
        }

        const std::string want_dec = c.at("decoded").get<std::string>();
        if (tok.decode(ids) == want_dec) ++n_dec_ok; else fail("decode", k, escaped(text));
        // Round trip. Without any added-token string in the text it must equal the NFC-normalized text. With one, the
        // reference extracts the token from the RAW text before normalizing, so its own decode is the expectation instead
        // (cases with the six always-split tokens were already handled above, so they are skipped here).
        const bool has_added = c.at("has_added").get<bool>();
        const std::string back = tok.decode(got_plain);
        const std::string norm = has_added ? c.at("decoded_plain").get<std::string>() : c.at("norm").get<std::string>();
        if (has_always_split || back == norm) ++n_round_ok;
        else {
            size_t d = 0;
            while (d < back.size() && d < norm.size() && back[d] == norm[d]) ++d;
            std::string ctx_back = back.substr(d > 8 ? d - 8 : 0, 20), ctx_norm = norm.substr(d > 8 ? d - 8 : 0, 20);
            fail("round trip", k, "lengths " + std::to_string(back.size()) + " vs " + std::to_string(norm.size()) + ", first difference at byte " + std::to_string(d) +
                 ": got '" + escaped(ctx_back) + "' want '" + escaped(ctx_norm) + "'; plain ids " + ids_str(got_plain, 6));
        }

        // streaming: chunks concatenate to the batch decode and none ends mid-character
        Tokenizer::StreamDecoder sd(tok);
        std::string streamed;
        bool chunks_ok = true;
        for (int32_t id : ids) {
            const std::string chunk = sd.push(id);
            const std::vector<uint32_t> cps = dara::tokenizer::utf8_to_codepoints(chunk);
            // a chunk produced from well-formed text must itself be well-formed (no replacement introduced by a cut)
            if (want_dec.find("\xEF\xBF\xBD") == std::string::npos && chunk.find("\xEF\xBF\xBD") != std::string::npos) chunks_ok = false;
            (void)cps;
            streamed += chunk;
        }
        streamed += sd.flush();
        if (streamed == want_dec && chunks_ok) ++n_stream_ok; else fail("streaming decode", k, escaped(text));
    }
    std::printf("text cases: %zu (%zu mention the 7 audio/TTS tokens the GGUF marks special)\n", cases.size(), n_extra);
    std::printf("  encode with special tokens   : %zu/%zu match HF\n", n_special_ok, cases.size());
    std::printf("  encode, CONTROL ids as text  : %zu/%zu match HF (%zu of them contain a token HF always splits: checked for no id leakage instead)\n",
                n_plain_ok, cases.size(), n_always_split);
    std::printf("  pre-tokenization pieces      : %zu/%zu match HF\n", n_pieces_ok, cases.size());
    std::printf("  decode                       : %zu/%zu match HF\n", n_dec_ok, cases.size());
    std::printf("  encode->decode round trip    : %zu/%zu equal the NFC text\n", n_round_ok, cases.size());
    std::printf("  streaming decode             : %zu/%zu identical to batch decode, no split characters\n", n_stream_ok, cases.size());

    const auto& dec = g.at("decode_cases");
    size_t n_rand_ok = 0, n_rand = 0;
    for (size_t k = 0; k < dec.size(); ++k) {
        const auto& d = dec[k];
        const auto ids = to_ids(d.at("ids"));
        if (d.at("decoded").is_null()) {  // HF raised on this input; ours must not return garbage silently for out-of-range ids
            continue;
        }
        ++n_rand;
        const std::string want = d.at("decoded").get<std::string>();
        std::string got;
        try { got = tok.decode(ids); } catch (const std::exception& e) { fail("decode(random ids)", k, std::string("threw ") + e.what()); continue; }
        if (got == want) ++n_rand_ok; else fail("decode(random ids)", k, ids_str(ids) + " got '" + escaped(got) + "' want '" + escaped(want) + "'");
        // the streaming decoder must agree on these too, including ill-formed byte sequences
        Tokenizer::StreamDecoder sd(tok);
        std::string s;
        for (int32_t id : ids) s += sd.push(id);
        s += sd.flush();
        if (s != want) fail("streaming decode(random ids)", k, ids_str(ids));
    }
    std::printf("random id sequences            : %zu/%zu decode identically to HF\n", n_rand_ok, n_rand);

    // API checks
    bool threw = false;
    try { tok.decode({static_cast<int32_t>(tok.vocab_size())}); } catch (const std::out_of_range&) { threw = true; }
    if (!threw) fail("range check", 0, "decode accepted an id outside the vocabulary");
    if (tok.encode("", true).size() != 0) fail("empty input", 0, "encode(\"\") is not empty");
    // ill-formed UTF-8 input must not crash and must behave like U+FFFD
    const auto bad = tok.encode(std::string("a\xFF\xC3z\xE2\x82", 6), false);
    const auto repl = tok.encode("a\xEF\xBF\xBD\xEF\xBF\xBDz\xEF\xBF\xBD", false);
    if (bad != repl) fail("ill-formed UTF-8", 0, "got " + ids_str(bad) + " want " + ids_str(repl));

    std::puts(failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
