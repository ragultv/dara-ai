"""Reference data for the tokenizer from the pinned Hugging Face tokenizer (the TEST ORACLE; the runtime never uses it).

Per text case: ids with special tokens recognized (HF default) and with special tokens treated as text, the byte-level
pre-tokenization pieces, the NFC-normalized text, and HF's decode of the ids. Plus random id sequences with HF's decode.
Cases: hand-written edge cases, and a seeded fuzz over weighted character pools (scripts, marks, emoji, whitespace, newlines).
Output: build/golden/tokenizer_golden.json   Usage: python -I tools/make_tokenizer_golden.py --model models/qwen3.5-4b
"""
import argparse
import json
import os
import random
import sys

sys.modules["torchao"] = None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TAMIL = "தமிழ் மொழி மிகவும் இனிமையானது. வணக்கம் உலகம்! கொ கோ கௌ"
HAND = [
    "", " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r", " \n", "\n ", "a\n\nb", "a \n b", "a  b", "a   ", "   a", " a", "a\tb\t\tc",
    "x\r\ny\r\n\r\nz", "line1\nline2\n", "trailing newlines\n\n\n", "\n\n\nleading", "  \n  \n  ", "a b", "a　b", "a b", "a​b",
    "﻿BOM", "tab\tseparated\tvalues",
    "The capital of France is", "Hello, world!", "hello", "Hello", "HELLO", "don't", "I'M", "WE'LL", "it'S", "they'RE", "I'VE", "he'd", "she'll", "'", "''s",
    "x's", "'tis", "o'clock", "rock'n'roll", "ſ'ſ", "can't've",
    "12345", "3.14159", "1,000,000", "0123456789", "٣٤٥٦", "²³", "Ⅷ", "½ cup", "x2", "a1b2c3", "1st 2nd 3rd", "2026-11-09", "$100.50", "50%",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }", "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}", "https://example.com/path?q=1&r=2#frag",
    "user@example.com", "# Heading\n\n* item one\n* item two\n\n```python\nprint('hi')\n```", "<div class=\"x\">text</div>", "!!!???...", "a--b__c==d",
    "“quoted” ‘single’ — dash – en … ellipsis", "(a) [b] {c} <d>", "~`!@#$%^&*()_+-=[]{}|;':\",./<>?",
    TAMIL, "கொ", "கொ", "கோ", "கோ", "கௌ", "கௌ", "ஸ்ரீ", "கூட்டம்", "இந்தியா 🇮🇳",
    "नमस्ते दुनिया", "مرحبا بالعالم", "שלום עולם", "你好，世界！", "こんにちは世界", "カタカナ", "안녕하세요 세계", "สวัสดีชาวโลก", "Привет, мир!", "Γειά σου Κόσμε",
    "한글 한글", "가", "각",
    "😀", "👍🏽", "👨‍👩‍👧‍👦", "❤️", "1️⃣", "🏳️‍🌈", "🇮🇳🇺🇸", "😀😃😄😁", "🧑‍💻 coding",
    "é", "café", "café", "Å", "Å", "Ω", "Ω", "ẛ̣", "ạ́", "ạ́", "̈́", "क़",
    "𝒜𝓑", "𠮷野家", "private", "à̀̀", "̀", "̀abc", " ̀", "é́",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", "<think>", "</think>", "<|endoftext|>", "<|im_end|><|im_end|>", "x<|im_end|>y",
    "<|im_end|", "<|im_end", "<im_end|>", "<|IM_END|>", "<|im_end|>\n", "text <|im_start|> text", "<tool_call>{}</tool_call>", "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>",
    "<|vision_start|><|image_pad|><|vision_end|>", "<|audio_start|>", "<tts_pad>", "<|audio_pad|>x", "<|", "|>", "<>", "<|>",
    "a" * 5000, " " * 3000, "ab" * 5000, "\n" * 1000, ("The quick brown fox jumps over the lazy dog. " * 200),
    "".join(random.Random(1).choice("abcdefghijklmnopqrstuvwxyz ,.\n") for _ in range(20000)),
]

POOLS = {
    "ascii": [chr(c) for c in range(0x20, 0x7F)],
    "space": [" ", " ", " ", "\t", "\n", "\r", "\r\n", " ", " ", "　", " ", "\u0085", "​"],
    "newline": ["\n", "\n\n", "\r\n", "\r"],
    "digit": list("0123456789") + list("٠١٢٣٤٥٦٧٨٩०१२३४५६७८९௦௧௨௩") + list("²³¹¼½Ⅷⅲ"),
    "latin": [chr(c) for c in range(0xC0, 0x250)],
    "tamil": [chr(c) for c in range(0x0B80, 0x0BFF) if len(chr(c).encode())],
    "devanagari": [chr(c) for c in range(0x0900, 0x097F)],
    "arabic": [chr(c) for c in range(0x0600, 0x06FF)],
    "cjk": [chr(c) for c in range(0x4E00, 0x4E80)] + [chr(c) for c in range(0x3040, 0x30FF)],
    "hangul": [chr(c) for c in range(0xAC00, 0xAC40)] + [chr(c) for c in range(0x1100, 0x1113)] + [chr(c) for c in range(0x1161, 0x1176)] + [chr(c) for c in range(0x11A8, 0x11C3)],
    "thai": [chr(c) for c in range(0x0E01, 0x0E5B)],
    "marks": [chr(c) for c in list(range(0x0300, 0x0370)) + list(range(0x0483, 0x0489)) + [0x200D, 0xFE0F, 0xFE0E, 0x20E3]],
    "emoji": [chr(c) for c in list(range(0x1F300, 0x1F5FF)) + list(range(0x1F600, 0x1F64F)) + list(range(0x1F900, 0x1F9FF)) + [0x2764, 0x2600, 0x2705]],
    "punct": list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + ["—", "–", "…", "“", "”", "‘", "’", "«", "»", "¿", "¡", "•", "·", "™", "©", "→"],
    "contr": ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL", "'", "ſ"],
    "nfc": ["Å", "Ω", "ẛ̣", "é", "ạ́", "ạ́", "ொ", "ொ", "கொ", "क़", "각", "ﬁ", "Å"],
    "special": ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "<|vision_start|>", "<|im_", "|>", "<|"],
}
WEIGHTS = {"ascii": 30, "space": 14, "newline": 4, "digit": 6, "latin": 5, "tamil": 7, "devanagari": 3, "arabic": 3, "cjk": 4, "hangul": 3, "thai": 2,
           "marks": 5, "emoji": 5, "punct": 8, "contr": 3, "nfc": 4, "special": 3}


def fuzz(rng, n, lo, hi):
    names = list(POOLS)
    w = [WEIGHTS[k] for k in names]
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(POOLS[rng.choices(names, w)[0]]) for _ in range(rng.randint(lo, hi))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/golden/tokenizer_golden.json")
    ap.add_argument("--fuzz", type=int, default=2500)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok_plain = AutoTokenizer.from_pretrained(a.model, split_special_tokens=True)  # added tokens treated as text
    bt = tok.backend_tokenizer
    rng = random.Random(20261109)
    texts = HAND + fuzz(rng, a.fuzz, 1, 40) + fuzz(rng, 150, 100, 1500)
    # the 7 audio / TTS control tokens that the GGUF flags as special but the checkpoint's tokenizer.json does not list
    extra = ["<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>"]
    added_strings = [a["content"] for a in json.load(open(os.path.join(a.model, "tokenizer.json"), encoding="utf-8"))["added_tokens"]] + extra
    cases = []
    for t in texts:
        norm = bt.normalizer.normalize_str(t)
        ids = tok(t, add_special_tokens=False)["input_ids"]
        ids_plain = tok_plain(t, add_special_tokens=False)["input_ids"]
        pieces = [p for p, _ in bt.pre_tokenizer.pre_tokenize_str(norm)]
        cases.append({"text": t, "norm": norm, "ids": ids, "ids_plain": ids_plain, "pieces": pieces, "decoded": tok.decode(ids),
                      "decoded_plain": tok.decode(ids_plain), "has_added": any(a in t for a in added_strings),
                      "has_extra_special": any(e in t for e in extra)})
    # decode of random id sequences, including special and unused ids
    vocab = len(tok)
    dec = []
    for _ in range(400):
        n = rng.randint(1, 30)
        ids = [rng.randrange(0, 248044) for _ in range(n)]
        dec.append({"ids": ids, "decoded": tok.decode(ids)})
    for _ in range(100):
        ids = [rng.choice([248044, 248045, 248046, 248058, 248068, 248070, 248076, 248077, 248200, 248319, rng.randrange(0, 248044)]) for _ in range(rng.randint(1, 10))]
        try:
            dec.append({"ids": ids, "decoded": tok.decode(ids), "mixed": True})
        except Exception as e:  # noqa: BLE001
            dec.append({"ids": ids, "decoded": None, "error": type(e).__name__, "mixed": True})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"hf_vocab_len": vocab, "cases": cases, "decode_cases": dec}, open(a.out, "w"), ensure_ascii=True)
    print(f"wrote {a.out}: {len(cases)} text cases, {len(dec)} decode cases, hf len(tok)={vocab}; "
          f"{sum(c['has_extra_special'] for c in cases)} cases mention the 7 extra special tokens; "
          f"{sum(1 for c in cases if c['ids'] != c['ids_plain'])} cases differ between special-aware and text-only encoding")


if __name__ == "__main__":
    main()
