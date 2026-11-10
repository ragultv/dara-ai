"""Reference data for the Gemma 4 tokenizer from the pinned Hugging Face tokenizer (the TEST ORACLE; the runtime never uses it).

Per text case: ids with special tokens recognized (HF default), ids with special tokens treated as text (split_special_tokens),
and HF's decode. Plus random id sequences with HF's decode. Texts: the hand-written edge cases and the seeded fuzz pools of
tools/make_tokenizer_golden.py (scripts, marks, emoji, whitespace, newlines), extended with Gemma's own special-token strings.
Output: build/golden/gemma_tokenizer_golden.json   Usage: python -I tools/make_gemma_tokenizer_golden.py --model models/gemma4-e2b
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.modules["torchao"] = None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import make_tokenizer_golden as q  # noqa: E402  (reuses its HAND cases and fuzz pools)

GEMMA_SPECIAL = ["<bos>", "<eos>", "<pad>", "<unk>", "<mask>", "<turn|>", "<|turn>", "<|channel>", "<channel|>", "<|think|>", "<|image|>", "<|audio|>",
                 "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", "<|\"|>", "<|", "|>", "<turn", "<0x41>", "<0x0A>"]
EXTRA_HAND = [
    "<bos>Hello", "Hello<eos>", "<|turn>user\nHi<turn|>\n<|turn>model\n", "a<turn|>b", "<0x41>", "<0xZZ>", "▁", "▁▁", "a▁b", " ▁ ", "▁▁x",
    "The capital of France is", "  two leading", "trailing  ", "a b", "\x00\x01\x02", "\u0085", "퟿", "\U0001F600\U0001F601",
    "\U0010FFFF", "�", "ab�cd", "x" * 3000, "\n" * 300, "\t" * 70, " " * 200,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/golden/gemma_tokenizer_golden.json")
    ap.add_argument("--fuzz", type=int, default=2000)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    tok_plain = AutoTokenizer.from_pretrained(a.model, split_special_tokens=True)
    q.POOLS["special"] = GEMMA_SPECIAL
    q.POOLS["nfc"] = ["Å", "Ω", "é", "ạ́", "ொ", "கொ", "각"]  # Gemma does not normalize; keep composed and decomposed forms in the mix
    rng = random.Random(20261110)
    texts = q.HAND + EXTRA_HAND + q.fuzz(rng, a.fuzz, 1, 40) + q.fuzz(rng, 150, 100, 1500)
    added = [t["content"] for t in json.load(open(os.path.join(a.model, "tokenizer.json"), encoding="utf-8"))["added_tokens"]]
    cases = []
    for t in texts:
        t = t.encode("utf-8", errors="replace").decode("utf-8")  # lone surrogates cannot be written to JSON/UTF-8
        ids = tok(t, add_special_tokens=False)["input_ids"]
        ids_plain = tok_plain(t, add_special_tokens=False)["input_ids"]
        cases.append({"text": t, "ids": ids, "ids_plain": ids_plain, "decoded": tok.decode(ids), "decoded_plain": tok.decode(ids_plain),
                      "has_added": any(x in t for x in added)})
    dec = []
    for _ in range(500):
        ids = [rng.randrange(0, 262144) for _ in range(rng.randint(1, 30))]
        dec.append({"ids": ids, "decoded": tok.decode(ids)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"cases": cases, "decode_cases": dec, "added": added}, open(a.out, "w"), ensure_ascii=True)
    print(f"wrote {a.out}: {len(cases)} text cases, {len(dec)} decode cases; "
          f"{sum(1 for c in cases if c['ids'] != c['ids_plain'])} cases differ between special-aware and text-only encoding")


if __name__ == "__main__":
    main()
