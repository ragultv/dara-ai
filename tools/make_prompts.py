"""Tokenize a fixed prompt set with the pinned Hugging Face tokenizer (Dara has no tokenizer yet; ids are fed directly).
Writes build/prompts/p<k>.tokens (space-separated ids) and p<k>.txt (the text).
Usage: python -I tools/make_prompts.py --model models/qwen3.5-4b
"""
import argparse
import os
import sys

sys.modules["torchao"] = None  # incompatible with torch 2.5.1; not needed here

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain in one sentence why the sky is blue.",
    "1, 1, 2, 3, 5, 8, 13, 21,",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/prompts")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    for k, text in enumerate(PROMPTS):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with open(f"{a.out}/p{k}.tokens", "w") as f:
            f.write(" ".join(map(str, ids)) + "\n")
        with open(f"{a.out}/p{k}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"p{k}: {len(ids)} tokens  {ids}  {text!r}")


if __name__ == "__main__":
    main()
