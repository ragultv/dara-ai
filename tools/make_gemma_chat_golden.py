"""Reference data for the Gemma 4 chat template from the Hugging Face renderer (the TEST ORACLE).

Random conversations over system/developer/user/assistant text messages (padded whitespace, empty content, unicode, consecutive
assistant messages, assistant messages carrying a thought channel, thinking mode, with/without generation prompt). For each: HF's
rendered string and the ids of that string. Output: build/golden/gemma_chat_golden.json
Usage: python -I tools/make_gemma_chat_golden.py --model models/gemma4-e2b
"""
import argparse
import json
import os
import random
import sys

sys.modules["torchao"] = None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SNIPPETS = ["Hello", "What is the capital of France?", "  padded with spaces  ", "\n\nleading and trailing newlines\n\n", "", "   ", "\t tabs \t",
            "Line one\nLine two\n  indented", "தமிழ் மொழி", "你好，世界", "emoji 😀👍🏽", "def f(x):\n    return x*2\n", "a  b   c", "Ünïcödé café",
            "<|channel>thought\nhidden reasoning<channel|>Visible answer", "Answer<|channel>thought\nnever closed", "x<channel|>y", " nbsp ",
            " em space ", "ends with newline\n", "{\"json\": [1, 2, 3]}", "a" * 300]


def conv(rng):
    n = rng.randint(1, 7)
    msgs = []
    if rng.random() < 0.35:
        msgs.append({"role": rng.choice(["system", "developer"]), "content": rng.choice(SNIPPETS)})
    for _ in range(n):
        role = rng.choice(["user", "user", "assistant", "assistant"])
        if role == "assistant" and rng.random() < 0.4:
            content = rng.choice(SNIPPETS[14:16] + SNIPPETS[:6])
        else:
            content = rng.choice(SNIPPETS)
        msgs.append({"role": role, "content": content})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/golden/gemma_chat_golden.json")
    ap.add_argument("--n", type=int, default=600)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    rng = random.Random(20261111)
    cases = []
    for _ in range(a.n):
        msgs = conv(rng)
        gen = rng.random() < 0.8
        think = rng.random() < 0.25
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=think)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        cases.append({"messages": msgs, "add_generation_prompt": gen, "enable_thinking": think, "text": text, "ids": ids})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"cases": cases}, open(a.out, "w"), ensure_ascii=True)
    print(f"wrote {a.out}: {len(cases)} conversations")


if __name__ == "__main__":
    main()
