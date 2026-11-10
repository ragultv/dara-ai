import sys
sys.modules["torchao"] = None
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/qwen3.5-4b")
import json
tj = json.load(open("models/qwen3.5-4b/tokenizer.json", encoding="utf-8"))
print("added-token flags (id: special/normalized/lstrip/rstrip/single_word):")
for a in tj["added_tokens"]:
    print(f"  {a['id']} {a['content']!r}: {a['special']}/{a['normalized']}/{a['lstrip']}/{a['rstrip']}/{a['single_word']}")
cases = {"fim": "<|fim_prefix|≯x", "think": "<think≯x", "im_end": "<|im_end|≯x"}
for k, t in cases.items():
    open(f"build/probe_{k}.txt", "w", encoding="utf-8").write(t)
    print(k, "HF ids:", tok(t, add_special_tokens=False)["input_ids"])
