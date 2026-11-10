import sys
import os

sys.modules["torchao"] = None
from transformers import AutoTokenizer

d = sys.argv[1]
print("files:", sorted(f for f in os.listdir(d) if not f.endswith(".safetensors") and not f.startswith(".")))
tok = AutoTokenizer.from_pretrained(d)
print("type:", type(tok).__name__, "| vocab size:", len(tok), "| is_fast:", getattr(tok, "is_fast", None))
for text in ["The capital of France is", "hello"]:
    enc = tok(text, add_special_tokens=False)
    print(repr(text), "->", enc["input_ids"], "| encode():", tok.encode(text, add_special_tokens=False), "| tokens:", tok.tokenize(text)[:8])
print("special:", tok.bos_token, tok.eos_token, tok.pad_token)
