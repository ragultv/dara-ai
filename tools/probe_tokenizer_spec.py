"""What the checkpoint's tokenizer does exactly (test oracle) and what the GGUF carries (runtime source)."""
import json
import sys

import numpy as np

tj = json.load(open("models/qwen3.5-4b/tokenizer.json", encoding="utf-8"))
print("model type:", tj["model"]["type"], "| vocab:", len(tj["model"]["vocab"]), "| merges:", len(tj["model"]["merges"]),
      "| byte_fallback:", tj["model"].get("byte_fallback"), "| ignore_merges:", tj["model"].get("ignore_merges"))
print("normalizer:", json.dumps(tj.get("normalizer"))[:300])
print("pre_tokenizer:", json.dumps(tj.get("pre_tokenizer"), ensure_ascii=False)[:900])
print("post_processor:", json.dumps(tj.get("post_processor"))[:200])
print("decoder:", json.dumps(tj.get("decoder"))[:300])
added = tj["added_tokens"]
print("added tokens:", len(added), "| special:", sum(1 for a in added if a["special"]), "| ids", min(a["id"] for a in added), "..", max(a["id"] for a in added))
print("  first few:", [(a["id"], a["content"], a["special"]) for a in added[:6]])
print("  non-special added:", [(a["id"], a["content"]) for a in added if not a["special"]][:12])
tc = json.load(open("models/qwen3.5-4b/tokenizer_config.json", encoding="utf-8"))
print("tokenizer_config:", {k: tc.get(k) for k in ["add_bos_token", "add_prefix_space", "bos_token", "eos_token", "pad_token", "unk_token", "clean_up_tokenization_spaces", "split_special_tokens", "model_max_length"]})

sys.path.insert(0, "tools")
from gguf import GGUFReader

r = GGUFReader("models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf")
fields = {k: v for k, v in r.fields.items() if k.startswith("tokenizer.")}
print("GGUF tokenizer keys:", sorted(fields))
tt = r.fields["tokenizer.ggml.token_type"]
types = np.array(tt.parts[tt.data[0]] if False else [tt.parts[i][0] for i in tt.data])
import collections
print("token_type histogram:", dict(collections.Counter(types.tolist())))
toks = r.fields["tokenizer.ggml.tokens"]
strs = [bytes(toks.parts[i]).decode("utf-8", "replace") for i in toks.data]
special_ids = [i for i, t in enumerate(types.tolist()) if t in (3, 4)]
print("GGUF control/user-defined tokens:", len(special_ids), "first:", [(i, strs[i], int(types[i])) for i in special_ids[:6]])
hf_added = {a["id"]: a for a in added}
print("same id set as HF added_tokens:", sorted(special_ids) == sorted(hf_added))
print("types of HF non-special added tokens in GGUF:", collections.Counter(int(types[a['id']]) for a in added if not a['special']))
mg = r.fields["tokenizer.ggml.merges"]
merges = [bytes(mg.parts[i]).decode("utf-8", "replace") for i in mg.data]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("GGUF merges:", len(merges), "first 3:", merges[:3], "| HF merges first 3:", tj["model"]["merges"][:3])
print("merges identical (as 'a b' strings):", merges == [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]])
print("GGUF tokens identical to HF base vocab for ids < 248044:", all(strs[i] == t for t, i in tj["model"]["vocab"].items() if i < len(strs) and False) or "see below")
hv = tj["model"]["vocab"]
bad = [(t, i) for t, i in hv.items() if strs[i] != t]
print("vocab entries whose GGUF string differs from HF:", len(bad), bad[:3])
only_gguf = [i for i in special_ids if i not in hf_added]
only_hf = [i for i in hf_added if i not in special_ids]
print("control/user-defined in GGUF but NOT HF added tokens:", [(i, strs[i], int(types[i])) for i in only_gguf])
print("HF added tokens NOT control/user-defined in GGUF:", [(i, hf_added[i]['content']) for i in only_hf])
print("GGUF types of all 26 HF added tokens:", [(i, int(types[i])) for i in sorted(hf_added)])
for k in ["tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id", "tokenizer.ggml.unknown_token_id"]:
    f = r.fields.get(k)
    print(k, "=", None if f is None else f.parts[f.data[0]].tolist())
