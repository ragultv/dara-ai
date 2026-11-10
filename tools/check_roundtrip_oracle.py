import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
g = json.load(open("build/golden/tokenizer_golden.json"))
bad = [(i, c) for i, c in enumerate(g["cases"]) if c["decoded"] != c["norm"] and c["ids"] == c["ids_plain"]]
print("cases where HF decode(HF ids) != HF normalized text:", len(bad))
for i, c in bad[:5]:
    d, n = c["decoded"], c["norm"]
    k = next((j for j in range(min(len(d), len(n))) if d[j] != n[j]), min(len(d), len(n)))
    print(f"case {i}: len decoded {len(d)}, len norm {len(n)}, first difference at index {k}:")
    print("   decoded:", [hex(ord(x)) for x in d[max(0, k - 2):k + 4]])
    print("   norm   :", [hex(ord(x)) for x in n[max(0, k - 2):k + 4]])
