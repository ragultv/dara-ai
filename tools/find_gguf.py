"""Find a pinned Q4_K_M GGUF for a model and print repo / revision / file / size. Usage: python -I tools/find_gguf.py <repo> ..."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402

for repo in sys.argv[1:]:
    try:
        api = json.loads(gp.get(f"https://huggingface.co/api/models/{repo}?blobs=true", 0, 1 << 24))
    except Exception as e:  # noqa: BLE001
        print(f"{repo}: not accessible ({type(e).__name__})")
        continue
    files = [(s["rfilename"], s.get("size")) for s in api["siblings"] if s["rfilename"].endswith(".gguf")]
    q4 = [(n, sz) for n, sz in files if "Q4_K_M" in n and "mmproj" not in n]
    print(f"{repo} @ {api['sha']}: {len(files)} gguf files; Q4_K_M: {[(n, round((sz or 0) / 2**30, 2)) for n, sz in q4]}")
