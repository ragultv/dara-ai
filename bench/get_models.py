"""Download the exact model files the benchmarks use (pinned revisions, so every machine tests the same weights).

Usage: python -I bench/get_models.py [--models 4b nano 9b] [--no-vision] [--dest models]
Sizes: 4B 2.6 GB (+0.6 GB vision), Nano 2.9 GB (+0.9 GB vision), 9B 5.3 GB (+0.9 GB vision).
"""
import argparse, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINNED = {   # model -> (repo, revision, subfolder, text GGUF)
    "4b": ("unsloth/Qwen3.5-4B-GGUF", "e87f176479d0855a907a41277aca2f8ee7a09523", "qwen3.5-4b-gguf", "Qwen3.5-4B-Q4_K_M.gguf"),
    "9b": ("unsloth/Qwen3.5-9B-GGUF", "3885219b6810b007914f3a7950a8d1b469d598a5", "qwen3.5-9b-gguf", "Qwen3.5-9B-Q4_K_M.gguf"),
    "nano": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2", "gemma4-e2b-gguf", "gemma-4-E2B-it-Q4_K_M.gguf"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["4b", "nano", "9b"])
    ap.add_argument("--no-vision", action="store_true", help="skip the image encoders (mmproj files)")
    ap.add_argument("--dest", default=os.path.join(ROOT, "models"))
    a = ap.parse_args()
    for m in a.models:
        repo, rev, sub, fname = PINNED[m]
        files = [fname] + ([] if a.no_vision else ["mmproj-F16.gguf"])
        dest = os.path.join(a.dest, sub)
        os.makedirs(dest, exist_ok=True)
        print(f"== {m}: {repo}@{rev[:8]} -> {dest}: {', '.join(files)}", flush=True)
        rc = subprocess.call([sys.executable, "-I", os.path.join(ROOT, "tools", "download_model.py"), "--repo", repo, "--rev", rev, "--dest", dest,
                              "--include", *files])
        if rc != 0: sys.exit(f"download failed for {m}")
    print("done")


if __name__ == "__main__":
    main()
