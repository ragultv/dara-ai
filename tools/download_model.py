"""Download a pinned model snapshot into models/<name>/ with plain HTTP range requests: parallel 16 MiB chunks, resumable,
every file verified against the size the hub lists. (huggingface_hub's Xet backend stalls on this machine; plain ranges to
the normal resolve URL have always worked here.)

Usage: python -I tools/download_model.py --repo Qwen/Qwen3.5-4B --rev <sha> --dest models/qwen3.5-4b --include '*.safetensors' 'config.json'
       python -I tools/download_model.py --repo unsloth/Qwen3.5-4B-GGUF --rev <sha> --dest models/qwen3.5-4b-gguf --include Qwen3.5-4B-Q4_K_M.gguf
Progress is recorded in <file>.parts.json so an interrupted run resumes without re-fetching finished chunks.
"""
import argparse
import concurrent.futures as cf
import fnmatch
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
import gguf_probe as gp  # noqa: E402

CHUNK = 16 << 20


def fetch_file(url, target, size, workers):
    parts_path = target + ".parts.json"
    done = set(json.load(open(parts_path))) if os.path.exists(parts_path) else set()
    n_chunks = (size + CHUNK - 1) // CHUNK
    mode = "r+b" if os.path.exists(target) and os.path.getsize(target) == size else "wb"
    with open(target, mode) as f:
        if mode == "wb":
            f.truncate(size)
            done = set()
    lock = threading.Lock()
    t0 = time.time()
    start_bytes = len(done) * CHUNK

    def work(i):
        lo = i * CHUNK
        hi = min(size, lo + CHUNK) - 1
        data = gp.get(url, lo, hi, tries=10)
        if len(data) != hi - lo + 1:
            raise IOError(f"chunk {i}: got {len(data)} bytes, wanted {hi - lo + 1}")
        with lock:
            with open(target, "r+b") as f:
                f.seek(lo)
                f.write(data)
            done.add(i)
            json.dump(sorted(done), open(parts_path, "w"))
            got = len(done) * CHUNK
            rate = (got - start_bytes) / max(1e-9, time.time() - t0) / 2**20
            print(f"  {os.path.basename(target)}: {len(done)}/{n_chunks} chunks ({min(got, size) / 2**30:.2f}/{size / 2**30:.2f} GiB, {rate:.1f} MiB/s)", flush=True)

    todo = [i for i in range(n_chunks) if i not in done]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in cf.as_completed([ex.submit(work, i) for i in todo]):
            fut.result()
    if os.path.getsize(target) != size:
        raise IOError(f"{target}: size {os.path.getsize(target)} != {size}")
    os.remove(parts_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--include", nargs="*", default=["*.safetensors", "*.json", "*.txt"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    info = json.loads(gp.get(f"https://huggingface.co/api/models/{a.repo}/revision/{a.rev}?blobs=true", 0, 1 << 24))
    files = [(s["rfilename"], s.get("size")) for s in info["siblings"] if any(fnmatch.fnmatch(s["rfilename"], p) for p in a.include)]
    os.makedirs(a.dest, exist_ok=True)
    for name, size in files:
        target = os.path.join(a.dest, name)
        if size and os.path.exists(target) and os.path.getsize(target) == size and not os.path.exists(target + ".parts.json"):
            print(f"have {name} ({size / 2**30:.2f} GiB)", flush=True)
            continue
        print(f"downloading {name} ({(size or 0) / 2**30:.2f} GiB)", flush=True)
        url = f"https://huggingface.co/{a.repo}/resolve/{a.rev}/{name}"
        if not size:  # small files without a listed size: single request
            with open(target, "wb") as f:
                f.write(gp.get(url, 0, (1 << 26) - 1, tries=10))
            continue
        for attempt in range(1, 6):
            try:
                fetch_file(url, target, size, a.workers)
                break
            except Exception as e:  # noqa: BLE001  transient network errors: resume from the recorded chunks
                print(f"  attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(5 * attempt)
        else:
            sys.exit(f"giving up on {name}")
        print(f"ok {name} {os.path.getsize(target)} bytes", flush=True)
    print("DOWNLOAD COMPLETE", flush=True)


if __name__ == "__main__":
    main()
