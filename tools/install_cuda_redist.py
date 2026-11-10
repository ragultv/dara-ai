"""Install a minimal CUDA toolkit from NVIDIA's redistributable archives (no installer, no admin).

Every archive is verified against the SHA-256 in NVIDIA's redistrib JSON before extraction.
Pinned to one toolkit release; the driver must support it (check `nvidia-smi`).

Usage: python -I tools/install_cuda_redist.py [--version 13.1.1] [--dest D:/tools/cuda]
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = ["cuda_nvcc", "libnvvm", "cuda_cudart", "cuda_cccl", "cuda_crt", "cuda_nvml_dev",
              "cuda_profiler_api", "cuda_sanitizer_api", "visual_studio_integration"]


def fetch(url, dest=None):
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if dest:
        Path(dest).write_bytes(data)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="13.1.1")
    ap.add_argument("--dest", default="D:/tools/cuda")
    a = ap.parse_args()

    root = Path(a.dest) / a.version
    dl = Path(a.dest) / "_downloads" / a.version
    dl.mkdir(parents=True, exist_ok=True)
    index = json.loads(fetch(f"{BASE}redistrib_{a.version}.json"))

    for comp in COMPONENTS:
        entry = index[comp]["windows-x86_64"]
        rel, want = entry["relative_path"], entry["sha256"]
        zpath = dl / Path(rel).name
        if not zpath.exists() or hashlib.sha256(zpath.read_bytes()).hexdigest() != want:
            fetch(BASE + rel, zpath)
        got = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if got != want:
            zpath.unlink()
            sys.exit(f"CHECKSUM MISMATCH for {comp}: expected {want}, got {got}")
        print(f"verified {comp} {index[comp]['version']}")
        with zipfile.ZipFile(zpath) as z:
            top = z.namelist()[0].split("/")[0]
            for m in z.infolist():
                if m.is_dir():
                    continue
                target = root / Path(*Path(m.filename).parts[1:])
                if not target.resolve().is_relative_to(root.resolve()):
                    sys.exit(f"unsafe path in {zpath.name}: {m.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    print(f"installed to {root}")


if __name__ == "__main__":
    main()
