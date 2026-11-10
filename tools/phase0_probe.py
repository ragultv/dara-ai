"""Phase 0 environment probe: records GPU, toolchain and reference-package state as JSON.

Usage: python tools/phase0_probe.py [out.json]
"""
import importlib.metadata as md
import json
import shutil
import subprocess
import sys

PACKAGES = ["torch", "transformers", "safetensors", "numpy", "pillow", "pytest"]
TOOLS = ["cmake", "ninja", "git", "nvcc", "cl", "nsys", "ncu", "compute-sanitizer"]


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def gpus():
    out = run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
               "--format=csv,noheader"])
    rows = []
    for line in (out or "").splitlines():
        name, cc, mem, drv = [x.strip() for x in line.split(",")]
        major, minor = (int(x) for x in cc.split("."))
        rows.append({
            "name": name, "compute_cap": cc, "memory": mem, "driver": drv,
            "bf16_tensor_cores": major >= 8,
            "fp8": (major, minor) >= (8, 9),
        })
    return rows


def pkg(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    report = {
        "python": sys.version.split()[0],
        "gpus": gpus(),
        "tools": {t: shutil.which(t) for t in TOOLS},
        "packages": {p: pkg(p) for p in PACKAGES},
    }
    report["missing"] = sorted(
        [k for k, v in report["tools"].items() if not v and k in ("ninja", "nvcc")] +
        [k for k, v in report["packages"].items() if not v])
    text = json.dumps(report, indent=2)
    print(text)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
