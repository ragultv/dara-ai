"""Shared settings for the benchmark scripts: where dara.exe, llama.cpp and the models are, and a record of the machine the numbers come from.

Every path can be given on the command line or through an environment variable, so the scripts run the same on any machine:
  --dara-exe   / DARA_EXE        dara.exe from a release zip (default: build/cuda/dara.exe of a local build)
  --llama-dir  / DARA_LLAMA_DIR  folder of a llama.cpp CUDA build (llama-completion.exe, llama-server.exe)
  --models-dir / DARA_MODELS     folder with the model subfolders (default: <repo>/models)
"""
import json, os, platform, re, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LLAMA_DIR = r"D:\oprel\bin\llama_cpp\b12189\cuda"

MODEL_FILES = {   # model -> (gguf, mmproj) relative to the models folder
    "4b": ("qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf", "qwen3.5-4b-gguf/mmproj-F16.gguf"),
    "9b": ("qwen3.5-9b-gguf/Qwen3.5-9B-Q4_K_M.gguf", "qwen3.5-9b-gguf/mmproj-F16.gguf"),
    "nano": ("gemma4-e2b-gguf/gemma-4-E2B-it-Q4_K_M.gguf", "gemma4-e2b-gguf/mmproj-F16.gguf"),
}
FAMILY = {"4b": "qwen", "9b": "qwen", "nano": "gemma"}


def add_args(ap):
    ap.add_argument("--dara-exe", default=os.environ.get("DARA_EXE", os.path.join(ROOT, "build", "cuda", "dara.exe")))
    ap.add_argument("--llama-dir", default=os.environ.get("DARA_LLAMA_DIR", DEFAULT_LLAMA_DIR))
    ap.add_argument("--models-dir", default=os.environ.get("DARA_MODELS", os.path.join(ROOT, "models")))
    ap.add_argument("--llama-ngl", nargs="*", default=[],
                    help="llama.cpp GPU layers per model, e.g. 9b=24 (default: 99 = all, and 'auto' for 9b: llama.cpp fits the layers to the free VRAM)")


def ngl_for(a, model):
    over = dict(kv.split("=", 1) for kv in a.llama_ngl)
    return over.get(model, "auto" if model == "9b" else "99")


def ngl_args(a, model):
    n = ngl_for(a, model)
    return [] if n == "auto" else ["-ngl", n]


def gguf(a, model): return os.path.join(a.models_dir, MODEL_FILES[model][0])


def mmproj(a, model): return os.path.join(a.models_dir, MODEL_FILES[model][1])


def check_paths(a, models, need_llama=True, need_mmproj=False):
    missing = [p for p in [a.dara_exe] + [gguf(a, m) for m in models] + ([mmproj(a, m) for m in models] if need_mmproj else []) if not os.path.exists(p)]
    if need_llama and not os.path.isdir(a.llama_dir): missing.append(a.llama_dir)
    if missing:
        raise SystemExit("missing (see docs/BENCHMARKING.md):\n  " + "\n  ".join(missing))


def machine_info():
    """GPU, driver, VRAM, CPU, RAM and power source: stored with every result so numbers from different laptops can be compared."""
    info = {"os": platform.platform(), "cpu": platform.processor()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout.strip().splitlines()
        if q: info["gpu"], info["driver"], info["vram"], info["gpu_power_limit"] = [x.strip() for x in q[0].split(",")][:4]
    except Exception:
        info["gpu"] = "nvidia-smi not available"
    if os.name == "nt":
        ps = ("$b = Get-CimInstance Win32_Battery; $m = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
              "$c = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name; "
              "Write-Output ('{0}|{1}|{2}' -f ($b.BatteryStatus -join ','), $m, $c)")
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
            batt, mem, cpu = (out.split("|") + ["", "", ""])[:3]
            info["ram_gb"] = round(int(mem) / 2**30, 1) if mem.isdigit() else None
            if cpu: info["cpu"] = cpu.strip()
            info["power"] = {"": "no battery (desktop)", "2": "AC"}.get(batt, "BATTERY (status " + batt + ")")
        except Exception:
            pass
    return info


def print_machine(info):
    print("machine: " + " | ".join(f"{k}={v}" for k, v in info.items()), flush=True)
    if str(info.get("power", "")).startswith("BATTERY"):
        print("WARNING: running on battery -- plug in the charger; laptop GPUs throttle on battery and the numbers are not comparable.", flush=True)
