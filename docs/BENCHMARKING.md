# Benchmarking Dara on another Windows laptop

This guide takes a Windows laptop with an NVIDIA GPU from nothing to a set of result files that can be compared with the numbers from
the development laptop (GTX 1650 4 GB). It covers getting `dara.exe`, the models, a matching llama.cpp build, and running the three benchmarks:

| Benchmark | What it measures | Script |
|---|---|---|
| Text speed | Decode tokens/s and time to first token on 16 real tasks, Dara vs llama.cpp | `bench\task_bench.py` |
| Text speed with Flash | The same, with Dara's speculative drafter | `bench\task_bench.py --dara-args ...` |
| Image time to first token | Image preprocessing + encoding + prompt, Dara vs llama-server | `bench\image_ttft.py` |

Every result file records the machine (GPU, driver, VRAM, CPU, RAM, and whether it was on AC power), so results from different laptops can
be compared directly.

> **First run on these GPUs.** Dara is only tested on a GTX 1650 (Turing, sm_75). The release also contains code for RTX 30xx (sm_86),
> RTX 40xx (sm_89) and RTX 50xx (sm_120), but none of it has run on real hardware yet. If something fails, the console output is exactly what
> is needed to fix it; see [Troubleshooting](#troubleshooting).

## 1. What you need

| | Requirement | How to check |
|---|---|---|
| GPU | NVIDIA GTX 16xx / RTX 20xx or newer, 4 GB VRAM or more | Device Manager → Display adapters |
| Driver | NVIDIA driver **R580 or newer** (CUDA 13 support) | `nvidia-smi`: the header must say `CUDA Version: 13.0` or higher. If it is older, install the current Game Ready or Studio driver from nvidia.com. |
| Python | 3.10 or newer (only for the download and benchmark scripts; the standard library is enough) | `python --version` |
| Git | any (or download the repository as a zip) | `git --version` |
| Disk | about 15 GB free (models 12 GB, llama.cpp 0.5 GB) | |
| Power | **charger plugged in** for the whole run | the scripts print a warning on battery |

## 2. Get the code and `dara.exe`

Open **PowerShell** and pick a folder, for example `C:\dara`:

```powershell
git clone https://github.com/ragultv/dara-ai.git C:\dara
cd C:\dara
```

Then download the prebuilt `dara.exe`:

1. Open **https://github.com/ragultv/dara-ai/releases** and take the newest release (named `Dara bNNN`).
2. Download `dara-bNNN-bin-win-cuda-13.1.1-x64.zip` (and optionally the `.sha256` file next to it).
3. Unzip it into `C:\dara\bin\`, so that `C:\dara\bin\dara.exe` and `C:\dara\bin\flash\` exist.

Check that it starts:

```powershell
.\bin\dara.exe
# prints: usage: dara --gguf <model.gguf> [options] ...
```

`dara.exe` is a single file. The CUDA runtime is built in, so there are no DLLs to copy. The release is not code-signed, so Windows SmartScreen
may ask the first time: choose **More info → Run anyway**.

To verify the download (optional), compare the hash with the `.sha256` file:

```powershell
Get-FileHash .\dara-bNNN-bin-win-cuda-13.1.1-x64.zip -Algorithm SHA256
```

Keep the checkout and the release from the same build: `git log -1` should match the commit named in the release notes. If they differ,
`git checkout <that commit>`.

## 3. Get the models

The benchmark uses exactly the same model files on every machine (pinned Hugging Face revisions):

```powershell
python -I bench\get_models.py --models 4b nano          # 4B + Nano with their image encoders, about 7 GB
python -I bench\get_models.py --models 9b               # optional: 9B, about 6.2 GB
```

The files go to `C:\dara\models\...`. Downloads resume if interrupted; just run the same command again.

## 4. The Flash drafters (already in the zip)

Flash (Dara's speculative decoding) needs one trained drafter file per model. Every release zip includes them, so after step 2 they are here:

| File | Size | For |
|---|---|---|
| `C:\dara\bin\flash\drafter_4b_v2.dflash` | 83 MB | Core 4B |
| `C:\dara\bin\flash\drafter_nano_v2.dflash` | 35 MB | Nano |

## 5. Get llama.cpp (for the comparison)

The reference numbers use llama.cpp **build b12189**, the CUDA build for Windows:

1. Open **https://github.com/ggml-org/llama.cpp/releases/tag/b12189**.
2. Download the Windows x64 CUDA build: the asset whose name contains `bin-win-cuda` and `x64`. Also download the matching CUDA runtime
   asset (its name starts with `cudart-llama-bin-win-cuda`) if the release has one.
3. Unzip both into the same folder, for example `C:\llama.cpp\`. That folder must contain `llama-completion.exe` and `llama-server.exe`.

If b12189 is gone, take the newest release with the same kind of asset and note its build number with your results.

Point the scripts at it once per PowerShell window (or pass `--llama-dir` every time):

```powershell
$env:DARA_LLAMA_DIR = "C:\llama.cpp"
$env:DARA_EXE = "C:\dara\bin\dara.exe"
```

## 6. Before you run

* Plug in the charger. Laptop GPUs clock down on battery; the scripts mark battery runs in the results and print a warning.
* Windows power mode: **Settings → System → Power → Best performance**. If the laptop has a vendor tool (Armoury Crate, Lenovo Vantage, OMEN Gaming
  Hub, ...), choose its performance profile and, if offered, the **dGPU-only / Ultimate** GPU mode.
* Close browsers, games and anything else that uses the GPU. `nvidia-smi` should show almost no memory in use.
* Do not let the laptop sleep during a run (the full run takes 30-60 minutes). On some laptops the GPU does not come back after Modern Standby;
  if a run stops making progress, check `nvidia-smi` and reboot if it reports an error.

## 7. Run the benchmarks

All commands run from `C:\dara` in PowerShell. Each prints a table as it goes and writes a JSON file with every number and the machine details.

### 7.1 Text speed, Dara vs llama.cpp (no Flash)

```powershell
python -I bench\task_bench.py --models 4b nano --out build\bench\tasks_plain
```

About 20 minutes. Per task it prints the llama.cpp and Dara decode speeds, their ratio, and time to first token, then geometric means per category.

### 7.2 Text speed with Flash

The drafter is model-specific, so run each model on its own:

```powershell
python -I bench\task_bench.py --models 4b   --out build\bench\tasks_flash_4b   --dara-args "--flash bin\flash\drafter_4b_v2.dflash --spec 4 --flash-min-p 0.6"
python -I bench\task_bench.py --models nano --out build\bench\tasks_flash_nano --dara-args "--flash bin\flash\drafter_nano_v2.dflash --spec 4 --flash-min-p 0.6"
```

### 7.3 The 9B (optional)

```powershell
python -I bench\task_bench.py --models 9b --out build\bench\tasks_9b
```

The 9B does not fit in 4-6 GB of VRAM. Dara places what fits on the GPU and runs the rest on the CPU. llama.cpp gets `-ngl auto`, so it fits
its own layer count to the free VRAM. To force a layer count (for example to match an earlier run), add `--llama-ngl 9b=24`.

### 7.4 Image time to first token

```powershell
python -I bench\image_ttft.py --models 4b nano --out build\bench\image_ttft.json
python -I bench\image_ttft.py --models 9b      --out build\bench\image_ttft_9b.json     # optional
```

This uses the two pictures in `bench\assets\`: a 1920x1080 photo, and the same photo after oprel-SDK's preprocessing (shrunk to at most 409,600
pixels). It reports Dara (preprocessing + encoding + prompt) and llama-server (request to first streamed token), with the model already loaded.
All answers should be `INVOICE 7731`.

### 7.5 How much context fits (one line per model)

```powershell
.\bin\dara.exe --gguf models\qwen3.5-4b-gguf\Qwen3.5-4B-Q4_K_M.gguf --chat "Hi" --new 8 | Select-String "context:|VRAM-resident"
.\bin\dara.exe --gguf models\gemma4-e2b-gguf\gemma-4-E2B-it-Q4_K_M.gguf --chat "Hi" --new 8 | Select-String "context:|VRAM-resident"
.\bin\dara.exe --gguf models\qwen3.5-9b-gguf\Qwen3.5-9B-Q4_K_M.gguf --chat "Hi" --new 8 | Select-String "context:|VRAM-resident"
```

`context:` is the largest context Dara chose automatically for this GPU; `VRAM-resident layers` shows how many layers are on the GPU.

## 8. What to send back

Zip the `build\bench` folder and send it. The important files are:

| File | Contents |
|---|---|
| `build\bench\tasks_plain\task_bench.json` | text speed, no Flash |
| `build\bench\tasks_flash_4b\task_bench.json`, `tasks_flash_nano\...` | text speed with Flash |
| `build\bench\tasks_9b\task_bench.json` | 9B (if run) |
| `build\bench\image_ttft*.json` | image time to first token |

Also paste the console output of 7.5 and the output of `nvidia-smi`.

## 9. Reference numbers (development laptop)

GTX 1650 4 GB (sm_75), Ryzen 5 5600H, on AC power, Dara vs llama.cpp b12189:

| | 4B | Nano | 9B |
|---|---|---|---|
| Text decode, Flash, geometric mean of 16 tasks | **1.69x** llama.cpp | **1.50x** | not measured |
| Image TTFT, oprel-preprocessed photo | **2.8 s** vs 7.7 s | **2.2 s** vs 2.5 s | **4.1 s** vs 16.1 s |
| Image TTFT, full 1920x1080 photo | 8.7 s (2.8 s with `--image-max-tokens 400`) vs 48.4 s | 2.2 s vs 16.1 s | 10.8 s vs 91.1 s |
| Auto context | 21.5K (10K with Flash) | 131K | 32K, 17 of 32 layers on the GPU |

A faster GPU raises both runtimes. The **ratios** are what to compare between laptops.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `no CUDA-capable device`, `driver version is insufficient`, or nothing happens | Driver older than R580: update the NVIDIA driver. On laptops with two GPUs, make sure the NVIDIA GPU is enabled (vendor tool / Device Manager). |
| `no kernel image is available for execution on the device` | The GPU architecture is missing from the build. Send the full output and the `nvidia-smi` header. |
| `cannot place the model`, `out of memory` | Something else is using VRAM. Close it and run again. For a manual `dara.exe` run you can also ask for a smaller context with `--kv 8192`. |
| `WARNING: running on battery` | Plug in the charger and run again; battery numbers are not comparable. |
| `missing (see docs/BENCHMARKING.md)` | A path is wrong: check `$env:DARA_EXE`, `$env:DARA_LLAMA_DIR` and that `models\` holds the downloaded files. |
| llama.cpp column shows `ERR` | The llama.cpp folder is missing `llama-completion.exe` / `llama-server.exe` or its CUDA runtime DLLs: unzip the cudart asset into the same folder. |
| A run stops making progress | Run `nvidia-smi`. If it errors, the GPU dropped out (sleep / Modern Standby): reboot, keep the charger in, and run again. |

## Building `dara.exe` yourself (optional)

Needs Visual Studio 2022 (any edition) with **Desktop development with C++**, and about 4 GB for the CUDA toolkit:

```powershell
python -I tools\install_cuda_redist.py --version 13.1.1 --dest C:\cuda   # minimal CUDA toolkit, no admin rights
$env:DARA_CUDA = "C:\cuda\13.1.1"
$env:DARA_ARCH = "86"        # 86 = RTX 30xx, 89 = RTX 40xx, 75 = GTX 16xx / RTX 20xx; leave unset to build for all (slow)
cmd /c tools\build_dara.bat  # -> build\cuda\dara.exe
```

The same steps run in GitHub Actions (`.github/workflows/build-release.yml`), which publishes a release for every push to `main`.
