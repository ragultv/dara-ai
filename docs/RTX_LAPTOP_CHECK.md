# Checking Dara on a new GPU (RTX laptop): build, run, send back

The long-context speed work (grouped-query decode attention, grouped prefill attention, int8 attention scores on GPUs without tensor-core attention,
and the parked-conversation prompt cache) was measured on one GPU only, a GTX 1650 (sm_75). This is how to build the engine on another laptop, check
that nothing got slower or wrong there, and send the result back. About 40 minutes, most of it the build and the model download.

## 0. What you need

* Windows 10/11, an NVIDIA GPU, a driver that reports CUDA 13 or newer (`nvidia-smi`, R580+), the charger plugged in.
* Git, Python 3.10+, and **Visual Studio 2022** (any edition) with *Desktop development with C++*.
* About 15 GB of disk (models 7 GB, CUDA toolkit 4 GB, build files).

## 1. Get the code (branch `dev`)

```powershell
git clone https://github.com/ragultv/dara-ai.git C:\dara
cd C:\dara
git checkout dev
git pull
git log -1 --oneline        # note this commit with your results
```

## 2. Build `dara.exe` for this GPU

```powershell
python -I tools\install_cuda_redist.py --version 13.1.1 --dest C:\cuda     # minimal CUDA toolkit, no admin rights (skip if you have CUDA 13.1)
$env:DARA_CUDA = "C:\cuda\13.1.1"
# pick YOUR GPU:  75 = GTX 16xx / RTX 20xx   86 = RTX 30xx   89 = RTX 40xx   120 = RTX 50xx     (unset = all of them, much slower to build)
$env:DARA_ARCH = "86"
cmd /c tools\build_dara.bat          # -> build\cuda\dara.exe   (about 10-15 minutes for one architecture)
```

`build\cuda\dara.exe` is what every script below uses. To build next to a running copy use `$env:DARA_OUT = "build\cuda\dara_new.exe"` and pass
`--dara-exe build\cuda\dara_new.exe` to the scripts.

## 3. Get the models

```powershell
python -I bench\get_models.py --models nano 4b       # about 7 GB, resumes if interrupted
```

## 4. The check (about 15 minutes)

Close everything that uses the GPU; `nvidia-smi` should show almost no memory in use. Then:

```powershell
python -I bench\prefill_ab.py --models nano 4b --sizes 1 4 16
```

It runs the **same** `dara.exe` twice per prompt length, once with the new paths turned off and once with the defaults, and prints a table:

```
model  prompt | prefill tok/s: before    after      x | decode tok/s: before    after      x
nano    15648 |                   507      623  1.23x |                 42.2     61.2  1.45x
...
VERDICT: not slower anywhere (every ratio >= 0.97)
```

* `VERDICT: not slower anywhere` is the result we want. `VERDICT: SLOWER ...` names the cases that lost more than 3%.
* Reference numbers from the GTX 1650 (before -> after): Nano 15.6K tokens prefill 1.23x, decode 1.45x; 4B 15.4K tokens prefill 1.65x, decode 1.32x.
* On RTX 30xx and newer the int8 attention-score path is deliberately not used (those GPUs keep their tensor-core attention), so expect the gain to come from
  grouped decode attention and grouped prefill attention.

If a row is slower, rerun with one new path off at a time to find which one (set the variable, run the same command, compare):

| Variable | Turns off |
|---|---|
| `$env:DARA_GQA_ATTN = "0"` | grouped-query decode attention (reads each K/V row once per group) |
| `$env:DARA_ATTN_GROUP_PREFILL = "0"` | grouped prefill attention |
| `$env:DARA_ATTN_QK_INT8 = "0"` | int8 attention scores (GTX 16xx / RTX 20xx only) |
| `$env:DARA_PROMPT_CACHE_MB = "0"` | the parked-conversation prompt cache |

### 4b. The prompt cache gives identical answers (about 5 minutes)

```powershell
python -I bench\multi_conv_test.py --model nano
```

Three long conversations are interleaved with the cache off and on. Expect `answers identical with the cache on and off: 7/7` and a first-token time for
the second visit to a conversation of roughly 100-400 ms instead of 1.5-3 s. (On the GTX 1650: Nano 150 ms vs 1.75-2.1 s.)

### 4c. Unit tests of the new kernels (optional, 10 minutes)

```powershell
cmd /c tools\build_gqa_test.bat
build\cuda\attn_gqa_test.exe             # every line must say [pass]; ends with "all GQA checks passed"
cmd /c tools\build_prefill_test.bat
build\cuda\attn_prefill_i8_test.exe      # "grouped heads vs per-head pairs: bitwise identical" on every Qwen line; "done" at the end
```

## 5. The full benchmark against llama.cpp (optional, 2-3 hours)

Follow `docs/BENCHMARKING.md`: step 5 (llama.cpp b12189), then section 7: `task_bench.py` (7.1-7.2) and `husky_bench.py`, `tool_bench.py`, `chat_ttft_bench.py`
(7.7). The Flash drafters are `build\flash\drafter_nano_v3.dflash` (Nano) and `build\flash\drafter_4b_v2.dflash` (Core 4B); they come in the release zip's
`flash\` folder (the `flash-drafters-v3` release) and are not in git.

## 6. What to send back

* The whole console output of step 4 (it ends with the VERDICT line) and of 4b.
* `build\bench\prefill_ab\prefill_ab.json` (it records the GPU, driver, RAM and whether the laptop was on AC).
* The output of `nvidia-smi` and the commit from step 1.
* If anything failed to build or run: the last 30 lines of the console.
