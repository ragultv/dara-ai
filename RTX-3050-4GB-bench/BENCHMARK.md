# Dara Benchmark Results: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)

Comprehensive benchmark results of Dara (`build\cuda\dara.exe` compiled from branch `dev` at commit `4b68f94`) vs. reference `llama.cpp` (build `b12189`, CUDA 13.5) on Windows 11.

---

## 1. System & Environment Specifications

| Component | Specification |
|---|---|
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (Ampere `sm_86`, 4096 MiB VRAM) |
| **Driver** | 621.24 WHQL (CUDA UMD Version: 13.5) |
| **CPU** | AMD Ryzen 7 6800HS with Radeon Graphics (8 cores / 16 threads) |
| **RAM** | 15.3 GB System RAM |
| **Operating System** | Windows 11 Home (Build 10.0.26300) |
| **Power State** | AC Power connected (95W dynamic boost limit) |
| **Dara Version** | Commit `4b68f94` (`origin/dev`), compiled with CUDA 13.1.1 statically linked (`-arch=sm_86`) |
| **Reference Runtime** | `llama.cpp` release build `b12189` with CUDA 13.5 runtime DLLs |

---

## 2. Text Decode Speed: Plain vs. llama.cpp (`tasks_plain`)

Evaluated over all 16 standardized prompt tasks across `edit`, `transform`, and `generate` categories.

### Core 4B (`Qwen3.5-4B-Q4_K_M`)
* **Edit geometric mean:** **1.02x** llama.cpp
* **Transform geometric mean:** **1.02x** llama.cpp
* **Generate geometric mean:** **1.02x** llama.cpp
* **Overall geometric mean:** **52.2 tok/s vs 51.3 tok/s (1.02x)**

### Gemma 4 E2B (`Nano-Q4_K_M`)
* **Edit geometric mean:** **0.93x** llama.cpp (with faster time-to-first-token: ~0.19s vs ~0.53s)
* **Transform geometric mean:** **0.93x** llama.cpp
* **Generate geometric mean:** **0.94x** llama.cpp
* **Overall geometric mean:** **80.5 tok/s vs 86.0 tok/s (0.94x)**

---

## 3. Flash Speculative Decoding Benchmark

Speculative drafting speeds up token decode throughput using Dara's compact drafter neural networks.

### Core 4B with `drafter_4b_v2.dflash` (`--spec 4 --flash-min-p 0.6`)
* **Overall Geometric Mean Ratio:** **1.59x faster than llama.cpp**
* **Edit Tasks Geometric Mean:** **2.00x**
  * `sql_rename`: **135.2 tok/s** vs. llama.cpp 51.3 tok/s (**2.64x**)
  * `function_edit`: **112.2 tok/s** vs. llama.cpp 51.3 tok/s (**2.19x**)
  * `json_add_field`: **102.6 tok/s** vs. llama.cpp 51.5 tok/s (**1.99x**)
  * `fix_typos`: **77.2 tok/s** vs. llama.cpp 51.1 tok/s (**1.51x**)
* **Transform Tasks Geometric Mean:** **1.57x**
  * `csv_to_table`: **89.9 tok/s** vs. llama.cpp 51.5 tok/s (**1.75x**)
  * `invoice_to_json`: **88.4 tok/s** vs. llama.cpp 51.4 tok/s (**1.72x**)
  * `data_to_table`: **77.4 tok/s** vs. llama.cpp 51.3 tok/s (**1.51x**)
* **Generate Tasks Geometric Mean:** **1.43x**
  * `tamil_summary`: **122.3 tok/s** vs. llama.cpp 51.3 tok/s (**2.39x**)
  * `write_function`: **76.3 tok/s** vs. llama.cpp 51.5 tok/s (**1.48x**)

### Gemma 4 E2B (Nano) with `drafter_nano_v3.dflash` (`--spec 4 --flash-min-p 0.6`)
* **Overall Geometric Mean Ratio:** **1.47x faster than llama.cpp**
* **Edit Tasks Geometric Mean:** **2.03x**
  * `function_edit`: **212.8 tok/s** vs. llama.cpp 86.1 tok/s (**2.47x**)
  * `sql_rename`: **210.7 tok/s** vs. llama.cpp 86.5 tok/s (**2.44x**)
  * `json_add_field`: **177.2 tok/s** vs. llama.cpp 86.5 tok/s (**2.05x**)
  * `fix_typos`: **118.8 tok/s** vs. llama.cpp 85.5 tok/s (**1.39x**)
* **Transform Tasks Geometric Mean:** **1.45x**
  * `invoice_to_json`: **147.2 tok/s** vs. llama.cpp 86.2 tok/s (**1.71x**)
  * `csv_to_table`: **125.9 tok/s** vs. llama.cpp 86.3 tok/s (**1.46x**)
  * `data_to_table`: **117.1 tok/s** vs. llama.cpp 85.9 tok/s (**1.36x**)
* **Generate Tasks Geometric Mean:** **1.26x**
  * `tone_rewrite`: **123.4 tok/s** vs. llama.cpp 86.4 tok/s (**1.43x**)
  * `project_plan`: **114.4 tok/s** vs. llama.cpp 86.6 tok/s (**1.32x**)
  * `notes_to_todos`: **113.8 tok/s** vs. llama.cpp 86.4 tok/s (**1.32x**)
  * `write_function`: **111.8 tok/s** vs. llama.cpp 85.8 tok/s (**1.30x**)

---

## 4. Vision / Image Time-to-First-Token (TTFT)

Measured with `bench/assets/photo1080.jpg` (Full 1920x1080) and `bench/assets/photo1080_oprel.jpg` (853x480).

| Model & Mode | Dara TTFT | llama-server TTFT | Speedup | Answer |
|---|---|---|---|---|
| **Core 4B — Full 1080p (cap 1024)** | **3.747 s** | 9.332 s | **2.49x faster** | `INVOICE 7731` |
| **Core 4B — Full 1080p (cap 400)** | **1.282 s** | 9.332 s | **7.28x faster** | `INVOICE 7731` |
| **Core 4B — Preprocessed 853x480** | **1.349 s** | 1.569 s | **1.16x faster** | `INVOICE 7731` |
| **Nano — Full 1080p** | **0.859 s** | 1.358 s | **1.58x faster** | `INVOICE 7731` |
| **Nano — Preprocessed 853x480** | 0.873 s | 0.259 s | 0.30x | `INVOICE 7731` |

---

## 5. Long-Context & Attention Kernel Verification (`prefill_ab.py`)

A/B performance comparison verifying the new grouped-query decode attention (`DARA_GQA_ATTN`) and grouped prefill attention (`DARA_ATTN_GROUP_PREFILL`):

```text
model  prompt | prefill tok/s: before    after      x | decode tok/s: before    after      x
nano      964 |                  1854     1909  1.03x |                 78.4     78.9  1.01x
nano     4084 |                  1844     1873  1.02x |                 73.6     79.1  1.07x
nano    16235 |                  1329     1332  1.00x |                 52.4     72.7  1.39x
4b        945 |                   654      825  1.26x |                 34.0     34.4  1.01x
4b       3997 |                   666      809  1.22x |                 32.5     33.6  1.03x
4b      15920 |                   375      596  1.59x |                 25.8     30.9  1.20x

GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VERDICT: not slower anywhere (every ratio >= 0.97)
```

---

## 6. Prompt Cache Interleaved Verification (`multi_conv_test.py`)

Tests the parked-conversation prompt cache across three interleaved conversations:

```text
request   prompt | cache OFF: reused   ttft ms | cache ON: reused   ttft ms | same text
A1          1226 |                 3       885 |                3       837 | yes
B1          1377 |                 5      1008 |                5       998 | yes
C1          1378 |                 5      1008 |                5       995 | yes
A2          1274 |                 5       850 |             1248       107 | yes
B2          1425 |                 5       990 |             1399       108 | yes
C2          1426 |                 5       984 |             1400       107 | yes
A3          1322 |                 5       974 |             1296       108 | yes

answers identical with the cache on and off: 7/7
```

* **Correctness:** 7/7 bitwise/exact match with cache ON vs OFF.
* **Latency Reduction:** Second-visit latency decreased from **~850–1000 ms to ~107 ms (~9x speedup)**.

---

## 7. Kernel Precision Unit Tests

* **`attn_gqa_test.exe`**: All GQA checks passed (0 failures).
  * Qwen 32k context: GQA reads at **125 GB/s (3.93x faster than per-head kernel)**.
  * Nano 131k context: GQA reads at **66 GB/s (3.67x faster than per-head kernel)**.
* **`attn_prefill_i8_test.exe`**: Grouped heads vs per-head pairs: bitwise identical across all context tiers (0 to 33,000 tokens).

---

## 8. Directory Files Overview

```text
RTX-3050-4GB-bench/
├── BENCHMARK.md                 # Complete benchmark documentation
├── image_ttft.json              # Vision TTFT timings
├── tasks_plain/
│   └── task_bench.json          # Plain text decode speeds (4B & Nano vs llama.cpp)
├── tasks_flash_4b/
│   └── task_bench.json          # Flash speculative decoding on Core 4B (v2 drafter)
└── tasks_flash_nano/
    └── task_bench.json          # Flash speculative decoding on Nano (v3 drafter)
```
