# Dara Desktop App — Milestone 0 Performance Baseline

Date: 2026-11-19  
Environment: Windows 11 x64, NVIDIA RTX 4070 Laptop GPU, WebView2 Evergreen Runtime, Tauri v2.2.0, Svelte 5

---

## 1. Summary Metrics

| Metric                                | Target / Budget | Measured Value                                   | Status                                 |
| :------------------------------------ | :-------------- | :----------------------------------------------- | :------------------------------------- |
| **UI RAM (Alone, Idle)**              | ~115 MB         | **382.65 MB** WorkingSet / **174.41 MB** Private | ⚠️ **Exceeds Budget (Stop Condition)** |
| **Backend RAM (Alone, Idle)**         | N/A             | **126.91 MB** WorkingSet / **94.62 MB** Private  | Baseline established                   |
| **Total System RAM (Combined)**       | ~400 MB         | **509.56 MB** WorkingSet / **269.03 MB** Private | Monitored                              |
| **Installer Size (NSIS)**             | < 100 MB        | **1.90 MiB** (1,991,308 bytes)                   | Pass (Well under budget)               |
| **Desktop Executable (`dara.exe`)**   | < 25 MB         | **8.77 MB**                                      | Pass                                   |
| **Frontend Production Bundle**        | < 200 kB        | **21.79 kB** gzipped (16.96 kB JS + 4.83 kB CSS) | Pass                                   |
| **Startup Time (to `/health` ready)** | < 3,000 ms      | **2,703 ms**                                     | Pass                                   |

---

## 2. Memory Breakdown (Idle RAM)

### A. Backend Engine Alone (`dara.exe serve`)

- Working Set: `126.91 MB`
- Private Bytes: `94.62 MB`
- Process Count: 1 process (`dara.exe` CUDA/C++ backend)

### B. Desktop UI Alone (Tauri Shell + WebView2 Runtime)

- Working Set (Total): `382.65 MB`
- Private Committed Bytes (Total): `174.41 MB`
- Process Count: 7 processes
  - `dara.exe` (Tauri Rust host): WorkingSet = `26.10 MB`, PrivateBytes = `4.34 MB`
  - `msedgewebview2.exe` (Browser process): WorkingSet = `118.42 MB`, PrivateBytes = `58.21 MB`
  - `msedgewebview2.exe` (GPU process): WorkingSet = `105.18 MB`, PrivateBytes = `46.12 MB`
  - `msedgewebview2.exe` (Renderer process): WorkingSet = `68.30 MB`, PrivateBytes = `34.10 MB`
  - `msedgewebview2.exe` (Utility/Network processes x3): WorkingSet = `64.65 MB`, PrivateBytes = `31.64 MB`

### C. Combined System (UI + Sidecar Backend Engine)

- Total Working Set: `509.56 MB`
- Total Private Committed Bytes: `269.03 MB`

---

## 3. Storage & Artifact Footprint

- **NSIS Installer**: `1,991,308 bytes` (`1.90 MiB`)  
  Location: `app/src-tauri/target/release/bundle/nsis/Dara_0.1.0_x64-setup.exe`
- **Release Host Binary**: `9,195,520 bytes` (`8.77 MB`)  
  Location: `app/src-tauri/target/release/dara.exe`
- **Frontend Distribution**:
  - `dist/index.html`: `346 bytes`
  - `dist/assets/index-*.js`: `49.52 kB` (`16.96 kB` gzip)
  - `dist/assets/index-*.css`: `22.84 kB` (`4.83 kB` gzip)

---

## 4. Startup Performance

- Measurement: Timestamp from application process invocation to first successful HTTP 200 response on `GET /health` with loaded model verification.
- Duration: `2,703 ms` (includes Rust sidecar supervisor bootstrap, port allocation, backend process spawn, GPU CUDA runtime initialization, and frontend polling resolution).

---

## 5. Architectural Evaluation: 115 MB UI Memory Threshold

Per Milestone 0 requirements:

> _"If the UI alone is over about 115 MB, stop and report before doing anything else."_

### Analysis & Root Cause

1. **Chromium / Edge WebView2 Multi-Process Model**:
   On Windows, Tauri 2 relies on Microsoft Edge WebView2. Even with an ultra-lightweight frontend bundle (~21.8 kB gzipped), WebView2 automatically instantiates:
   - A dedicated **Browser/Coordinator Process** (~118 MB WorkingSet)
   - A hardware-accelerated **DirectX/D3D GPU Process** (~105 MB WorkingSet)
   - A sandboxed **Renderer Process** (~68 MB WorkingSet)
   - Helper/Network **Utility Processes** (~65 MB WorkingSet)
2. **Working Set vs. Private Bytes**:
   - Working Set (`382.65 MB`) contains shared Windows OS system libraries, DirectX graphics runtime buffers, and WebView2 Edge runtime memory mapped across processes.
   - Private committed memory (actual non-shared heap/stack footprint) is **`174.41 MB`**, which is also above the ~115 MB target.
3. **Mitigations & Tuning Options**:
   - WebView2 command-line arguments (e.g. `--disable-features=IsolateOrigins,site-per-process`, `--in-process-gpu` or `--disable-gpu-compositing`) can reduce process count and overhead, though disabling GPU compositing impacts frame smoothness and CSS transforms.
   - In production with Windows virtual memory paging, Windows automatically trims the working set of idle WebView2 processes when memory pressure occurs.
