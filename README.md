<h1 align="center">Dara</h1>

<p align="center">
  <b>Private AI that runs on your own NVIDIA laptop GPU.</b><br />
  A Windows desktop app with its own CUDA inference engine for Gemma 4 and Qwen3.5 models.<br />
  Your chats, documents and images stay on your computer.
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-5b21b6" /></a>
  <img alt="Platform: Windows" src="https://img.shields.io/badge/platform-Windows%2010%2F11-5b21b6" />
  <img alt="GPU: NVIDIA GTX 16 / RTX 20 and newer" src="https://img.shields.io/badge/GPU-NVIDIA%20GTX%2016%20%2F%20RTX%2020%2B-5b21b6" />
</p>

---

## What it is

- **A desktop app** (Tauri + React) for chatting with local models: documents, images, and optional tool connectors (MCP). Everything is stored in a database on your computer.
- **Its own engine** (C++20 and CUDA, no Python or PyTorch at runtime). It reads standard GGUF files, plans how many layers and how much context fit your VRAM, and runs 8-bit integer kernels, CUDA Graphs and attention kernels built for long conversations.
- **Dara Flash**: a small trained drafter plus prompt lookup guess several words ahead and the model checks them in one pass, so the same memory read can produce more than one word. The model always writes its own answer.
- **A local API**: the same engine runs as an OpenAI-compatible server on `127.0.0.1`.
- It is built for **laptop GPUs with 4 to 12 GB of VRAM**. No account, no cloud, no telemetry.

## Models

Models are downloaded inside the app (they are not part of the installer) as GGUF files at Q4_K_M, together with their image encoder.

| Model | Based on | Download | GPU memory | Flash drafter |
| --- | --- | --- | --- | --- |
| Dara Nano | Gemma 4 E2B (instruct) | 3.1 GB + 1.0 GB image encoder | from 3 GB, comfortable at 4 GB | v3 (37 MB) |
| Dara Core 4B | Qwen3.5 4B | 2.7 GB + 0.7 GB image encoder | from 3 GB, comfortable at 4 GB | v2 (86 MB) |
| Dara Pro 9B | Qwen3.5 9B | 5.7 GB + 0.9 GB image encoder | from 4 GB, comfortable at 8 GB | not yet |

Pro 9B runs partly on the CPU on GPUs under 8 GB, so it is noticeably slower there. Sizes come from [`src/serve/catalog.cpp`](src/serve/catalog.cpp).

## Speed

Measured against llama.cpp (build b12189) on the same GGUF files, on a laptop with a 4 GB GPU, over 16 everyday tasks, greedy decoding, median of three runs. The number is Dara's decode speed divided by llama.cpp's (geometric mean).

| Model | Dara with Flash | Dara plain (Flash off) |
| --- | --- | --- |
| Dara Nano | 1.61x | 1.14x |
| Dara Core 4B | 1.65x | 1.25x |

Edits (where most of the answer is already in your prompt) gain the most, up to about 2.5x. Open-ended writing gains around 1.3 to 1.6x, not several times. Only a 4 GB laptop GPU has been measured so far. To check yours, see [docs/RTX_LAPTOP_CHECK.md](docs/RTX_LAPTOP_CHECK.md) (about 15 minutes), and [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for how every number is produced.

Long conversations stay usable: the attention work reads the conversation history once per head group instead of once per head, so decoding slows far less as the context grows (Dara Nano decodes 38.7 tokens/s at 65K tokens of context).

## Install

1. Download `Dara_<version>_x64-setup.exe` from the [releases page](../../releases). It installs for your user account only, with no administrator prompt.
2. You need an NVIDIA GPU from the **GTX 16 or RTX 20 series on**, and a driver that supports **CUDA 13 (R580 or newer)**.
3. Open **Models** in the app, pick one that fits your GPU and download it.

The installer is not code-signed yet, so Windows SmartScreen may warn when you run it (More info, then Run anyway). Check the SHA-256 file published with each release if you want to verify the download. There is no automatic update yet: install the new version over the old one.

## Code signing policy

The Windows installer is **not code-signed yet**, which is why SmartScreen warns. Signing is being prepared with the free programme of [SignPath Foundation](https://signpath.org) for open-source projects; the build workflow already supports it, and the steps are in [docs/CODE_SIGNING.md](docs/CODE_SIGNING.md). Until then, check the SHA-256 file published with each release.

- **Team and roles.** Authors, reviewers and approvers: [ragultv](https://github.com/ragultv), the repository owner. Every release is built by the public GitHub Actions workflow from the source in this repository.
- **Privacy.** Dara has no accounts, analytics or update checks and sends your chats and files nowhere. The network is used only when you start it: downloading a model (from Hugging Face) and its Flash drafter (from this project's releases), the installer fetching Microsoft's WebView2 runtime if it is missing, and a connector you add talking to its own service. The project site's Privacy page says the same in full.
- **Installing and removing.** The installer installs for your user account only, tells you where, and the uninstaller removes it (your models folder is kept).

## Use it from your own code

The installer adds Dara's `engine` folder to your user PATH (and the uninstaller removes it), so `dara` works in any terminal you open **after** installing. A terminal that was already open needs to be reopened.

```powershell
dara pull dara-nano
dara serve --model dara-nano

curl http://127.0.0.1:11435/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{"model":"dara-nano","messages":[{"role":"user","content":"Hello"}]}'
```

The server listens on `127.0.0.1` only unless you turn on LAN access, which forces an API key. The full API and options are in [docs/SERVER.md](docs/SERVER.md).

## Build from source

You need Windows 10/11, Visual Studio 2022 Build Tools (C++), the CUDA Toolkit 13.x, and CMake. For the desktop app also Node 22, pnpm and Rust.

```bat
tools\build_dara.bat              :: the engine -> build\cuda\dara.exe
app\tools\build_installer.bat     :: stages the engine, then builds the installer
```

Set `DARA_ARCH` to your GPU's architecture (75, 86, 89, 120) to build only for it. [docs/RELEASE.md](docs/RELEASE.md) describes the installer and the release workflow; [docs/RTX_LAPTOP_CHECK.md](docs/RTX_LAPTOP_CHECK.md) walks through building and testing on a new GPU.

## Repository layout

| Folder | What is in it |
| --- | --- |
| `src/` | The engine: model code, CUDA kernels, GGUF loader, tokenizer, the HTTP server |
| `app/` | The desktop app (Tauri 2, React 19, TypeScript) |
| `flash/` | The Flash drafter for Dara Nano |
| `bench/`, `tests/` | Benchmarks and kernel tests |
| `tools/` | Build scripts and training and conversion tools |
| `docs/` | Server API, benchmarking, release, and the UI handbook (`docs/ui/`) |
| `third_party/` | Vendored libraries, each with its own license ([list](third_party/README.md)) |

## Privacy

Dara runs on your computer. It has no accounts, no analytics, no crash reporting and no update pings. The only network traffic is the model downloads you start (Hugging Face and this repository's releases) and the tool connectors you choose to add. Details are in [docs/ui/05-security.md](docs/ui/05-security.md).

## Contributing

Issues and pull requests are welcome. Read [docs/ui/06-coding-standards.md](docs/ui/06-coding-standards.md) for the app and [docs/BENCHMARKING.md](docs/BENCHMARKING.md) before changing anything that affects speed: new kernel paths need a bitwise or quality check and an off switch. Work happens on `dev`; pull requests go from `dev` to `main`.

## License

[MIT](LICENSE). Vendored code in `third_party/` keeps its own license. Models downloaded by the app (Gemma, Qwen) are not part of this repository and are covered by their own licenses.

## Star history

<a href="https://star-history.com/#ragultv/dara-ai&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ragultv/dara-ai&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ragultv/dara-ai&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ragultv/dara-ai&type=Date" />
  </picture>
</a>
