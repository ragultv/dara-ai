#include "serve/catalog.h"

#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace dara::serve {
namespace {

std::string u8(const fs::path& p) { const auto s = p.u8string(); return std::string(s.begin(), s.end()); }

std::string hf(const char* repo, const char* rev, const char* file) {
    return std::string("https://huggingface.co/") + repo + "/resolve/" + rev + "/" + file;
}

// Flash drafters: release assets of the Dara repository (override the base with DARA_FLASH_BASE_URL, e.g. a Hugging Face resolve URL).
std::string flash_url(const char* file) {
    const char* base = std::getenv("DARA_FLASH_BASE_URL");
    std::string b = base && *base ? base : "https://github.com/ragultv/dara-ai/releases/download/flash-drafters-v3";
    if (!b.empty() && b.back() != '/') b += '/';
    return b + file;
}

std::vector<CatalogEntry> build() {
    std::vector<CatalogEntry> v;
    {
        CatalogEntry e;
        e.id = "dara-nano"; e.name = "Dara Nano"; e.family = "gemma4"; e.base_model = "Gemma 4 E2B (instruct)"; e.quant = "Q4_K_M";
        e.description = "Smallest and fastest tier: chat, quick answers, image understanding. Up to 128K context on 4 GB GPUs.";
        e.legacy_dir = "gemma4-e2b-gguf"; e.min_vram_mb = 3072; e.full_vram_mb = 4096;
        const char* repo = "unsloth/gemma-4-E2B-it-GGUF"; const char* rev = "0314792d7f1f7e229411f620751375812bb9faf2";
        e.files.push_back({"model", "gemma-4-E2B-it-Q4_K_M.gguf", hf(repo, rev, "gemma-4-E2B-it-Q4_K_M.gguf"), 3106738272ull, "740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8"});
        e.files.push_back({"mmproj", "mmproj-F16.gguf", hf(repo, rev, "mmproj-F16.gguf"), 985654080ull, "140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa"});
        e.files.push_back({"flash", "drafter_nano_v3.dflash", flash_url("drafter_nano_v3.dflash"), 36911148ull, "5423c6ec09d0ec063eea32a4d6cf42d44b7027211614b876990d0b72b9b124ef"});
        v.push_back(e);
    }
    {
        CatalogEntry e;
        e.id = "dara-core-4b"; e.name = "Dara Core 4B"; e.family = "qwen35"; e.base_model = "Qwen3.5 4B"; e.quant = "Q4_K_M";
        e.description = "Balanced tier: reasoning, coding, documents, image understanding.";
        e.legacy_dir = "qwen3.5-4b-gguf"; e.min_vram_mb = 3072; e.full_vram_mb = 4096;
        const char* repo = "unsloth/Qwen3.5-4B-GGUF"; const char* rev = "e87f176479d0855a907a41277aca2f8ee7a09523";
        e.files.push_back({"model", "Qwen3.5-4B-Q4_K_M.gguf", hf(repo, rev, "Qwen3.5-4B-Q4_K_M.gguf"), 2740937888ull, "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"});
        e.files.push_back({"mmproj", "mmproj-F16.gguf", hf(repo, rev, "mmproj-F16.gguf"), 672423616ull, "cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864"});
        e.files.push_back({"flash", "drafter_4b_v2.dflash", flash_url("drafter_4b_v2.dflash"), 86497324ull, "3c0c9a270944f8bd57cb6e945df89e853f8d95bf56ffa55def15c0464ba37f43"});
        v.push_back(e);
    }
    {
        CatalogEntry e;
        e.id = "dara-pro-9b"; e.name = "Dara Pro 9B"; e.family = "qwen35"; e.base_model = "Qwen3.5 9B"; e.quant = "Q4_K_M";
        e.description = "Most capable tier: harder reasoning and coding. Runs partly on the CPU on GPUs under 8 GB.";
        e.legacy_dir = "qwen3.5-9b-gguf"; e.min_vram_mb = 4096; e.full_vram_mb = 8192;
        const char* repo = "unsloth/Qwen3.5-9B-GGUF"; const char* rev = "3885219b6810b007914f3a7950a8d1b469d598a5";
        e.files.push_back({"model", "Qwen3.5-9B-Q4_K_M.gguf", hf(repo, rev, "Qwen3.5-9B-Q4_K_M.gguf"), 5680522464ull, "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"});
        e.files.push_back({"mmproj", "mmproj-F16.gguf", hf(repo, rev, "mmproj-F16.gguf"), 918166080ull, "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"});
        v.push_back(e);   // no Flash drafter trained for 9B yet
    }
    return v;
}

}  // namespace

const CatalogFile* CatalogEntry::file(const std::string& kind) const {
    for (const auto& f : files) if (f.kind == kind) return &f;
    return nullptr;
}

const std::vector<CatalogEntry>& catalog() {
    static const std::vector<CatalogEntry> c = build();
    return c;
}

const CatalogEntry* catalog_find(const std::string& id) {
    std::string k = id;
    for (char& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (k == "nano") k = "dara-nano";
    else if (k == "4b" || k == "core" || k == "core-4b") k = "dara-core-4b";
    else if (k == "9b" || k == "pro" || k == "pro-9b") k = "dara-pro-9b";
    for (const auto& e : catalog()) if (e.id == k) return &e;
    return nullptr;
}

std::string exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return u8(fs::path(buf).parent_path());
#endif
    return ".";
}

std::string u8s_path(const std::string& p) { return u8(fs::u8path(p)); }

std::string default_models_dir() {
    if (const char* e = std::getenv("DARA_MODELS"); e && *e) return e;
    if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) return u8(fs::u8path(l) / "Dara" / "models");
    return "models";
}

std::string target_path(const std::string& models_dir, const CatalogEntry& e, const CatalogFile& f) {
    return u8(fs::u8path(models_dir) / e.id / f.name);
}

std::string installed_path(const std::string& models_dir, const CatalogEntry& e, const CatalogFile& f) {
    std::vector<fs::path> c = {fs::u8path(models_dir) / e.id / f.name, fs::u8path(models_dir) / e.legacy_dir / f.name};
    if (f.kind == "flash") { c.push_back(fs::u8path(exe_dir()) / "flash" / f.name); c.push_back(fs::path("build") / "flash" / f.name); }
    std::error_code ec;
    for (const auto& p : c) {
        const auto sz = fs::file_size(p, ec);
        if (!ec && sz == f.size) return u8(fs::absolute(p, ec));
    }
    return "";
}

}  // namespace dara::serve
