// The models Dara knows how to run and download: Dara ids (dara-nano, dara-core-4b, dara-pro-9b) mapped to pinned Hugging Face GGUF files, their
// image encoders (mmproj) and Flash drafters. Files are verified against the sizes and SHA-256 recorded here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace dara::serve {

struct CatalogFile {
    std::string kind;      // "model", "mmproj", "flash"
    std::string name;      // file name on disk
    std::string url;
    uint64_t size = 0;
    std::string sha256;    // lower-case hex
};

struct CatalogEntry {
    std::string id, name, description, family, base_model, quant;
    std::string legacy_dir;           // older layout: <models>/<legacy_dir>/<file> (bench/get_models.py)
    int min_vram_mb = 0, full_vram_mb = 0;   // runs at all / every layer in VRAM at 32K context
    std::vector<CatalogFile> files;
    const CatalogFile* file(const std::string& kind) const;
};

const std::vector<CatalogEntry>& catalog();
const CatalogEntry* catalog_find(const std::string& id);   // also accepts short names: nano, 4b, core, 9b, pro

// Where models live: --models-dir, else DARA_MODELS, else %LOCALAPPDATA%\Dara\models.
std::string default_models_dir();

// The installed path of a catalog file ("" if absent or the size differs): <dir>/<id>/<name>, then <dir>/<legacy_dir>/<name>; drafters also
// next to dara.exe (flash\) and build\flash.
std::string installed_path(const std::string& models_dir, const CatalogEntry& e, const CatalogFile& f);
// Download target for a file: <dir>/<id>/<name>.
std::string target_path(const std::string& models_dir, const CatalogEntry& e, const CatalogFile& f);

std::string exe_dir();
std::string u8s_path(const std::string& p);   // normalised UTF-8 path

}  // namespace dara::serve
