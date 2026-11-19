// The model options a user can choose and what each does on THIS machine, for GET /dara/models/{id}/options and `dara options`:
//   * placement (context, layers on the GPU, output head location) from the real placement planner (ModelPlanner: a dry run, nothing is loaded),
//   * quality label of every K/V format (from the quality gate, quality_table.h),
//   * measured speeds from the one-time per-machine benchmark (CalibrationStore), or "not measured" -- never a guess presented as a measurement.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "serve/catalog.h"
#include "serve/engine.h"

namespace dara::serve {

// What a calibration is valid for: another GPU, another driver or another Dara version measures differently.
struct MachineKey {
    std::string gpu;
    uint64_t vram_mb = 0;
    int driver = 0;            // cudaDriverGetVersion
    std::string dara;          // DARA_VERSION
    bool operator==(const MachineKey& o) const { return gpu == o.gpu && vram_mb == o.vram_mb && driver == o.driver && dara == o.dara; }
    bool operator!=(const MachineKey& o) const { return !(*this == o); }
};
MachineKey current_machine();
std::string current_power();      // "ac", "battery" or "unknown"

// One measured configuration. Speed at an empty context depends only on the placement, so it is keyed by it, not by the context setting.
struct CalibEntry {
    MachineKey machine;
    std::string model, kv;
    bool flash = false;               // the drafter was loaded
    int layers_gpu = 0;
    bool head_cpu = false;
    size_t context = 0;               // the context the measurement was made at
    double decode_tps = 0;            // generation speed with an empty context (greedy, 128 tokens)
    double prefill_tps = 0;           // prompt processing speed (a ~600-token prompt)
    double full_est_tps = 0;          // estimated generation speed with `full_est_context` tokens held (linear in the held context, from a probe); 0 = none
    size_t full_est_context = 0;
    std::string measured_at;          // UTC, ISO 8601
    std::string power = "unknown";    // "ac", "battery" or "unknown": laptop GPUs run at lower clocks on battery
};

class CalibrationStore {
public:
    explicit CalibrationStore(std::string path);   // path of calibration.json
    static std::string default_path();             // DARA_CALIBRATION, else %LOCALAPPDATA%\Dara\calibration.json
    const std::string& path() const { return path_; }
    void load();
    bool save() const;                             // false when the file cannot be written
    const CalibEntry* find(const MachineKey& m, const std::string& model, const std::string& kv, bool flash, int layers_gpu, bool head_cpu) const;
    void put(const CalibEntry& e);                 // replaces an entry for the same machine and key
    size_t count(const MachineKey& m, const std::string& model) const;
    size_t count_other_machines(const MachineKey& m, const std::string& model) const;
    size_t count_on_battery(const MachineKey& m, const std::string& model) const;
    std::string latest(const MachineKey& m, const std::string& model) const;
    void clear(const MachineKey& m, const std::string& model);

private:
    std::string path_;
    mutable std::mutex mu_;
    std::vector<CalibEntry> entries_;
};

struct OptionsRequest {
    const CatalogEntry* entry = nullptr;
    std::string models_dir;
    ModelPlanner* planner = nullptr;
    const CalibrationStore* cal = nullptr;
    MachineKey machine;
    size_t assume_free_bytes = 0;      // VRAM to treat as free (0 = ask the driver)
    bool vision = true;                // include the image encoder (a few MB reserved; used when its mmproj is installed)
    std::vector<std::string> formats;  // empty: every format
};

nlohmann::json build_options(const OptionsRequest& rq);

// A configuration the benchmark should measure (distinct placements only).
struct BenchTarget {
    std::string kv;
    bool flash = false;
    size_t context = 0;       // load with this context
    int layers_gpu = 0;
    bool head_cpu = false;
    bool depth_probe = false; // also measure with a few thousand tokens held, for the full-context estimate
    std::string why;          // "fast", "default 32K", "max"
};
std::vector<BenchTarget> bench_targets(const OptionsRequest& rq, bool skip_measured);

const char* kv_format_name(const std::string& kv);
std::string utc_now();

}  // namespace dara::serve
