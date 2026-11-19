#include "serve/options.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include "core/version.h"
#include "serve/quality_table.h"

namespace dara::serve {

using json = nlohmann::json;
namespace fs = std::filesystem;

namespace {

struct FormatInfo { const char* id; const char* name; const char* description; double bits_per_value; bool recommendable; const char* caveat; };
// In order of quality (the reference first): the recommendation walks this list.
constexpr FormatInfo kFormats[] = {
    {"q8", "INT8 (default)", "8-bit keys and values: the reference quality.", 9.0, true, ""},
    {"k8v4", "INT8 keys + 4-bit values", "Keys stay 8-bit, values are 4-bit: about 1.4x the context of INT8.", 6.5, true, ""},
    {"q4", "Q4 (KIVI-style)", "4-bit keys (one scale per channel) and values; the newest 128 positions stay 8-bit: about 1.75x the context of INT8.", 5.0, true, ""},
    {"tq4", "TurboQuant 4-bit", "Rotated 4-bit codes: the smallest cache.", 4.1, false,
     "Never chosen automatically: it has no batched prompt-attention kernel yet, so long prompts are processed more slowly; its speed has not been measured."},
};

const FormatInfo* format_info(const std::string& id) {
    for (const FormatInfo& f : kFormats) if (id == f.id) return &f;
    return nullptr;
}

const QualityRow* quality_row(const std::string& model, const std::string& kv) {
    for (const QualityRow& r : kQualityTable) if (model == r.model && kv == r.kv) return &r;
    return nullptr;
}

json quality_json(const std::string& model, const std::string& kv) {
    const QualityRow* r = quality_row(model, kv);
    if (!r) return {{"label", "experimental"}, {"text", "Experimental: not checked on this model"}};
    const std::string l = r->label;
    return {{"label", l}, {"text", l == "high" ? "High quality" : l == "medium" ? "Medium quality" : "Lower quality"},
            {"min_top1_of_128", r->min_top1_of_128}, {"worst_clear_errors_of_128", r->worst_clear_errors}, {"worst_kl", r->worst_kl},
            {"checked_on", "5 workloads against the FP32 reference (english, code, structured, 1.5K and 8K-token prompts)"}};
}

double r2(double v) { return std::round(v * 100.0) / 100.0; }
double mb(size_t b) { return r2(b / 1048576.0); }

}  // namespace

const char* kv_format_name(const std::string& kv) { const FormatInfo* f = format_info(kv); return f ? f->name : kv.c_str(); }

std::string utc_now() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string current_power() {
#ifdef _WIN32
    SYSTEM_POWER_STATUS sps{};
    if (GetSystemPowerStatus(&sps)) { if (sps.ACLineStatus == 1) return "ac"; if (sps.ACLineStatus == 0) return "battery"; }
#endif
    return "unknown";
}

MachineKey current_machine() {
    MachineKey m;
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) { m.gpu = p.name; m.vram_mb = p.totalGlobalMem >> 20; }
    }
    cudaDriverGetVersion(&m.driver);
    m.dara = DARA_VERSION;
    return m;
}

// ---- calibration store --------------------------------------------------------------------------------------------------------------------------------

std::string CalibrationStore::default_path() {
    if (const char* e = std::getenv("DARA_CALIBRATION"); e && *e) return e;
    if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) { const auto p = (fs::u8path(l) / "Dara" / "calibration.json").u8string(); return std::string(p.begin(), p.end()); }
    return "calibration.json";
}

CalibrationStore::CalibrationStore(std::string path) : path_(std::move(path)) { load(); }

void CalibrationStore::load() {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.clear();
    std::ifstream f(fs::u8path(path_));
    if (!f) return;
    try {
        const json j = json::parse(f);
        if (j.value("version", 0) != 1) return;   // an unknown layout is ignored, never half-read
        for (const json& e : j.value("entries", json::array())) {
            CalibEntry c;
            c.machine = {e["machine"].value("gpu", ""), e["machine"].value("vram_mb", uint64_t{0}), e["machine"].value("driver", 0), e["machine"].value("dara", "")};
            c.model = e.value("model", ""); c.kv = e.value("kv", ""); c.flash = e.value("flash", false);
            c.layers_gpu = e.value("layers_gpu", 0); c.head_cpu = e.value("head_cpu", false);
            c.context = e.value("context", size_t{0});
            c.decode_tps = e.value("decode_tps", 0.0); c.prefill_tps = e.value("prefill_tps", 0.0);
            c.full_est_tps = e.value("full_est_tps", 0.0); c.full_est_context = e.value("full_est_context", size_t{0});
            c.measured_at = e.value("measured_at", "");
            c.power = e.value("power", "unknown");
            entries_.push_back(std::move(c));
        }
    } catch (const std::exception&) {
        entries_.clear();   // a damaged file is treated as empty (the next benchmark rewrites it)
    }
}

bool CalibrationStore::save() const {
    std::lock_guard<std::mutex> lk(mu_);
    json arr = json::array();
    for (const CalibEntry& c : entries_)
        arr.push_back({{"machine", {{"gpu", c.machine.gpu}, {"vram_mb", c.machine.vram_mb}, {"driver", c.machine.driver}, {"dara", c.machine.dara}}},
                       {"model", c.model}, {"kv", c.kv}, {"flash", c.flash}, {"layers_gpu", c.layers_gpu}, {"head_cpu", c.head_cpu}, {"context", c.context},
                       {"decode_tps", c.decode_tps}, {"prefill_tps", c.prefill_tps}, {"full_est_tps", c.full_est_tps}, {"full_est_context", c.full_est_context},
                       {"measured_at", c.measured_at}, {"power", c.power}});
    std::error_code ec;
    fs::create_directories(fs::u8path(path_).parent_path(), ec);
    const std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << json{{"version", 1}, {"entries", arr}}.dump(1);
        if (!f) return false;
    }
    fs::rename(fs::u8path(tmp), fs::u8path(path_), ec);
    if (ec) { fs::remove(fs::u8path(path_), ec); fs::rename(fs::u8path(tmp), fs::u8path(path_), ec); }
    return !ec;
}

const CalibEntry* CalibrationStore::find(const MachineKey& m, const std::string& model, const std::string& kv, bool flash, int layers_gpu, bool head_cpu) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const CalibEntry& c : entries_)
        if (c.machine == m && c.model == model && c.kv == kv && c.flash == flash && c.layers_gpu == layers_gpu && c.head_cpu == head_cpu) return &c;
    return nullptr;
}

void CalibrationStore::put(const CalibEntry& e) {
    std::lock_guard<std::mutex> lk(mu_);
    for (CalibEntry& c : entries_)
        if (c.machine == e.machine && c.model == e.model && c.kv == e.kv && c.flash == e.flash && c.layers_gpu == e.layers_gpu && c.head_cpu == e.head_cpu) { c = e; return; }
    entries_.push_back(e);
}

size_t CalibrationStore::count(const MachineKey& m, const std::string& model) const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const CalibEntry& c : entries_) n += (c.machine == m && c.model == model);
    return n;
}

size_t CalibrationStore::count_on_battery(const MachineKey& m, const std::string& model) const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const CalibEntry& c : entries_) n += (c.machine == m && c.model == model && c.power == "battery");
    return n;
}

size_t CalibrationStore::count_other_machines(const MachineKey& m, const std::string& model) const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const CalibEntry& c : entries_) n += (c.machine != m && c.model == model);
    return n;
}

std::string CalibrationStore::latest(const MachineKey& m, const std::string& model) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::string best;
    for (const CalibEntry& c : entries_) if (c.machine == m && c.model == model && c.measured_at > best) best = c.measured_at;
    return best;
}

void CalibrationStore::clear(const MachineKey& m, const std::string& model) {
    std::lock_guard<std::mutex> lk(mu_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](const CalibEntry& c) { return c.machine == m && c.model == model; }), entries_.end());
}

// ---- options ------------------------------------------------------------------------------------------------------------------------------------------

namespace {

constexpr size_t kDefaultContext = 32768;
constexpr size_t kMaxLadderContext = 131072;

// The formats to plan: those asked for; else every format (the listing) or only INT8 and Q4 (the benchmark, which is the slow part).
std::vector<std::string> wanted_formats(const OptionsRequest& rq, bool benchmark_defaults) {
    std::vector<std::string> v;
    for (const FormatInfo& f : kFormats) {
        const bool asked = std::find(rq.formats.begin(), rq.formats.end(), f.id) != rq.formats.end();
        if (!rq.formats.empty() ? asked : (!benchmark_defaults || std::string(f.id) == "q8" || std::string(f.id) == "q4")) v.push_back(f.id);
    }
    return v;
}
struct Context {
    const OptionsRequest& rq;
    std::string flash_path, mmproj_path;
};

Context make_context(const OptionsRequest& rq) {
    Context c{rq, "", ""};
    if (const CatalogFile* f = rq.entry->file("flash")) c.flash_path = installed_path(rq.models_dir, *rq.entry, *f);
    if (rq.vision) if (const CatalogFile* f = rq.entry->file("mmproj")) c.mmproj_path = installed_path(rq.models_dir, *rq.entry, *f);
    return c;
}

PlanResult plan_one(const Context& c, const std::string& kv, bool flash, size_t context) {
    EngineOptions q;
    q.kv_format = kv;
    q.kv = context;                 // 0: the largest context that keeps everything in VRAM (the fast profile)
    q.min_context = 0;
    q.flash = flash ? c.flash_path : std::string();
    q.flash_mode = flash ? "on" : "off";
    q.mmproj = c.mmproj_path;
    q.assume_free_bytes = c.rq.assume_free_bytes;
    return c.rq.planner->plan(q);
}

json speed_json(const Context& c, const std::string& kv, const PlanResult& p) {
    const CalibEntry* e = c.rq.cal ? c.rq.cal->find(c.rq.machine, c.rq.entry->id, kv, p.flash_active, p.layers_gpu, p.head_on_cpu) : nullptr;
    if (!e) return {{"source", "not_measured"}};
    json j = {{"source", "measured"}, {"decode_tps_start", r2(e->decode_tps)}, {"prefill_tps", r2(e->prefill_tps)}, {"measured_at", e->measured_at}, {"power", e->power}};
    if (e->full_est_tps > 0) { j["decode_tps_full_estimate"] = r2(e->full_est_tps); j["full_estimate_at_tokens"] = e->full_est_context; j["full_estimate_source"] = "estimated (linear in the held context, from a probe)"; }
    return j;
}

json point_json(const Context& c, const std::string& kv, const PlanResult& p, bool with_speed) {
    json j = {{"ok", p.ok}};
    if (!p.ok) { j["error"] = p.error; return j; }
    j["context"] = p.context;
    j["layers"] = p.layers;
    j["layers_gpu"] = p.layers_gpu;
    j["head"] = p.head_on_cpu ? "cpu" : "gpu";
    j["all_layers_on_gpu"] = p.layers_gpu == p.layers;
    j["fully_on_gpu"] = p.layers_gpu == p.layers && !p.head_on_cpu;
    j["flash_active"] = p.flash_active;
    j["kv_cache_mb"] = mb(p.state_bytes);
    if (with_speed) j["speed"] = speed_json(c, kv, p);
    return j;
}

std::vector<size_t> ladder(size_t native) {
    std::vector<size_t> v;
    for (size_t x : {size_t{4096}, size_t{8192}, size_t{16384}, size_t{32768}, size_t{65536}, size_t{131072}}) if (!native || x <= native) v.push_back(x);
    if (native > kMaxLadderContext) v.push_back(native);
    return v;
}

}  // namespace

json build_options(const OptionsRequest& rq) {
    const Context c = make_context(rq);
    const std::string& id = rq.entry->id;
    json out;
    out["model"] = id;
    out["name"] = rq.entry->name;
    out["family"] = rq.planner->family();
    out["machine"] = {{"gpu", rq.machine.gpu}, {"vram_total_mb", rq.machine.vram_mb}, {"driver", rq.machine.driver}, {"dara", rq.machine.dara}};
    const bool vision_in = !c.mmproj_path.empty();
    out["vision"] = {{"requested", rq.vision}, {"included", vision_in}, {"reserved_mb", vision_in ? 24 : 0},
                     {"note", vision_in ? "The image encoder streams its weights from disk and borrows its working memory from the unused context: it costs about 24 MB."
                                        : (rq.vision ? "The image encoder is not installed, so no image input is planned." : "Image input is not included.")}};
    const bool flash_installed = !c.flash_path.empty();
    out["flash"] = {{"available", rq.entry->file("flash") != nullptr}, {"installed", flash_installed},
                    {"note", !rq.entry->file("flash") ? "No drafter exists for this model yet." : (flash_installed ? "" : "Download the drafter (dara pull) to see Flash numbers.")}};

    json formats = json::array();
    json options = json::array();
    size_t native = 0, free_assumed = 0;
    struct Fast { std::string kv; bool flash; PlanResult p; };
    std::vector<Fast> fasts;
    size_t covered = 0, total_fast = 0;
    for (const std::string& kv : wanted_formats(rq, false)) {
        const FormatInfo* fi = format_info(kv);
        formats.push_back({{"id", kv}, {"name", fi->name}, {"description", fi->description}, {"bits_per_value", fi->bits_per_value}, {"quality", quality_json(id, kv)},
                        {"recommendable", fi->recommendable}, {"caveat", fi->caveat}});
        for (int flash = 0; flash < 2; ++flash) {
            if (flash && !flash_installed) continue;
            json opt = {{"kv", kv}, {"flash", flash == 1}};
            const PlanResult fast = plan_one(c, kv, flash == 1, 0);
            if (fast.ok) { native = std::max(native, fast.native_context); free_assumed = fast.vram_free_assumed; }
            opt["fast"] = point_json(c, kv, fast, true);   // the largest context with every layer and the output head in VRAM
            if (kv == "q8" || kv == "q4") {   // the formats the default benchmark measures
                ++total_fast;
                if (fast.ok && opt["fast"]["speed"]["source"] == "measured") ++covered;
            }
            json points = json::array();
            for (size_t ctx : ladder(native ? native : fast.native_context)) {
                const PlanResult p = plan_one(c, kv, flash == 1, ctx);
                json pj = point_json(c, kv, p, true);
                if (p.ok) pj["requested_context"] = ctx; else pj["context"] = ctx;
                points.push_back(pj);
            }
            opt["points"] = points;
            fasts.push_back({kv, flash == 1, fast});
            options.push_back(std::move(opt));
        }
    }
    out["formats"] = formats;
    out["options"] = options;
    out["native_context"] = native;
    out["machine"]["vram_free_assumed_mb"] = mb(free_assumed);

    // recommendation: the best-quality format (high label) whose fast profile reaches 32K tokens; else the high-quality one with the most context
    // when nothing keeps 32K fully on the GPU: fully on the GPU first, then the most layers on the GPU (that decides speed), then the most context
    auto better = [](const PlanResult& a, const PlanResult& b) {
        const bool fa = a.layers_gpu == a.layers && !a.head_on_cpu, fb = b.layers_gpu == b.layers && !b.head_on_cpu;
        if (fa != fb) return fa;
        if (a.layers_gpu != b.layers_gpu) return a.layers_gpu > b.layers_gpu;
        return a.context > b.context;
    };
    const Fast* pick = nullptr;
    auto label_of = [&](const std::string& kv) { const QualityRow* r = quality_row(id, kv); return std::string(r ? r->label : "experimental"); };
    for (const Fast& f : fasts) {
        if (f.flash || !f.p.ok || label_of(f.kv) != "high" || !format_info(f.kv)->recommendable) continue;
        if (f.p.layers_gpu == f.p.layers && !f.p.head_on_cpu && f.p.context >= kDefaultContext) { pick = &f; break; }
    }
    if (!pick)
        for (const Fast& f : fasts) {
            if (f.flash || !f.p.ok || label_of(f.kv) != "high" || !format_info(f.kv)->recommendable) continue;
            if (!pick || better(f.p, pick->p)) pick = &f;
        }
    if (!pick)
        for (const Fast& f : fasts) if (!f.flash && f.p.ok && format_info(f.kv)->recommendable && (!pick || better(f.p, pick->p))) pick = &f;
    if (pick) {
        std::string reason = std::string(kv_format_name(pick->kv)) + ": " + (label_of(pick->kv) == "high" ? "high quality" : "the best available") + ", ";
        const Fast* with_flash = nullptr;
        for (const Fast& f : fasts) if (f.flash && f.kv == pick->kv && f.p.ok) with_flash = &f;
        bool flash_on = false;
        if (with_flash && with_flash->p.flash_active && with_flash->p.layers_gpu == with_flash->p.layers && !with_flash->p.head_on_cpu && with_flash->p.context >= kDefaultContext) flash_on = true;
        const PlanResult& use = flash_on ? with_flash->p : pick->p;
        reason += std::to_string(use.context) + " tokens, ";
        if (use.layers_gpu == use.layers && !use.head_on_cpu) reason += "every layer and the output head on the GPU";
        else if (use.layers_gpu == use.layers) reason += "every layer on the GPU (the output head runs on the CPU)";
        else reason += std::to_string(use.layers_gpu) + " of " + std::to_string(use.layers) + " layers on the GPU (the rest run on the CPU: this model does not fit this GPU fully)";
        if (flash_on) reason += "; Flash keeps at least 32K tokens here, so it is on";
        else if (with_flash && with_flash->p.ok) reason += "; Flash would limit the context to " + std::to_string(with_flash->p.context) + " tokens on this GPU, so it is off";
        out["recommended"] = {{"kv", pick->kv}, {"flash", flash_on}, {"context", use.context}, {"reason", reason},
                              {"rule", "highest-quality K/V format that keeps >= 32K tokens fully on the GPU (else the high-quality format with the most layers on the GPU, then the most context); Flash on only if it keeps >= 32K tokens fully on the GPU"}};
    }

    // calibration status
    const size_t mine = rq.cal ? rq.cal->count(rq.machine, id) : 0, others = rq.cal ? rq.cal->count_other_machines(rq.machine, id) : 0;
    std::string status = covered == 0 ? (others > 0 ? "stale" : "none") : (covered == total_fast ? "measured" : "partial");
    json cal = {{"status", status}, {"entries", mine}, {"fast_options_measured", covered}, {"fast_options", total_fast},
                {"entries_from_other_machines", others}, {"file", rq.cal ? rq.cal->path() : ""},
                {"run", {{"method", "POST"}, {"url", "/dara/models/" + id + "/benchmark"}}}};
    if (rq.cal && mine) cal["measured_at"] = rq.cal->latest(rq.machine, id);
    const size_t on_battery = rq.cal ? rq.cal->count_on_battery(rq.machine, id) : 0;
    cal["entries_measured_on_battery"] = on_battery;
    cal["note"] = status == "none" ? "Speeds are measured once on your machine: run the benchmark."
                : status == "stale" ? "The saved measurements are from another GPU, driver or Dara version: run the benchmark again."
                : status == "partial" ? "Some options are measured; run the benchmark again for the rest."
                : "Speeds were measured on this machine for INT8 and Q4 (the benchmark's default formats); run it with formats k8v4 / tq4 to measure those too.";
    if (on_battery) cal["note"] = cal["note"].get<std::string>() + " Warning: " + std::to_string(on_battery) + " measurement(s) were taken on battery power, when laptop GPUs run slower: run the benchmark again on AC power."; 
    out["calibration"] = cal;
    out["notes"] = json::array({"Speeds are generation speeds with an empty context; they fall as the conversation fills (see decode_tps_full_estimate where measured).",
                                "A smaller context keeps more of the model on the GPU: that is what makes the fast profile fast."});
    return out;
}

std::vector<BenchTarget> bench_targets(const OptionsRequest& rq, bool skip_measured) {
    const Context c = make_context(rq);
    std::vector<BenchTarget> targets;
    for (const std::string& kv : wanted_formats(rq, true)) {
        for (int flash = 0; flash < 2; ++flash) {
            if (flash && c.flash_path.empty()) continue;
            size_t native = 0;
            struct Cand { size_t ctx; const char* why; };
            std::vector<BenchTarget> mine;
            auto consider = [&](size_t ctx, const char* why) {
                const PlanResult p = plan_one(c, kv, flash == 1, ctx);
                if (!p.ok) return;
                native = std::max(native, p.native_context);
                for (const BenchTarget& t : mine) if (t.layers_gpu == p.layers_gpu && t.head_cpu == p.head_on_cpu) return;   // same placement: the same speed
                if (flash == 1 && !p.flash_active) return;
                BenchTarget t;
                t.kv = kv; t.flash = flash == 1; t.context = p.context; t.layers_gpu = p.layers_gpu; t.head_cpu = p.head_on_cpu; t.why = why;
                t.depth_probe = std::string(why) == "fast";
                mine.push_back(t);
            };
            consider(0, "fast");
            consider(kDefaultContext, "default 32K");
            consider(native ? std::min(native, kMaxLadderContext) : kMaxLadderContext, "max");
            for (BenchTarget& t : mine) {
                if (skip_measured && rq.cal && rq.cal->find(rq.machine, rq.entry->id, t.kv, t.flash, t.layers_gpu, t.head_cpu)) continue;
                targets.push_back(t);
            }
        }
    }
    return targets;
}

}  // namespace dara::serve
