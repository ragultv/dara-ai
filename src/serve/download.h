// Model downloads for `dara pull` and POST /dara/models/pull: plain HTTPS range requests (WinHTTP, the system proxy), several 16 MiB chunks in
// parallel, resumable (<file>.part + <file>.part.chunks), every file checked against its size and SHA-256 before it is renamed into place.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/catalog.h"

namespace dara::serve {

struct DownloadProgress {
    std::string state;          // "queued", "running", "verifying", "done", "failed", "cancelled"
    std::string current_file, error;
    uint64_t bytes_done = 0, bytes_total = 0;
    double bytes_per_second = 0, seconds = 0;
    int files_done = 0, files_total = 0;
    std::vector<std::string> skipped;   // files already installed
};

class DownloadJob {
public:
    // files: what to fetch (with their final paths)
    DownloadJob(std::string id, std::string model, std::vector<std::pair<CatalogFile, std::string>> files, int workers = 4);
    ~DownloadJob();
    void start();
    void cancel() { cancel_ = true; }
    void wait();
    bool finished() const;
    DownloadProgress progress() const;
    const std::string& id() const { return id_; }
    const std::string& model() const { return model_; }

private:
    void run();
    void fetch(const CatalogFile& f, const std::string& path);
    std::string id_, model_;
    std::vector<std::pair<CatalogFile, std::string>> files_;
    int workers_;
    std::thread th_;
    std::atomic<bool> cancel_{false};
    std::atomic<uint64_t> done_bytes_{0};
    mutable std::mutex mu_;
    DownloadProgress p_;
    double t0_ = 0;
};

// SHA-256 of a file as lower-case hex (Windows CNG). progress(bytes) is called as it reads; returning false aborts ("" result).
std::string sha256_file(const std::string& path, const std::function<bool(uint64_t)>& progress = nullptr);

}  // namespace dara::serve
