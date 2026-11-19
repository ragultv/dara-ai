#include "serve/download.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "bcrypt.lib")

namespace fs = std::filesystem;

namespace dara::serve {
namespace {

constexpr uint64_t kChunk = 16ull << 20;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring w(static_cast<size_t>(n > 0 ? n : 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, w.data(), n);
    w.resize(wcslen(w.c_str()));
    return w;
}

struct WinHttp {
    HINTERNET h = nullptr;
    WinHttp() = default;
    explicit WinHttp(HINTERNET x) : h(x) {}
    ~WinHttp() { if (h) WinHttpCloseHandle(h); }
    WinHttp(const WinHttp&) = delete;
    WinHttp& operator=(const WinHttp&) = delete;
};

HINTERNET session() {
    static HINTERNET s = [] {
        HINTERNET x = WinHttpOpen(L"dara/1.0", WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (!x) x = WinHttpOpen(L"dara/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
        if (x) {
            DWORD proto = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | 0x00002000 /* TLS 1.3 */;
            if (!WinHttpSetOption(x, WINHTTP_OPTION_SECURE_PROTOCOLS, &proto, sizeof proto)) {
                proto = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;
                WinHttpSetOption(x, WINHTTP_OPTION_SECURE_PROTOCOLS, &proto, sizeof proto);
            }
            WinHttpSetTimeouts(x, 30000, 30000, 30000, 60000);
        }
        return x;
    }();
    return s;
}

std::string last_error(const char* what) { return std::string(what) + " failed (error " + std::to_string(GetLastError()) + ")"; }

// GET url with Range [lo, hi]; writes the body at file offset lo. Returns bytes written; throws on HTTP / network errors.
uint64_t get_range(const std::string& url, uint64_t lo, uint64_t hi, HANDLE file, std::mutex& file_mu, std::atomic<uint64_t>& counter,
                   std::atomic<uint64_t>& live, const std::atomic<bool>& cancel) {
    const std::wstring wurl = widen(url);
    URL_COMPONENTS uc{};
    uc.dwStructSize = sizeof uc;
    wchar_t host[256], path[4096];
    uc.lpszHostName = host; uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path; uc.dwUrlPathLength = 4096;
    wchar_t extra[2048]; uc.lpszExtraInfo = extra; uc.dwExtraInfoLength = 2048;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) throw std::runtime_error(last_error("parsing the URL"));
    const std::wstring full_path = std::wstring(path) + extra;
    WinHttp con(WinHttpConnect(session(), host, uc.nPort, 0));
    if (!con.h) throw std::runtime_error(last_error("connect"));
    WinHttp req(WinHttpOpenRequest(con.h, L"GET", full_path.c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES,
                                   uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0));
    if (!req.h) throw std::runtime_error(last_error("open request"));
    const std::wstring range = L"Range: bytes=" + std::to_wstring(lo) + L"-" + std::to_wstring(hi);
    if (!WinHttpSendRequest(req.h, range.c_str(), static_cast<DWORD>(-1L), WINHTTP_NO_REQUEST_DATA, 0, 0, 0) || !WinHttpReceiveResponse(req.h, nullptr))
        throw std::runtime_error(last_error("HTTP request"));
    DWORD status = 0, sz = sizeof status;
    WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz, WINHTTP_NO_HEADER_INDEX);
    if (status != 206 && !(status == 200 && lo == 0)) throw std::runtime_error("HTTP " + std::to_string(status) + (status == 404 ? " (not found or not public)" : ""));
    std::vector<char> buf(1 << 20);
    uint64_t off = lo, got = 0;
    const uint64_t want = hi - lo + 1;
    for (;;) {
        if (cancel) throw std::runtime_error("cancelled");
        DWORD n = 0;
        if (!WinHttpReadData(req.h, buf.data(), static_cast<DWORD>(buf.size()), &n)) throw std::runtime_error(last_error("read"));
        if (n == 0) break;
        const DWORD use = static_cast<DWORD>(std::min<uint64_t>(n, want - got));
        {
            std::lock_guard<std::mutex> lk(file_mu);
            LARGE_INTEGER li; li.QuadPart = static_cast<LONGLONG>(off);
            DWORD w = 0;
            if (!SetFilePointerEx(file, li, nullptr, FILE_BEGIN) || !WriteFile(file, buf.data(), use, &w, nullptr) || w != use) throw std::runtime_error(last_error("write"));
        }
        off += use; got += use;
        counter += use;
        live += use;
        if (got >= want) break;
    }
    if (got != want) throw std::runtime_error("short read (" + std::to_string(got) + " of " + std::to_string(want) + " bytes)");
    return got;
}

}  // namespace

std::string sha256_file(const std::string& path, const std::function<bool(uint64_t)>& progress) {
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE h = nullptr;
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) throw std::runtime_error("SHA-256 unavailable");
    struct Close { BCRYPT_ALG_HANDLE a; BCRYPT_HASH_HANDLE* h; ~Close() { if (*h) BCryptDestroyHash(*h); BCryptCloseAlgorithmProvider(a, 0); } } c{alg, &h};
    if (BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) != 0) throw std::runtime_error("SHA-256 init failed");
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<char> buf(8 << 20);
    uint64_t total = 0;
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const auto n = f.gcount();
        if (n <= 0) break;
        BCryptHashData(h, reinterpret_cast<PUCHAR>(buf.data()), static_cast<ULONG>(n), 0);
        total += static_cast<uint64_t>(n);
        if (progress && !progress(total)) return "";
    }
    unsigned char d[32];
    BCryptFinishHash(h, d, 32, 0);
    static const char* hx = "0123456789abcdef";
    std::string s;
    for (unsigned char b : d) { s += hx[b >> 4]; s += hx[b & 15]; }
    return s;
}

DownloadJob::DownloadJob(std::string id, std::string model, std::vector<std::pair<CatalogFile, std::string>> files, int workers)
    : id_(std::move(id)), model_(std::move(model)), files_(std::move(files)), workers_(workers) {
    p_.state = "queued";
    p_.files_total = static_cast<int>(files_.size());
    for (const auto& f : files_) p_.bytes_total += f.first.size;
}

DownloadJob::~DownloadJob() { cancel_ = true; wait(); }

void DownloadJob::start() { th_ = std::thread([this] { run(); }); }
void DownloadJob::wait() { if (th_.joinable()) th_.join(); }

bool DownloadJob::finished() const {
    std::lock_guard<std::mutex> lk(mu_);
    return p_.state == "done" || p_.state == "failed" || p_.state == "cancelled";
}

DownloadProgress DownloadJob::progress() const {
    std::lock_guard<std::mutex> lk(mu_);
    DownloadProgress p = p_;
    p.bytes_done = done_bytes_.load();
    if (p.state == "running" || p.state == "verifying") {
        p.seconds = now_s() - t0_;
    }
    return p;
}

void DownloadJob::run() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        p_.state = "running";
        t0_ = now_s();
    }
    uint64_t base = 0;
    try {
        // room on disk for everything still missing (+256 MiB)
        uint64_t need = 0;
        for (const auto& [f, path] : files_) need += f.size;
        std::error_code ec;
        const fs::path dir = fs::u8path(files_.empty() ? "." : files_[0].second).parent_path();
        fs::create_directories(dir, ec);
        const auto sp = fs::space(dir, ec);
        if (!ec && sp.available < need + (256ull << 20))
            throw std::runtime_error("not enough disk space: need " + std::to_string((need >> 20) + 256) + " MiB, " + std::to_string(sp.available >> 20) + " MiB free");
        for (const auto& [f, path] : files_) {
            { std::lock_guard<std::mutex> lk(mu_); p_.current_file = f.name; }
            fetch(f, path);
            base += f.size;
            done_bytes_ = base;
            std::lock_guard<std::mutex> lk(mu_);
            ++p_.files_done;
        }
        std::lock_guard<std::mutex> lk(mu_);
        p_.state = "done";
        p_.current_file.clear();
        p_.seconds = now_s() - t0_;
        p_.bytes_per_second = p_.seconds > 0 ? p_.bytes_total / p_.seconds : 0;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(mu_);
        p_.state = cancel_ ? "cancelled" : "failed";
        p_.error = cancel_ ? "" : e.what();
        p_.seconds = now_s() - t0_;
    }
}

void DownloadJob::fetch(const CatalogFile& f, const std::string& path) {
    const fs::path final_p = fs::u8path(path);
    fs::path part = final_p, chunks = final_p;
    part += ".part";
    chunks += ".part.chunks";
    std::error_code ec;
    fs::create_directories(final_p.parent_path(), ec);
    const uint64_t base = done_bytes_.load();
    const uint64_t n_chunks = (f.size + kChunk - 1) / kChunk;
    std::set<uint64_t> done;
    if (fs::exists(part, ec) && fs::file_size(part, ec) == f.size) {   // resume
        std::ifstream cf(chunks);
        uint64_t i;
        while (cf >> i) done.insert(i);
    }
    HANDLE h = CreateFileW(part.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error(last_error(("create " + path + ".part").c_str()));
    struct CloseH { HANDLE h; ~CloseH() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); } } ch{h};
    if (done.empty()) {
        LARGE_INTEGER li; li.QuadPart = static_cast<LONGLONG>(f.size);
        if (!SetFilePointerEx(h, li, nullptr, FILE_BEGIN) || !SetEndOfFile(h)) throw std::runtime_error(last_error("allocate file"));
    }
    std::atomic<uint64_t> counter{0};
    for (uint64_t i : done) counter += std::min(kChunk, f.size - i * kChunk);
    done_bytes_ = base + counter.load();
    std::mutex file_mu, list_mu;
    std::atomic<uint64_t> next{0};
    std::string err;
    std::vector<std::thread> ws;
    std::atomic<bool> failed{false};
    auto work = [&] {
        for (;;) {
            uint64_t i;
            {
                std::lock_guard<std::mutex> lk(list_mu);
                do { i = next++; } while (i < n_chunks && done.count(i));
            }
            if (i >= n_chunks || failed || cancel_) return;
            const uint64_t lo = i * kChunk, hi = std::min(f.size, lo + kChunk) - 1;
            bool ok = false;
            std::string e;
            for (int attempt = 0; attempt < 10 && !ok && !cancel_; ++attempt) {
                std::atomic<uint64_t> c{0};   // this attempt's bytes (also added to done_bytes_ live; taken back if it fails)
                try {
                    get_range(f.url, lo, hi, h, file_mu, c, done_bytes_, cancel_);
                    ok = true;
                    counter += c.load();
                } catch (const std::exception& ex) {
                    done_bytes_ -= c.load();
                    e = ex.what();
                    if (e.rfind("HTTP 4", 0) == 0) break;   // not found / forbidden: retrying will not help
                    Sleep(static_cast<DWORD>(1000 * (attempt + 1)));
                }
            }
            if (!ok) {
                std::lock_guard<std::mutex> lk(list_mu);
                if (err.empty()) err = f.name + ": " + (cancel_ ? std::string("cancelled") : e);
                failed = true;
                return;
            }
            std::lock_guard<std::mutex> lk(list_mu);
            done.insert(i);
            std::ofstream cf(chunks, std::ios::app);
            cf << i << "\n";
        }
    };
    for (int k = 0; k < std::max(1, workers_); ++k) ws.emplace_back(work);
    for (auto& t : ws) t.join();
    if (cancel_) throw std::runtime_error("cancelled");
    if (failed) throw std::runtime_error(err);
    FlushFileBuffers(h);
    CloseHandle(h);
    ch.h = INVALID_HANDLE_VALUE;
    {
        std::lock_guard<std::mutex> lk(mu_);
        p_.state = "verifying";
    }
    const auto pu = part.u8string();
    const std::string got = sha256_file(std::string(pu.begin(), pu.end()), [&](uint64_t) { return !cancel_.load(); });
    if (cancel_) throw std::runtime_error("cancelled");
    if (got != f.sha256) {
        fs::remove(part, ec);
        fs::remove(chunks, ec);
        throw std::runtime_error(f.name + ": SHA-256 mismatch (got " + got + "), the partial file was deleted");
    }
    fs::remove(chunks, ec);
    fs::rename(part, final_p, ec);
    if (ec) throw std::runtime_error("cannot move " + f.name + " into place: " + ec.message());
    std::lock_guard<std::mutex> lk(mu_);
    p_.state = "running";
}

}  // namespace dara::serve
