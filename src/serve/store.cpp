#include "serve/store.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "httplib/httplib.h"
#include "sqlite/sqlite3.h"
#include "serve/store_schema.h"

namespace dara::serve {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// kStoreMigrationSql[i] (store_schema.h, from docs/ui/db/000N_*.sql) upgrades schema version i+1 to i+2 (forward only; an applied migration is never edited).
constexpr int kSchemaLatest = 1 + kStoreMigrationCount;

constexpr size_t kMaxContent = size_t(2) << 20;       // 2 MB of text per message
constexpr size_t kMaxAttachment = size_t(20) << 20;   // 20 MB per image
constexpr size_t kMaxSetting = 64 << 10;
constexpr int kSnippetChars = 600;   // how much of an artifact the gallery list carries (the card preview)
constexpr int64_t kVersionGapMs = 120000;   // an edit more than this long after the last save snapshots the earlier text first
constexpr int kVersionCap = 200;            // versions kept per artifact
constexpr int kVersionLabelMax = 120;
constexpr int kVersionSnippetChars = 200;
constexpr size_t kMaxChatData = size_t(200) << 10;        // a saved chat message (serialized JSON)
constexpr int kChatCap = 500;                             // chat messages kept per artifact
constexpr int kCommentBodyMax = 8000, kCommentQuoteMax = 500;   // characters
constexpr int kCommentsPerArtifact = 1000, kRepliesPerThread = 100;
constexpr size_t kAssetMax = size_t(25) << 20;            // one image or video
constexpr size_t kAssetsPerArtifact = size_t(200) << 20;
constexpr size_t kAssetNameMax = 200;                     // characters

// Both limits can be lowered for tests with DARA_VERSION_GAP_MS and DARA_VERSION_CAP (read once; not a user setting).
int64_t version_gap_ms() {
    static const int64_t v = [] {
        const char* e = std::getenv("DARA_VERSION_GAP_MS");
        if (!e || !*e) return kVersionGapMs;
        char* end = nullptr;
        const long long n = std::strtoll(e, &end, 10);
        return (end && !*end && n >= 0) ? static_cast<int64_t>(n) : kVersionGapMs;
    }();
    return v;
}
int version_cap() {
    static const int v = [] {
        const char* e = std::getenv("DARA_VERSION_CAP");
        if (!e || !*e) return kVersionCap;
        char* end = nullptr;
        const long n = std::strtol(e, &end, 10);
        return (end && !*end && n >= 1) ? static_cast<int>(n) : kVersionCap;
    }();
    return v;
}

// Test-only overrides of three limits above: DARA_CHAT_CAP, DARA_ASSET_MAX_BYTES, DARA_ASSET_ARTIFACT_MAX_BYTES (read once; not a user setting).
long long env_limit(const char* name, long long def, long long min) {
    const char* e = std::getenv(name);
    if (!e || !*e) return def;
    char* end = nullptr;
    const long long n = std::strtoll(e, &end, 10);
    return (end && !*end && n >= min) ? n : def;
}
int chat_cap() { static const int v = static_cast<int>(env_limit("DARA_CHAT_CAP", kChatCap, 1)); return v; }
size_t asset_max() { static const size_t v = static_cast<size_t>(env_limit("DARA_ASSET_MAX_BYTES", static_cast<long long>(kAssetMax), 1)); return v; }
size_t asset_artifact_max() { static const size_t v = static_cast<size_t>(env_limit("DARA_ASSET_ARTIFACT_MAX_BYTES", static_cast<long long>(kAssetsPerArtifact), 1)); return v; }

std::string u8s(const fs::path& p) { const auto s = p.u8string(); return std::string(s.begin(), s.end()); }
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }
int64_t now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }

std::string new_uuid() {
    static std::mutex mu;
    static std::mt19937_64 rng = [] { std::random_device rd; return std::mt19937_64((uint64_t(rd()) << 32) ^ rd() ^ (uint64_t(rd()) << 16)); }();
    uint8_t b[16];
    { std::lock_guard<std::mutex> lk(mu); for (int i = 0; i < 2; ++i) { const uint64_t v = rng(); std::memcpy(b + 8 * i, &v, 8); } }
    b[6] = static_cast<uint8_t>((b[6] & 0x0F) | 0x40);
    b[8] = static_cast<uint8_t>((b[8] & 0x3F) | 0x80);
    char out[37];
    std::snprintf(out, sizeof out, "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return out;
}

bool is_uuid(const std::string& s) {
    if (s.size() != 36) return false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) { if (s[i] != '-') return false; }
        else if (!std::isxdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

bool valid_key(const std::string& k) {
    if (k.empty() || k.size() > 64) return false;
    for (char c : k) if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-')) return false;
    return true;
}

[[noreturn]] void fail(int status, const std::string& msg, const char* type = "invalid_request_error") { throw StoreError(status, msg, type); }

// ---- SQLite helpers ----
struct Stmt {
    sqlite3_stmt* s = nullptr;
    sqlite3* db;
    Stmt(sqlite3* d, const char* sql) : db(d) {
        if (sqlite3_prepare_v2(d, sql, -1, &s, nullptr) != SQLITE_OK) throw std::runtime_error(std::string("sql: ") + sqlite3_errmsg(d));
    }
    ~Stmt() { sqlite3_finalize(s); }
    Stmt(const Stmt&) = delete;
    Stmt& operator=(const Stmt&) = delete;
    Stmt& text(int i, const std::string& v) { sqlite3_bind_text(s, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT); return *this; }
    Stmt& opt_text(int i, const json& v) { if (v.is_string()) return text(i, v.get<std::string>()); sqlite3_bind_null(s, i); return *this; }
    Stmt& num(int i, int64_t v) { sqlite3_bind_int64(s, i, v); return *this; }
    Stmt& null(int i) { sqlite3_bind_null(s, i); return *this; }
    Stmt& blob(int i, const std::string& v) { sqlite3_bind_blob64(s, i, v.data(), static_cast<sqlite3_uint64>(v.size()), SQLITE_STATIC); return *this; }   // `v` must outlive the step
    bool row() {
        const int rc = sqlite3_step(s);
        if (rc == SQLITE_ROW) return true;
        if (rc == SQLITE_DONE) return false;
        throw std::runtime_error(std::string("sql: ") + sqlite3_errmsg(db));
    }
    void run() { row(); }
    std::string str(int c) const { const unsigned char* t = sqlite3_column_text(s, c); return t ? std::string(reinterpret_cast<const char*>(t), static_cast<size_t>(sqlite3_column_bytes(s, c))) : std::string(); }
    int64_t i64(int c) const { return sqlite3_column_int64(s, c); }
    std::string bytes(int c) const { const void* b = sqlite3_column_blob(s, c); return b ? std::string(static_cast<const char*>(b), static_cast<size_t>(sqlite3_column_bytes(s, c))) : std::string(); }
    bool is_null(int c) const { return sqlite3_column_type(s, c) == SQLITE_NULL; }
};

void exec(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        const std::string m = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("sql: " + m);
    }
}

struct Tx {   // BEGIN IMMEDIATE ... COMMIT, rolled back when destroyed uncommitted
    sqlite3* db;
    bool done = false;
    explicit Tx(sqlite3* d) : db(d) { exec(db, "BEGIN IMMEDIATE"); }
    void commit() { exec(db, "COMMIT"); done = true; }
    ~Tx() { if (!done) sqlite3_exec(db, "ROLLBACK", nullptr, nullptr, nullptr); }
};

json parse_or_null(const std::string& s) { json j = json::parse(s, nullptr, false); return j.is_discarded() ? json(nullptr) : j; }

std::string sniff_mime(const std::string& b) {
    auto at = [&](size_t i) { return i < b.size() ? static_cast<unsigned char>(b[i]) : 0; };
    if (b.size() >= 8 && at(0) == 0x89 && b.compare(1, 3, "PNG") == 0) return "image/png";
    if (b.size() >= 3 && at(0) == 0xFF && at(1) == 0xD8 && at(2) == 0xFF) return "image/jpeg";
    if (b.size() >= 12 && b.compare(0, 4, "RIFF") == 0 && b.compare(8, 4, "WEBP") == 0) return "image/webp";
    return "";
}

const char* ext_of(const std::string& mime) { return mime == "image/png" ? "png" : mime == "image/jpeg" ? "jpg" : "webp"; }

std::string need_string(const json& b, const char* key, size_t max, bool required = false) {
    if (!b.contains(key) || b[key].is_null()) { if (required) fail(400, std::string("'") + key + "' is required"); return ""; }
    if (!b[key].is_string()) fail(400, std::string("'") + key + "' must be a string");
    const std::string s = b[key].get<std::string>();
    if (s.size() > max) fail(400, std::string("'") + key + "' is too long");
    return s;
}

size_t utf8_len(const std::string& s) { size_t n = 0; for (unsigned char c : s) if ((c & 0xC0) != 0x80) ++n; return n; }

// A string field limited in characters (not bytes). `required` = present and not blank.
std::string need_text(const json& b, const char* key, size_t max_chars, bool required) {
    const std::string s = need_string(b, key, max_chars * 4, required);
    if (utf8_len(s) > max_chars) fail(400, std::string("'") + key + "' is too long (at most " + std::to_string(max_chars) + " characters)");
    if (required && s.find_first_not_of(" \t\r\n") == std::string::npos) fail(400, std::string("'") + key + "' must not be empty");
    return s;
}

std::string url_decode(const std::string& in) {   // %XX escapes (the X-Dara-Name header); anything else is kept as it is
    auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    std::string out;
    for (size_t i = 0; i < in.size(); ++i) {
        if (in[i] == '%' && i + 2 < in.size() && hex(in[i + 1]) >= 0 && hex(in[i + 2]) >= 0) { out.push_back(static_cast<char>(hex(in[i + 1]) * 16 + hex(in[i + 2]))); i += 2; }
        else out.push_back(in[i]);
    }
    return out;
}

// What a stored image or video may be: the declared type has to match the first bytes (a mislabeled file is refused, not stored).
bool asset_type_ok(const std::string& mime) {
    static const char* const ok[] = {"image/png", "image/jpeg", "image/gif", "image/webp", "video/mp4", "video/webm"};
    for (const char* m : ok) if (mime == m) return true;
    return false;
}
bool asset_magic_ok(const std::string& mime, const std::string& b) {
    auto at = [&](size_t i) { return i < b.size() ? static_cast<unsigned char>(b[i]) : 0; };
    if (mime == "image/png") return b.size() >= 8 && at(0) == 0x89 && b.compare(1, 3, "PNG") == 0 && at(4) == 0x0D && at(5) == 0x0A && at(6) == 0x1A && at(7) == 0x0A;
    if (mime == "image/jpeg") return b.size() >= 3 && at(0) == 0xFF && at(1) == 0xD8 && at(2) == 0xFF;
    if (mime == "image/gif") return b.size() >= 6 && (b.compare(0, 6, "GIF87a") == 0 || b.compare(0, 6, "GIF89a") == 0);
    if (mime == "image/webp") return b.size() >= 12 && b.compare(0, 4, "RIFF") == 0 && b.compare(8, 4, "WEBP") == 0;
    if (mime == "video/mp4") return b.size() >= 12 && b.compare(4, 4, "ftyp") == 0;
    if (mime == "video/webm") return b.size() >= 4 && at(0) == 0x1A && at(1) == 0x45 && at(2) == 0xDF && at(3) == 0xA3;
    return false;
}

json params_of(const json& b) {
    if (!b.contains("params") || b["params"].is_null()) return json::object();
    if (!b["params"].is_object()) fail(400, "'params' must be an object");
    if (dump(b["params"]).size() > kMaxSetting) fail(400, "'params' is too large");
    return b["params"];
}

}  // namespace

struct Store::Impl {
    sqlite3* db = nullptr;
    fs::path dir, file, attachments;
    std::string note;

    ~Impl() { if (db) sqlite3_close(db); }

    void open_db() {
        if (sqlite3_open_v2(u8s(file).c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK) {
            const std::string m = db ? sqlite3_errmsg(db) : "out of memory";
            if (db) { sqlite3_close(db); db = nullptr; }
            throw std::runtime_error("cannot open " + u8s(file) + ": " + m);
        }
        sqlite3_extended_result_codes(db, 1);
    }
    bool integrity_ok() {
        try { Stmt q(db, "PRAGMA quick_check(1)"); return q.row() && q.str(0) == "ok"; } catch (const std::exception&) { return false; }
    }
    int user_version() { Stmt q(db, "PRAGMA user_version"); q.row(); return static_cast<int>(q.i64(0)); }
    void set_user_version(int v) { exec(db, ("PRAGMA user_version = " + std::to_string(v)).c_str()); }

    void remove_files(const std::vector<std::string>& names) {
        for (const auto& n : names) {
            if (n.empty() || n.find_first_of("/\\") != std::string::npos || n.find("..") != std::string::npos) continue;
            std::error_code ec; fs::remove(attachments / fs::u8path(n), ec);
        }
    }
    std::vector<std::string> files_of_conversation(const std::string& cid) {
        std::vector<std::string> v;
        Stmt q(db, "SELECT a.file FROM attachments a JOIN messages m ON m.id=a.message_id WHERE m.conversation_id=?1");
        q.text(1, cid);
        while (q.row()) v.push_back(q.str(0));
        return v;
    }

    json attachments_of(const std::string& mid) {
        json arr = json::array();
        Stmt q(db, "SELECT id,mime,bytes,width,height FROM attachments WHERE message_id=?1 ORDER BY rowid");
        q.text(1, mid);
        while (q.row()) arr.push_back({{"id", q.str(0)}, {"mime", q.str(1)}, {"bytes", q.i64(2)}, {"url", "/dara/attachments/" + q.str(0)},
                                       {"width", q.is_null(3) ? json(nullptr) : json(q.i64(3))}, {"height", q.is_null(4) ? json(nullptr) : json(q.i64(4))}});
        return arr;
    }
    static json message_row(Stmt& q) {   // columns: id,conversation_id,position,role,content,finish_reason,error,stats,model,created_at
        return {{"id", q.str(0)}, {"conversation_id", q.str(1)}, {"position", q.i64(2)}, {"role", q.str(3)}, {"content", q.str(4)},
                {"finish_reason", q.is_null(5) ? json(nullptr) : json(q.str(5))}, {"error", q.is_null(6) ? json(nullptr) : json(q.str(6))},
                {"stats", q.is_null(7) ? json(nullptr) : parse_or_null(q.str(7))}, {"model", q.is_null(8) ? json(nullptr) : json(q.str(8))}, {"created_at", q.i64(9)}};
    }
    json message_by_id(const std::string& id) {
        Stmt q(db, "SELECT id,conversation_id,position,role,content,finish_reason,error,stats,model,created_at FROM messages WHERE id=?1");
        q.text(1, id);
        if (!q.row()) fail(404, "message not found", "not_found");
        json m = message_row(q);
        m["attachments"] = attachments_of(id);
        return m;
    }
    static json conversation_row(Stmt& q) {   // columns: id,title,model,system_prompt,params,pinned,created_at,updated_at,message_count
        return {{"id", q.str(0)}, {"title", q.str(1)}, {"model", q.is_null(2) ? json(nullptr) : json(q.str(2))}, {"system_prompt", q.str(3)}, {"params", parse_or_null(q.str(4))},
                {"pinned", q.i64(5) != 0}, {"created_at", q.i64(6)}, {"updated_at", q.i64(7)}, {"message_count", q.i64(8)}};
    }
    json conversation_by_id(const std::string& id) {
        Stmt q(db, "SELECT id,title,model,system_prompt,params,pinned,created_at,updated_at,(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=conversations.id) FROM conversations WHERE id=?1");
        q.text(1, id);
        if (!q.row()) fail(404, "conversation not found", "not_found");
        return conversation_row(q);
    }
    static json artifact_row(Stmt& q, bool full) {   // columns: id,kind,title,pinned,created_at,updated_at,length(content),content-or-snippet
        json j = {{"id", q.str(0)}, {"kind", q.str(1)}, {"title", q.str(2)}, {"pinned", q.i64(3) != 0}, {"created_at", q.i64(4)}, {"updated_at", q.i64(5)}, {"chars", q.i64(6)}};
        j[full ? "content" : "snippet"] = q.str(7);
        return j;
    }
    json artifact_by_id(const std::string& id) {
        Stmt q(db, "SELECT id,kind,title,pinned,created_at,updated_at,length(content),content FROM artifacts WHERE id=?1");
        q.text(1, id);
        if (!q.row()) fail(404, "artifact not found", "not_found");
        return artifact_row(q, true);
    }
    struct ArtifactState { std::string title, content; int64_t updated_at = 0; };
    ArtifactState artifact_state(const std::string& id) {   // 404 when absent
        Stmt q(db, "SELECT title,content,updated_at FROM artifacts WHERE id=?1");
        q.text(1, id);
        if (!q.row()) fail(404, "artifact not found", "not_found");
        return {q.str(0), q.str(1), q.i64(2)};
    }
    // Stores `cur` as the next version of the artifact. Skipped when it is empty or identical to the newest version. Caller holds the lock and a transaction.
    bool snapshot_state(const std::string& id, const ArtifactState& cur, const std::string& label) {
        if (cur.title.empty() && cur.content.empty()) return false;
        int64_t next = 1;
        {
            Stmt q(db, "SELECT rev,title,content FROM artifact_versions WHERE artifact_id=?1 ORDER BY rev DESC LIMIT 1");
            q.text(1, id);
            if (q.row()) {
                if (q.str(1) == cur.title && q.str(2) == cur.content) return false;
                next = q.i64(0) + 1;
            }
        }
        { Stmt q(db, "INSERT INTO artifact_versions(artifact_id,rev,title,content,label,created_at) VALUES(?1,?2,?3,?4,?5,?6)"); q.text(1, id).num(2, next).text(3, cur.title).text(4, cur.content).text(5, label).num(6, cur.updated_at).run(); }
        { Stmt q(db, "DELETE FROM artifact_versions WHERE artifact_id=?1 AND rev NOT IN (SELECT rev FROM artifact_versions WHERE artifact_id=?1 ORDER BY rev DESC LIMIT ?2)"); q.text(1, id).num(2, version_cap()).run(); }
        return true;
    }
    void touch(const std::string& cid) { Stmt q(db, "UPDATE conversations SET updated_at=?2 WHERE id=?1"); q.text(1, cid).num(2, now_ms()).run(); }

    // ---- what belongs to an artifact (docs/ui/10-artifact-collab.md) ----
    static json chat_row(Stmt& q) {   // columns: id,seq,data,created_at,updated_at
        return {{"id", q.str(0)}, {"seq", q.i64(1)}, {"data", parse_or_null(q.str(2))}, {"created_at", q.i64(3)}, {"updated_at", q.i64(4)}};
    }
    json chat_rows(const std::string& aid) {
        json arr = json::array();
        Stmt q(db, "SELECT id,seq,data,created_at,updated_at FROM artifact_chat WHERE artifact_id=?1 ORDER BY seq");
        q.text(1, aid);
        while (q.row()) arr.push_back(chat_row(q));
        return arr;
    }
    static json comment_row(Stmt& q) {   // columns: id,parent_id,author,body,quote,nth,resolved,created_at,updated_at
        return {{"id", q.str(0)}, {"parent_id", q.is_null(1) ? json(nullptr) : json(q.str(1))}, {"author", q.str(2)}, {"body", q.str(3)}, {"quote", q.str(4)}, {"nth", q.i64(5)},
                {"resolved", q.i64(6) != 0}, {"created_at", q.i64(7)}, {"updated_at", q.i64(8)}};
    }
    json comment_rows(const std::string& aid) {
        json arr = json::array();
        Stmt q(db, "SELECT id,parent_id,author,body,quote,nth,resolved,created_at,updated_at FROM artifact_comments WHERE artifact_id=?1 ORDER BY created_at, rowid");
        q.text(1, aid);
        while (q.row()) arr.push_back(comment_row(q));
        return arr;
    }
    json comment_by_id(const std::string& aid, const std::string& cid) {
        Stmt q(db, "SELECT id,parent_id,author,body,quote,nth,resolved,created_at,updated_at FROM artifact_comments WHERE id=?1 AND artifact_id=?2");
        q.text(1, cid).text(2, aid);
        if (!q.row()) fail(404, "comment not found", "not_found");
        return comment_row(q);
    }
    json asset_rows(const std::string& aid) {   // metadata only
        json arr = json::array();
        Stmt q(db, "SELECT id,name,mime,size,created_at FROM artifact_assets WHERE artifact_id=?1 ORDER BY created_at, rowid");
        q.text(1, aid);
        while (q.row()) arr.push_back({{"id", q.str(0)}, {"name", q.str(1)}, {"mime", q.str(2)}, {"size", q.i64(3)}, {"created_at", q.i64(4)}});
        return arr;
    }
    void delete_artifact_children(const std::string& aid) {
        for (const char* sql : {"DELETE FROM artifact_versions WHERE artifact_id=?1", "DELETE FROM artifact_chat WHERE artifact_id=?1", "DELETE FROM artifact_comments WHERE artifact_id=?1",
                                "DELETE FROM artifact_assets WHERE artifact_id=?1"}) { Stmt q(db, sql); q.text(1, aid).run(); }
    }
};

std::string Store::default_data_dir() {
    if (const char* e = std::getenv("DARA_DATA_DIR"); e && *e) return e;
    if (const char* l = std::getenv("LOCALAPPDATA"); l && *l) return u8s(fs::u8path(l) / "Dara");
    return "dara-data";
}

Store::Store(const std::string& data_dir) : impl_(new Impl), dir_(data_dir) {
    impl_->dir = fs::u8path(data_dir);
    impl_->file = impl_->dir / "dara.db";
    impl_->attachments = impl_->dir / "attachments";
    std::error_code ec;
    fs::create_directories(impl_->attachments, ec);
    if (ec) throw std::runtime_error("cannot create " + u8s(impl_->attachments) + ": " + ec.message());
    impl_->open_db();
    if (!impl_->integrity_ok()) {   // keep the damaged file, start a fresh database
        sqlite3_close(impl_->db); impl_->db = nullptr;
        const std::string tag = std::to_string(now_ms());
        const fs::path moved = impl_->dir / ("dara.db.corrupt-" + tag);
        fs::rename(impl_->file, moved, ec);
        if (ec) throw std::runtime_error("the database is damaged and cannot be set aside: " + ec.message());
        for (const char* suffix : {"-wal", "-shm"}) { std::error_code e2; fs::remove(fs::u8path(u8s(impl_->file) + suffix), e2); }
        impl_->note = "the previous database failed its integrity check and was kept as " + u8s(moved.filename());
        std::fprintf(stderr, "WARNING: %s\n", impl_->note.c_str());
        impl_->open_db();
    }
    exec(impl_->db, "PRAGMA journal_mode=WAL");
    exec(impl_->db, "PRAGMA synchronous=NORMAL");
    exec(impl_->db, "PRAGMA foreign_keys=ON");
    exec(impl_->db, "PRAGMA busy_timeout=5000");
    exec(impl_->db, "PRAGMA cache_size=-2000");   // about 2 MB: the database adds almost nothing to the server's memory
    int ver = impl_->user_version();
    if (ver == 0) {
        Tx tx(impl_->db);
        exec(impl_->db, kStoreSchemaV1);
        impl_->set_user_version(1);
        tx.commit();
        ver = 1;
    }
    if (ver > kSchemaLatest) throw std::runtime_error("the database was written by a newer Dara (schema " + std::to_string(ver) + "); update Dara or use another --data-dir");
    while (ver < kSchemaLatest) {   // forward-only migrations, after a backup copy
        std::error_code e2;
        fs::copy_file(impl_->file, fs::u8path(u8s(impl_->file) + ".bak-v" + std::to_string(ver)), fs::copy_options::overwrite_existing, e2);
        Tx tx(impl_->db);
        exec(impl_->db, kStoreMigrationSql[ver - 1]);
        ++ver;
        impl_->set_user_version(ver);
        tx.commit();
    }
}

Store::~Store() {
    if (impl_ && impl_->db) sqlite3_exec(impl_->db, "PRAGMA wal_checkpoint(TRUNCATE)", nullptr, nullptr, nullptr);
}

json Store::info() {
    std::lock_guard<std::mutex> lk(mu_);
    auto count = [&](const char* table) { Stmt q(impl_->db, (std::string("SELECT COUNT(*) FROM ") + table).c_str()); q.row(); return q.i64(0); };
    std::error_code ec;
    const auto size = fs::file_size(impl_->file, ec);
    json j = {{"path", u8s(impl_->file)}, {"attachments_dir", u8s(impl_->attachments)}, {"schema_version", impl_->user_version()}, {"size_bytes", ec ? 0 : static_cast<int64_t>(size)},
              {"conversations", count("conversations")}, {"messages", count("messages")}, {"attachments", count("attachments")}, {"settings", count("settings")},
              {"artifacts", count("artifacts")}, {"artifact_comments", count("artifact_comments")}, {"artifact_assets", count("artifact_assets")}};
    if (!impl_->note.empty()) j["note"] = impl_->note;
    return j;
}

// ---- settings ----
json Store::settings_all() {
    std::lock_guard<std::mutex> lk(mu_);
    json o = json::object();
    Stmt q(impl_->db, "SELECT key,value FROM settings ORDER BY key");
    while (q.row()) o[q.str(0)] = parse_or_null(q.str(1));
    return o;
}

void Store::settings_set(const json& object) {
    if (!object.is_object() || object.empty()) fail(400, "send a JSON object of {key: value}");
    if (object.size() > 100) fail(400, "too many settings in one request");
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!valid_key(it.key())) fail(400, "invalid setting key '" + it.key() + "' (lowercase letters, digits, '_', '.', '-'; at most 64)");
        if (dump(it.value()).size() > kMaxSetting) fail(400, "setting '" + it.key() + "' is too large");
    }
    std::lock_guard<std::mutex> lk(mu_);
    Tx tx(impl_->db);
    for (auto it = object.begin(); it != object.end(); ++it) {
        Stmt q(impl_->db, "INSERT INTO settings(key,value,updated_at) VALUES(?1,?2,?3) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at");
        q.text(1, it.key()).text(2, dump(it.value())).num(3, now_ms()).run();
    }
    tx.commit();
}

bool Store::settings_delete(const std::string& key) {
    if (!valid_key(key)) fail(400, "invalid setting key");
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "DELETE FROM settings WHERE key=?1");
    q.text(1, key).run();
    return sqlite3_changes(impl_->db) > 0;
}

// ---- conversations ----
json Store::conversations_list(int limit, int offset, bool pinned_first) {
    limit = std::clamp(limit, 1, 500);
    offset = std::max(offset, 0);
    std::lock_guard<std::mutex> lk(mu_);
    const char* sql = pinned_first
        ? "SELECT id,title,model,system_prompt,params,pinned,created_at,updated_at,(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=conversations.id) FROM conversations ORDER BY pinned DESC, updated_at DESC LIMIT ?1 OFFSET ?2"
        : "SELECT id,title,model,system_prompt,params,pinned,created_at,updated_at,(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=conversations.id) FROM conversations ORDER BY updated_at DESC LIMIT ?1 OFFSET ?2";
    Stmt q(impl_->db, sql);
    q.num(1, limit).num(2, offset);
    json arr = json::array();
    while (q.row()) arr.push_back(Impl::conversation_row(q));
    Stmt c(impl_->db, "SELECT COUNT(*) FROM conversations");
    c.row();
    return {{"object", "list"}, {"data", arr}, {"total", c.i64(0)}, {"limit", limit}, {"offset", offset}};
}

json Store::conversation_get(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    json c = impl_->conversation_by_id(id);
    json msgs = json::array();
    {
        Stmt q(impl_->db, "SELECT id,conversation_id,position,role,content,finish_reason,error,stats,model,created_at FROM messages WHERE conversation_id=?1 ORDER BY position");
        q.text(1, id);
        while (q.row()) msgs.push_back(Impl::message_row(q));
    }
    for (auto& m : msgs) m["attachments"] = impl_->attachments_of(m["id"].get<std::string>());
    c["messages"] = msgs;
    return c;
}

json Store::conversation_create(const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::string id = need_string(b, "id", 36);
    if (id.empty()) id = new_uuid();
    else if (!is_uuid(id)) fail(400, "'id' must be a UUID");
    const std::string title = need_string(b, "title", 500), sys = need_string(b, "system_prompt", kMaxContent);
    const json params = params_of(b);
    std::lock_guard<std::mutex> lk(mu_);
    {
        Stmt dup(impl_->db, "SELECT 1 FROM conversations WHERE id=?1");
        dup.text(1, id);
        if (dup.row()) fail(409, "a conversation with this id exists", "conflict");
    }
    const int64_t t = b.contains("created_at") && b["created_at"].is_number_integer() ? b["created_at"].get<int64_t>() : now_ms();
    Stmt q(impl_->db, "INSERT INTO conversations(id,title,model,system_prompt,params,pinned,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?7)");
    q.text(1, id).text(2, title).opt_text(3, b.value("model", json(nullptr))).text(4, sys).text(5, dump(params)).num(6, b.value("pinned", false) ? 1 : 0).num(7, t).run();
    return impl_->conversation_by_id(id);
}

json Store::conversation_patch(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->conversation_by_id(id);   // 404 when absent
    Tx tx(impl_->db);
    if (b.contains("title")) { Stmt q(impl_->db, "UPDATE conversations SET title=?2 WHERE id=?1"); q.text(1, id).text(2, need_string(b, "title", 500)).run(); }
    if (b.contains("model")) { Stmt q(impl_->db, "UPDATE conversations SET model=?2 WHERE id=?1"); q.text(1, id).opt_text(2, b["model"]).run(); }
    if (b.contains("system_prompt")) { Stmt q(impl_->db, "UPDATE conversations SET system_prompt=?2 WHERE id=?1"); q.text(1, id).text(2, need_string(b, "system_prompt", kMaxContent)).run(); }
    if (b.contains("params")) { Stmt q(impl_->db, "UPDATE conversations SET params=?2 WHERE id=?1"); q.text(1, id).text(2, dump(params_of(b))).run(); }
    if (b.contains("pinned")) { Stmt q(impl_->db, "UPDATE conversations SET pinned=?2 WHERE id=?1"); q.text(1, id).num(2, b["pinned"].is_boolean() && b["pinned"].get<bool>() ? 1 : 0).run(); }
    impl_->touch(id);
    tx.commit();
    return impl_->conversation_by_id(id);
}

bool Store::conversation_delete(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    const auto files = impl_->files_of_conversation(id);
    Tx tx(impl_->db);
    Stmt q(impl_->db, "DELETE FROM conversations WHERE id=?1");
    q.text(1, id).run();
    const bool deleted = sqlite3_changes(impl_->db) > 0;
    tx.commit();
    if (deleted) impl_->remove_files(files);
    return deleted;
}

// ---- artifacts ----
namespace {
bool valid_kind(const std::string& k) { return k == "doc" || k == "slides" || k == "design"; }
}

json Store::artifacts_list(const std::string& kind, int limit, int offset) {
    limit = std::clamp(limit, 1, 500);
    offset = std::max(offset, 0);
    if (!kind.empty() && !valid_kind(kind)) fail(400, "'kind' must be doc, slides or design");
    std::lock_guard<std::mutex> lk(mu_);
    const std::string where = kind.empty() ? "" : " WHERE kind=?3";
    Stmt q(impl_->db, ("SELECT id,kind,title,pinned,created_at,updated_at,length(content),substr(content,1," + std::to_string(kSnippetChars) + ") FROM artifacts" + where +
                       " ORDER BY pinned DESC, updated_at DESC LIMIT ?1 OFFSET ?2").c_str());
    q.num(1, limit).num(2, offset);
    if (!kind.empty()) q.text(3, kind);
    json arr = json::array();
    while (q.row()) arr.push_back(Impl::artifact_row(q, false));
    Stmt c(impl_->db, ("SELECT COUNT(*) FROM artifacts" + where).c_str());
    if (!kind.empty()) c.text(3, kind);   // bound by the same index: ?3 only exists in the filtered statement
    c.row();
    return {{"object", "list"}, {"data", arr}, {"total", c.i64(0)}, {"limit", limit}, {"offset", offset}};
}

json Store::artifact_get(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    return impl_->artifact_by_id(id);
}

json Store::artifact_create(const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::string id = need_string(b, "id", 36);
    if (id.empty()) id = new_uuid();
    else if (!is_uuid(id)) fail(400, "'id' must be a UUID");
    const std::string kind = b.contains("kind") ? need_string(b, "kind", 16, true) : std::string("doc");
    if (!valid_kind(kind)) fail(400, "'kind' must be doc, slides or design");
    const std::string title = need_string(b, "title", 500), content = need_string(b, "content", kMaxContent);
    std::lock_guard<std::mutex> lk(mu_);
    {
        Stmt dup(impl_->db, "SELECT 1 FROM artifacts WHERE id=?1");
        dup.text(1, id);
        if (dup.row()) fail(409, "an artifact with this id exists", "conflict");
    }
    const int64_t t = b.contains("created_at") && b["created_at"].is_number_integer() ? b["created_at"].get<int64_t>() : now_ms();
    Stmt q(impl_->db, "INSERT INTO artifacts(id,kind,title,content,pinned,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?6)");
    q.text(1, id).text(2, kind).text(3, title).text(4, content).num(5, b.value("pinned", false) ? 1 : 0).num(6, t).run();
    return impl_->artifact_by_id(id);
}

json Store::artifact_patch(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    // validate everything before anything is written
    const bool has_title = b.contains("title"), has_content = b.contains("content"), has_snapshot = b.contains("snapshot");
    const std::string title = has_title ? need_string(b, "title", 500) : std::string();
    const std::string content = has_content ? need_string(b, "content", kMaxContent) : std::string();
    std::string label;
    if (has_snapshot) {
        if (!b["snapshot"].is_string()) fail(400, "'snapshot' must be a string");
        label = b["snapshot"].get<std::string>();
        if (label.size() > static_cast<size_t>(kVersionLabelMax)) fail(400, "'snapshot' is too long");
    }
    std::lock_guard<std::mutex> lk(mu_);
    const auto cur = impl_->artifact_state(id);   // 404 when absent
    const bool title_changes = has_title && title != cur.title, content_changes = has_content && content != cur.content;
    const int64_t now = now_ms();
    Tx tx(impl_->db);
    if (title_changes || content_changes) {
        // an explicit snapshot (the client is about to make a big change), or the first content edit after a long pause
        if (has_snapshot || (content_changes && now - cur.updated_at > version_gap_ms())) impl_->snapshot_state(id, cur, label);
    }
    if (has_title) { Stmt q(impl_->db, "UPDATE artifacts SET title=?2 WHERE id=?1"); q.text(1, id).text(2, title).run(); }
    if (has_content) { Stmt q(impl_->db, "UPDATE artifacts SET content=?2 WHERE id=?1"); q.text(1, id).text(2, content).run(); }
    if (b.contains("pinned")) { Stmt q(impl_->db, "UPDATE artifacts SET pinned=?2 WHERE id=?1"); q.text(1, id).num(2, b["pinned"].is_boolean() && b["pinned"].get<bool>() ? 1 : 0).run(); }
    { Stmt q(impl_->db, "UPDATE artifacts SET updated_at=?2 WHERE id=?1"); q.text(1, id).num(2, now).run(); }
    tx.commit();
    return impl_->artifact_by_id(id);
}

bool Store::artifact_delete(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    Tx tx(impl_->db);
    impl_->delete_artifact_children(id);
    Stmt q(impl_->db, "DELETE FROM artifacts WHERE id=?1");
    q.text(1, id).run();
    const bool deleted = sqlite3_changes(impl_->db) > 0;
    tx.commit();
    return deleted;
}

json Store::artifact_versions_list(const std::string& id, int limit, int offset) {
    limit = std::clamp(limit, 1, 500);
    offset = std::max(offset, 0);
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    Stmt q(impl_->db, ("SELECT rev,title,label,created_at,length(content),substr(content,1," + std::to_string(kVersionSnippetChars) + ") FROM artifact_versions WHERE artifact_id=?1 ORDER BY rev DESC LIMIT ?2 OFFSET ?3").c_str());
    q.text(1, id).num(2, limit).num(3, offset);
    json arr = json::array();
    while (q.row()) arr.push_back({{"rev", q.i64(0)}, {"title", q.str(1)}, {"label", q.str(2)}, {"created_at", q.i64(3)}, {"chars", q.i64(4)}, {"snippet", q.str(5)}});
    Stmt c(impl_->db, "SELECT COUNT(*) FROM artifact_versions WHERE artifact_id=?1");
    c.text(1, id).row();
    return {{"object", "list"}, {"data", arr}, {"total", c.i64(0)}, {"limit", limit}, {"offset", offset}};
}

json Store::artifact_version_get(const std::string& id, int64_t rev) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "SELECT rev,title,content,label,created_at,length(content) FROM artifact_versions WHERE artifact_id=?1 AND rev=?2");
    q.text(1, id).num(2, rev);
    if (!q.row()) fail(404, "version not found", "not_found");
    return {{"rev", q.i64(0)}, {"title", q.str(1)}, {"content", q.str(2)}, {"label", q.str(3)}, {"created_at", q.i64(4)}, {"chars", q.i64(5)}};
}

json Store::artifact_restore(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    if (!b.contains("rev") || !b["rev"].is_number_integer()) fail(400, "'rev' (the version number to restore) is required and must be an integer");
    const int64_t rev = b["rev"].get<int64_t>();
    std::lock_guard<std::mutex> lk(mu_);
    const auto cur = impl_->artifact_state(id);   // 404 when absent
    std::string title, content;
    {
        Stmt q(impl_->db, "SELECT title,content FROM artifact_versions WHERE artifact_id=?1 AND rev=?2");
        q.text(1, id).num(2, rev);
        if (!q.row()) fail(404, "version not found", "not_found");
        title = q.str(0);
        content = q.str(1);
    }
    Tx tx(impl_->db);
    impl_->snapshot_state(id, cur, "Before restore");
    { Stmt q(impl_->db, "UPDATE artifacts SET title=?2, content=?3, updated_at=?4 WHERE id=?1"); q.text(1, id).text(2, title).text(3, content).num(4, now_ms()).run(); }
    tx.commit();
    return impl_->artifact_by_id(id);
}

// ---- what belongs to an artifact: chat, comments, images and video, duplicate ----
json Store::artifact_chat_list(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    json arr = impl_->chat_rows(id);
    const auto total = arr.size();
    return {{"object", "list"}, {"data", arr}, {"total", total}};
}

json Store::artifact_chat_put(const std::string& id, const std::string& mid, const json& b) {
    if (mid.empty() || mid.size() > 64 || mid.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
        fail(400, "the message id must be 1 to 64 characters of A-Z, a-z, 0-9, _ and -");
    if (!b.is_object() || !b.contains("data") || !b["data"].is_object()) fail(400, "send {\"data\": {...}} with a JSON object");
    const std::string data = dump(b["data"]);
    if (data.size() > kMaxChatData) fail(400, "'data' is too large (at most 200 KB)");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    const int64_t now = now_ms();
    Tx tx(impl_->db);
    bool exists = false;
    {
        Stmt q(impl_->db, "SELECT 1 FROM artifact_chat WHERE artifact_id=?1 AND id=?2");
        q.text(1, id).text(2, mid);
        exists = q.row();
    }
    if (exists) {
        Stmt q(impl_->db, "UPDATE artifact_chat SET data=?3, updated_at=?4 WHERE artifact_id=?1 AND id=?2");
        q.text(1, id).text(2, mid).text(3, data).num(4, now).run();
    } else {
        int64_t next = 1;
        { Stmt q(impl_->db, "SELECT COALESCE(MAX(seq),0)+1 FROM artifact_chat WHERE artifact_id=?1"); q.text(1, id); if (q.row()) next = q.i64(0); }
        { Stmt q(impl_->db, "INSERT INTO artifact_chat(artifact_id,id,seq,data,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?5)"); q.text(1, id).text(2, mid).num(3, next).text(4, data).num(5, now).run(); }
        Stmt q(impl_->db, "DELETE FROM artifact_chat WHERE artifact_id=?1 AND seq NOT IN (SELECT seq FROM artifact_chat WHERE artifact_id=?1 ORDER BY seq DESC LIMIT ?2)");
        q.text(1, id).num(2, chat_cap()).run();
    }
    tx.commit();
    Stmt q(impl_->db, "SELECT id,seq,data,created_at,updated_at FROM artifact_chat WHERE artifact_id=?1 AND id=?2");
    q.text(1, id).text(2, mid);
    q.row();
    return Impl::chat_row(q);
}

int64_t Store::artifact_chat_clear(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    Stmt q(impl_->db, "DELETE FROM artifact_chat WHERE artifact_id=?1");
    q.text(1, id).run();
    return sqlite3_changes(impl_->db);
}

json Store::artifact_comments_list(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    json arr = impl_->comment_rows(id);
    const auto total = arr.size();
    return {{"object", "list"}, {"data", arr}, {"total", total}};
}

json Store::artifact_comment_create(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::string cid = need_string(b, "id", 36);
    if (!cid.empty() && !is_uuid(cid)) fail(400, "'id' must be a UUID");
    if (cid.empty()) cid = new_uuid();
    const std::string author = b.contains("author") ? need_string(b, "author", 8, true) : std::string("user");
    if (author != "user" && author != "dara") fail(400, "'author' must be user or dara");
    const std::string body = need_text(b, "body", kCommentBodyMax, true);
    const std::string quote = need_text(b, "quote", kCommentQuoteMax, false);
    int64_t nth = 0;
    if (b.contains("nth") && !b["nth"].is_null()) {
        if (!b["nth"].is_number_integer() || b["nth"].get<int64_t>() < 0) fail(400, "'nth' must be a whole number, 0 or more");
        nth = b["nth"].get<int64_t>();
    }
    const std::string parent = need_string(b, "parent_id", 36);
    if (!parent.empty() && !is_uuid(parent)) fail(400, "'parent_id' must be a UUID");
    if (!parent.empty() && !quote.empty()) fail(400, "a reply has no 'quote': only the first comment of a thread points at text");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    {
        Stmt dup(impl_->db, "SELECT 1 FROM artifact_comments WHERE id=?1");
        dup.text(1, cid);
        if (dup.row()) fail(409, "a comment with this id exists", "conflict");
    }
    {
        Stmt c(impl_->db, "SELECT COUNT(*) FROM artifact_comments WHERE artifact_id=?1");
        c.text(1, id).row();
        if (c.i64(0) >= kCommentsPerArtifact) fail(409, "this document has the most comments it can keep (" + std::to_string(kCommentsPerArtifact) + ")", "conflict");
    }
    if (!parent.empty()) {
        Stmt q(impl_->db, "SELECT parent_id FROM artifact_comments WHERE id=?1 AND artifact_id=?2");
        q.text(1, parent).text(2, id);
        if (!q.row()) fail(404, "the comment to reply to was not found", "not_found");
        if (!q.is_null(0)) fail(400, "reply to the first comment of the thread: threads are one level deep");
        Stmt c(impl_->db, "SELECT COUNT(*) FROM artifact_comments WHERE parent_id=?1");
        c.text(1, parent).row();
        if (c.i64(0) >= kRepliesPerThread) fail(409, "this thread has the most replies it can keep (" + std::to_string(kRepliesPerThread) + ")", "conflict");
    }
    const int64_t now = now_ms();
    Stmt q(impl_->db, "INSERT INTO artifact_comments(id,artifact_id,parent_id,author,body,quote,nth,resolved,created_at,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,0,?8,?8)");
    q.text(1, cid).text(2, id);
    if (parent.empty()) q.null(3); else q.text(3, parent);
    q.text(4, author).text(5, body).text(6, quote).num(7, parent.empty() ? nth : 0).num(8, now).run();
    return impl_->comment_by_id(id, cid);
}

json Store::artifact_comment_patch(const std::string& id, const std::string& cid, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    const bool has_body = b.contains("body"), has_resolved = b.contains("resolved"), has_quote = b.contains("quote"), has_nth = b.contains("nth");
    if (!has_body && !has_resolved && !has_quote && !has_nth) fail(400, "send body, resolved, quote or nth");
    const std::string body = has_body ? need_text(b, "body", kCommentBodyMax, true) : std::string();
    const std::string quote = has_quote ? need_text(b, "quote", kCommentQuoteMax, false) : std::string();
    if (has_resolved && !b["resolved"].is_boolean()) fail(400, "'resolved' must be true or false");
    int64_t nth = 0;
    if (has_nth) {
        if (!b["nth"].is_number_integer() || b["nth"].get<int64_t>() < 0) fail(400, "'nth' must be a whole number, 0 or more");
        nth = b["nth"].get<int64_t>();
    }
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    const json cur = impl_->comment_by_id(id, cid);   // 404 when absent
    if (!cur["parent_id"].is_null() && (has_resolved || has_quote || has_nth)) fail(400, "only the first comment of a thread can be resolved or anchored");
    Tx tx(impl_->db);
    if (has_body) { Stmt q(impl_->db, "UPDATE artifact_comments SET body=?2 WHERE id=?1"); q.text(1, cid).text(2, body).run(); }
    if (has_resolved) { Stmt q(impl_->db, "UPDATE artifact_comments SET resolved=?2 WHERE id=?1"); q.text(1, cid).num(2, b["resolved"].get<bool>() ? 1 : 0).run(); }
    if (has_quote) { Stmt q(impl_->db, "UPDATE artifact_comments SET quote=?2 WHERE id=?1"); q.text(1, cid).text(2, quote).run(); }
    if (has_nth) { Stmt q(impl_->db, "UPDATE artifact_comments SET nth=?2 WHERE id=?1"); q.text(1, cid).num(2, nth).run(); }
    { Stmt q(impl_->db, "UPDATE artifact_comments SET updated_at=?2 WHERE id=?1"); q.text(1, cid).num(2, now_ms()).run(); }
    tx.commit();
    return impl_->comment_by_id(id, cid);
}

int64_t Store::artifact_comment_delete(const std::string& id, const std::string& cid) {
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    {
        Stmt q(impl_->db, "SELECT 1 FROM artifact_comments WHERE id=?1 AND artifact_id=?2");
        q.text(1, cid).text(2, id);
        if (!q.row()) return 0;
    }
    Tx tx(impl_->db);
    int64_t n = 0;
    { Stmt q(impl_->db, "DELETE FROM artifact_comments WHERE parent_id=?1 AND artifact_id=?2"); q.text(1, cid).text(2, id).run(); n += sqlite3_changes(impl_->db); }
    { Stmt q(impl_->db, "DELETE FROM artifact_comments WHERE id=?1 AND artifact_id=?2"); q.text(1, cid).text(2, id).run(); n += sqlite3_changes(impl_->db); }
    tx.commit();
    return n;
}

json Store::artifact_asset_add(const std::string& id, const std::string& mime, const std::string& name_in, const std::string& bytes) {
    if (!asset_type_ok(mime)) fail(415, "only PNG, JPEG, GIF and WebP images and MP4 and WebM video can be stored (send the type in Content-Type)", "unsupported_media_type");
    if (bytes.empty()) fail(400, "the request body is empty");
    if (!asset_magic_ok(mime, bytes)) fail(415, "the file is not a valid " + mime + " (its first bytes do not match the type)", "unsupported_media_type");
    if (bytes.size() > asset_max()) fail(413, "the file is too large (at most " + std::to_string(asset_max()) + " bytes)", "payload_too_large");
    std::string name = name_in.empty() ? std::string("file") : name_in;
    for (auto& c : name) if (static_cast<unsigned char>(c) < 0x20 || c == 0x7F) c = ' ';
    if (utf8_len(name) > kAssetNameMax) fail(400, "the file name is too long (at most 200 characters)");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    {
        Stmt q(impl_->db, "SELECT COALESCE(SUM(size),0) FROM artifact_assets WHERE artifact_id=?1");
        q.text(1, id).row();
        if (static_cast<uint64_t>(q.i64(0)) + bytes.size() > asset_artifact_max())
            fail(413, "this document has used all the space for images and video (at most " + std::to_string(asset_artifact_max()) + " bytes)", "payload_too_large");
    }
    const std::string aid = new_uuid();
    const int64_t now = now_ms();
    {
        Stmt q(impl_->db, "INSERT INTO artifact_assets(id,artifact_id,name,mime,size,data,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7)");
        q.text(1, aid).text(2, id).text(3, name).text(4, mime).num(5, static_cast<int64_t>(bytes.size())).blob(6, bytes).num(7, now).run();
    }
    return {{"id", aid}, {"name", name}, {"mime", mime}, {"size", static_cast<int64_t>(bytes.size())}, {"created_at", now}};
}

json Store::artifact_assets_list(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    impl_->artifact_state(id);   // 404 when absent
    json arr = impl_->asset_rows(id);
    const auto total = arr.size();
    return {{"object", "list"}, {"data", arr}, {"total", total}};
}

bool Store::artifact_asset_open(const std::string& id, const std::string& asset_id, std::string& mime, std::string& bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "SELECT mime,data FROM artifact_assets WHERE id=?1 AND artifact_id=?2");
    q.text(1, asset_id).text(2, id);
    if (!q.row()) return false;
    mime = q.str(0);
    bytes = q.bytes(1);
    return true;
}

bool Store::artifact_asset_delete(const std::string& id, const std::string& asset_id) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "DELETE FROM artifact_assets WHERE id=?1 AND artifact_id=?2");
    q.text(1, asset_id).text(2, id).run();
    return sqlite3_changes(impl_->db) > 0;
}

json Store::artifact_duplicate(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::string nid = need_string(b, "id", 36);
    if (!nid.empty() && !is_uuid(nid)) fail(400, "'id' must be a UUID");
    if (nid.empty()) nid = new_uuid();
    const bool has_title = b.contains("title") && !b["title"].is_null();
    const std::string given = has_title ? need_string(b, "title", 500) : std::string();
    std::lock_guard<std::mutex> lk(mu_);
    std::string kind, title, content;
    {
        Stmt q(impl_->db, "SELECT kind,title,content FROM artifacts WHERE id=?1");
        q.text(1, id);
        if (!q.row()) fail(404, "artifact not found", "not_found");
        kind = q.str(0); title = q.str(1); content = q.str(2);
    }
    {
        Stmt dup(impl_->db, "SELECT 1 FROM artifacts WHERE id=?1");
        dup.text(1, nid);
        if (dup.row()) fail(409, "an artifact with this id exists", "conflict");
    }
    if (has_title) title = given; else if (!title.empty()) title += " (copy)";
    // every asset gets a new id, and the copy's text points at the new ids
    std::vector<std::pair<std::string, std::string>> renamed;   // old asset id -> new asset id
    {
        Stmt q(impl_->db, "SELECT id FROM artifact_assets WHERE artifact_id=?1 ORDER BY created_at, rowid");
        q.text(1, id);
        while (q.row()) renamed.emplace_back(q.str(0), new_uuid());
    }
    for (const auto& [o, n] : renamed) {
        const std::string from = "dara-asset:" + o, to = "dara-asset:" + n;
        for (size_t pos = 0; (pos = content.find(from, pos)) != std::string::npos; pos += to.size()) content.replace(pos, from.size(), to);
    }
    const int64_t now = now_ms();
    Tx tx(impl_->db);
    { Stmt q(impl_->db, "INSERT INTO artifacts(id,kind,title,content,pinned,created_at,updated_at) VALUES(?1,?2,?3,?4,0,?5,?5)"); q.text(1, nid).text(2, kind).text(3, title).text(4, content).num(5, now).run(); }
    for (const auto& [o, n] : renamed) {
        Stmt q(impl_->db, "INSERT INTO artifact_assets(id,artifact_id,name,mime,size,data,created_at) SELECT ?1,?2,name,mime,size,data,created_at FROM artifact_assets WHERE id=?3 AND artifact_id=?4");
        q.text(1, n).text(2, nid).text(3, o).text(4, id).run();
    }
    tx.commit();
    return impl_->artifact_by_id(nid);
}

// ---- messages ----
json Store::message_append(const std::string& cid, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    const std::string role = need_string(b, "role", 16, true);
    if (role != "system" && role != "user" && role != "assistant") fail(400, "'role' must be system, user or assistant");
    const std::string content = need_string(b, "content", kMaxContent);
    std::string id = need_string(b, "id", 36);
    if (id.empty()) id = new_uuid(); else if (!is_uuid(id)) fail(400, "'id' must be a UUID");
    const std::string finish = need_string(b, "finish_reason", 32), error = need_string(b, "error", 4096);
    std::string stats;
    if (b.contains("stats") && !b["stats"].is_null()) { if (!b["stats"].is_object()) fail(400, "'stats' must be an object"); stats = dump(b["stats"]); if (stats.size() > kMaxSetting) fail(400, "'stats' is too large"); }
    std::lock_guard<std::mutex> lk(mu_);
    impl_->conversation_by_id(cid);   // 404 when absent
    {
        Stmt dup(impl_->db, "SELECT 1 FROM messages WHERE id=?1");
        dup.text(1, id);
        if (dup.row()) fail(409, "a message with this id exists", "conflict");
    }
    Tx tx(impl_->db);
    int64_t pos = 0;
    { Stmt p(impl_->db, "SELECT COALESCE(MAX(position)+1,0) FROM messages WHERE conversation_id=?1"); p.text(1, cid); p.row(); pos = p.i64(0); }
    Stmt q(impl_->db, "INSERT INTO messages(id,conversation_id,position,role,content,finish_reason,error,stats,model,created_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)");
    q.text(1, id).text(2, cid).num(3, pos).text(4, role).text(5, content);
    finish.empty() ? q.null(6) : q.text(6, finish);
    error.empty() ? q.null(7) : q.text(7, error);
    stats.empty() ? q.null(8) : q.text(8, stats);
    q.opt_text(9, b.value("model", json(nullptr)));
    q.num(10, b.contains("created_at") && b["created_at"].is_number_integer() ? b["created_at"].get<int64_t>() : now_ms()).run();
    impl_->touch(cid);
    tx.commit();
    return impl_->message_by_id(id);
}

json Store::message_patch(const std::string& id, const json& b) {
    if (!b.is_object()) fail(400, "send a JSON object");
    std::lock_guard<std::mutex> lk(mu_);
    const json cur = impl_->message_by_id(id);   // 404 when absent
    Tx tx(impl_->db);
    if (b.contains("content")) { Stmt q(impl_->db, "UPDATE messages SET content=?2 WHERE id=?1"); q.text(1, id).text(2, need_string(b, "content", kMaxContent)).run(); }
    if (b.contains("finish_reason")) { Stmt q(impl_->db, "UPDATE messages SET finish_reason=?2 WHERE id=?1"); q.text(1, id).opt_text(2, b["finish_reason"]).run(); }
    if (b.contains("error")) { Stmt q(impl_->db, "UPDATE messages SET error=?2 WHERE id=?1"); q.text(1, id).opt_text(2, b["error"]).run(); }
    if (b.contains("stats")) {
        Stmt q(impl_->db, "UPDATE messages SET stats=?2 WHERE id=?1");
        q.text(1, id);
        if (b["stats"].is_null()) q.null(2);
        else { if (!b["stats"].is_object()) fail(400, "'stats' must be an object"); const std::string s = dump(b["stats"]); if (s.size() > kMaxSetting) fail(400, "'stats' is too large"); q.text(2, s); }
        q.run();
    }
    impl_->touch(cur["conversation_id"].get<std::string>());
    tx.commit();
    return impl_->message_by_id(id);
}

int Store::messages_truncate(const std::string& cid, int from) {
    if (from < 0) fail(400, "'from' must be 0 or more");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->conversation_by_id(cid);
    std::vector<std::string> files;
    {
        Stmt q(impl_->db, "SELECT a.file FROM attachments a JOIN messages m ON m.id=a.message_id WHERE m.conversation_id=?1 AND m.position>=?2");
        q.text(1, cid).num(2, from);
        while (q.row()) files.push_back(q.str(0));
    }
    Tx tx(impl_->db);
    Stmt q(impl_->db, "DELETE FROM messages WHERE conversation_id=?1 AND position>=?2");
    q.text(1, cid).num(2, from).run();
    const int n = sqlite3_changes(impl_->db);
    impl_->touch(cid);
    tx.commit();
    impl_->remove_files(files);
    return n;
}

// ---- attachments ----
json Store::attachment_add(const std::string& mid, const std::string& /*mime_hint*/, const std::string& bytes) {
    if (bytes.empty()) fail(400, "no image data");
    if (bytes.size() > kMaxAttachment) fail(413, "the image is larger than 20 MB", "invalid_request_error");
    const std::string mime = sniff_mime(bytes);
    if (mime.empty()) fail(400, "only PNG, JPEG and WebP images are accepted");
    std::lock_guard<std::mutex> lk(mu_);
    impl_->message_by_id(mid);   // 404 when absent
    const std::string id = new_uuid(), file = id + "." + ext_of(mime);
    {
        std::ofstream f(impl_->attachments / fs::u8path(file), std::ios::binary);
        f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!f) { std::error_code ec; fs::remove(impl_->attachments / fs::u8path(file), ec); throw std::runtime_error("cannot write the attachment file (disk full?)"); }
    }
    try {
        Stmt q(impl_->db, "INSERT INTO attachments(id,message_id,mime,file,bytes) VALUES(?1,?2,?3,?4,?5)");
        q.text(1, id).text(2, mid).text(3, mime).text(4, file).num(5, static_cast<int64_t>(bytes.size())).run();
    } catch (...) { impl_->remove_files({file}); throw; }
    return {{"id", id}, {"message_id", mid}, {"mime", mime}, {"bytes", bytes.size()}, {"url", "/dara/attachments/" + id}};
}

bool Store::attachment_open(const std::string& id, std::string& mime, std::string& bytes) {
    std::string file;
    {
        std::lock_guard<std::mutex> lk(mu_);
        Stmt q(impl_->db, "SELECT mime,file FROM attachments WHERE id=?1");
        q.text(1, id);
        if (!q.row()) return false;
        mime = q.str(0);
        file = q.str(1);
    }
    if (file.find_first_of("/\\") != std::string::npos) return false;
    std::ifstream f(impl_->attachments / fs::u8path(file), std::ios::binary);
    if (!f) return false;
    bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool Store::attachment_delete(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    std::string file;
    { Stmt q(impl_->db, "SELECT file FROM attachments WHERE id=?1"); q.text(1, id); if (!q.row()) return false; file = q.str(0); }
    Stmt q(impl_->db, "DELETE FROM attachments WHERE id=?1");
    q.text(1, id).run();
    impl_->remove_files({file});
    return true;
}

// ---- search ----
json Store::search(const std::string& query, int limit) {
    limit = std::clamp(limit, 1, 100);
    // every word becomes a quoted FTS5 term (so user text can never be FTS syntax); the last one is a prefix match
    std::vector<std::string> words;
    std::string cur;
    auto flush = [&] { if (!cur.empty()) { words.push_back(cur); cur.clear(); } };
    for (char c : query) { if (std::isspace(static_cast<unsigned char>(c))) flush(); else if (c != '"') cur += c; }
    flush();
    if (words.empty()) return {{"object", "list"}, {"data", json::array()}};
    if (words.size() > 12) words.resize(12);
    std::string match;
    for (size_t i = 0; i < words.size(); ++i) match += (i ? " " : "") + std::string("\"") + words[i] + "\"" + (i + 1 == words.size() ? "*" : "");
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "SELECT m.id,m.conversation_id,m.position,m.role,snippet(messages_fts,0,'[',']','...',16),c.title,c.updated_at "
                      "FROM messages_fts JOIN messages m ON m.rowid=messages_fts.rowid JOIN conversations c ON c.id=m.conversation_id "
                      "WHERE messages_fts MATCH ?1 ORDER BY rank LIMIT ?2");
    q.text(1, match).num(2, limit);
    json arr = json::array();
    while (q.row()) arr.push_back({{"message_id", q.str(0)}, {"conversation_id", q.str(1)}, {"position", q.i64(2)}, {"role", q.str(3)}, {"snippet", q.str(4)}, {"title", q.str(5)}, {"updated_at", q.i64(6)}});
    return {{"object", "list"}, {"data", arr}};
}

// ---- model configs ----
json Store::model_configs_all() {
    std::lock_guard<std::mutex> lk(mu_);
    json o = json::object();
    Stmt q(impl_->db, "SELECT model,kv,flash,vision,context,last_loaded_at,updated_at FROM model_configs ORDER BY model");
    while (q.row()) o[q.str(0)] = {{"kv", q.str(1)}, {"flash", q.str(2)}, {"vision", q.i64(3) != 0}, {"context", q.is_null(4) ? json(nullptr) : json(q.i64(4))},
                                    {"last_loaded_at", q.is_null(5) ? json(nullptr) : json(q.i64(5))}, {"updated_at", q.i64(6)}};
    return o;
}

json Store::model_config_put(const std::string& model, const json& b) {
    if (!valid_key(model)) fail(400, "invalid model id");
    if (!b.is_object()) fail(400, "send a JSON object");
    const std::string kv = b.value("kv", std::string("auto")), flash_s = b.contains("flash") ? (b["flash"].is_boolean() ? (b["flash"].get<bool>() ? "on" : "off") : b["flash"].is_string() ? b["flash"].get<std::string>() : "?") : "auto";
    static const char* kvs[] = {"auto", "q8", "k8v4", "q4", "tq4", "k16v8"};
    if (std::find_if(std::begin(kvs), std::end(kvs), [&](const char* k) { return kv == k; }) == std::end(kvs)) fail(400, "unknown K/V format '" + kv + "' (auto, q8, k8v4, q4, tq4, k16v8)");
    if (flash_s != "auto" && flash_s != "on" && flash_s != "off") fail(400, "'flash' must be auto, on, off, true or false");
    const bool vision = b.value("vision", true);
    int64_t ctx = -1;
    if (b.contains("context") && !b["context"].is_null()) { if (!b["context"].is_number_integer() || b["context"].get<int64_t>() < 256 || b["context"].get<int64_t>() > (int64_t(1) << 24)) fail(400, "'context' must be an integer between 256 and 16777216, or null"); ctx = b["context"].get<int64_t>(); }
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "INSERT INTO model_configs(model,kv,flash,vision,context,updated_at) VALUES(?1,?2,?3,?4,?5,?6) "
                      "ON CONFLICT(model) DO UPDATE SET kv=excluded.kv, flash=excluded.flash, vision=excluded.vision, context=excluded.context, updated_at=excluded.updated_at");
    q.text(1, model).text(2, kv).text(3, flash_s).num(4, vision ? 1 : 0);
    ctx < 0 ? q.null(5) : q.num(5, ctx);
    q.num(6, now_ms()).run();
    return {{"model", model}, {"kv", kv}, {"flash", flash_s}, {"vision", vision}, {"context", ctx < 0 ? json(nullptr) : json(ctx)}};
}

bool Store::model_config_delete(const std::string& model) {
    std::lock_guard<std::mutex> lk(mu_);
    Stmt q(impl_->db, "DELETE FROM model_configs WHERE model=?1");
    q.text(1, model).run();
    return sqlite3_changes(impl_->db) > 0;
}

// ---- export / delete ----
json Store::export_all() {
    json out = {{"format", "dara-export"}, {"version", 1}, {"exported_at", now_ms()}, {"settings", settings_all()}, {"model_configs", model_configs_all()}};
    json convs = json::array();
    std::vector<std::string> ids;
    {
        std::lock_guard<std::mutex> lk(mu_);
        Stmt q(impl_->db, "SELECT id FROM conversations ORDER BY created_at");
        while (q.row()) ids.push_back(q.str(0));
    }
    for (const auto& id : ids) { try { convs.push_back(conversation_get(id)); } catch (const StoreError&) {} }
    out["conversations"] = convs;
    json arts = json::array();
    {
        std::lock_guard<std::mutex> lk(mu_);
        Stmt q(impl_->db, "SELECT id,kind,title,pinned,created_at,updated_at,length(content),content FROM artifacts ORDER BY created_at");
        while (q.row()) arts.push_back(Impl::artifact_row(q, true));
        // chat and comments are text and go with their artifact; images and video only as metadata (the bytes: GET .../assets/{id}); versions are not exported
        for (auto& a : arts) {
            const std::string aid = a["id"].get<std::string>();
            a["chat"] = impl_->chat_rows(aid);
            a["comments"] = impl_->comment_rows(aid);
            a["assets"] = impl_->asset_rows(aid);
        }
    }
    out["artifacts"] = arts;
    return out;
}

void Store::delete_all() {
    std::lock_guard<std::mutex> lk(mu_);
    {
        Tx tx(impl_->db);
        exec(impl_->db, "DELETE FROM messages");   // attachments cascade, the search index follows by trigger
        exec(impl_->db, "DELETE FROM conversations");
        exec(impl_->db, "DELETE FROM model_configs");
        exec(impl_->db, "DELETE FROM artifact_versions");
        exec(impl_->db, "DELETE FROM artifact_chat");
        exec(impl_->db, "DELETE FROM artifact_comments");
        exec(impl_->db, "DELETE FROM artifact_assets");
        exec(impl_->db, "DELETE FROM artifacts");
        tx.commit();
    }
    exec(impl_->db, "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')");
    exec(impl_->db, "VACUUM");
    std::error_code ec;
    for (auto& e : fs::directory_iterator(impl_->attachments, ec)) { std::error_code e2; if (e.is_regular_file(e2)) fs::remove(e.path(), e2); }
}

// ---- HTTP ----
void register_store_routes(httplib::Server& svr, Store* store, const std::string& unavailable) {
    auto reply = [](httplib::Response& res, int status, const json& j) { res.status = status; res.set_content(dump(j), "application/json"); };
    auto error = [&](httplib::Response& res, int status, const std::string& msg, const char* type) { reply(res, status, {{"error", {{"message", msg}, {"type", type}}}}); };
    // wraps a handler: store missing -> 503; StoreError -> its status; anything else -> 500
    auto guard = [=](auto fn) {
        return [=](const httplib::Request& req, httplib::Response& res) {
            if (!store) { json e = {{"error", {{"message", "the app database is not available: " + unavailable}, {"type", "server_error"}, {"code", "store_unavailable"}}}}; res.status = 503; res.set_content(dump(e), "application/json"); return; }
            try { fn(req, res); }
            catch (const StoreError& e) { json j = {{"error", {{"message", e.what()}, {"type", e.type}}}}; res.status = e.status; res.set_content(dump(j), "application/json"); }
            catch (const std::exception& e) { json j = {{"error", {{"message", e.what()}, {"type", "server_error"}}}}; res.status = 500; res.set_content(dump(j), "application/json"); }
        };
    };
    auto body_of = [](const httplib::Request& req) -> json {
        json b = json::parse(req.body.empty() ? "{}" : req.body, nullptr, false);
        if (b.is_discarded()) fail(400, "the request body is not valid JSON");
        return b;
    };
    auto intp = [](const httplib::Request& req, const char* name, int def) {
        if (!req.has_param(name)) return def;
        try { return std::stoi(req.get_param_value(name)); } catch (...) { fail(400, std::string("'") + name + "' must be a number"); }
    };

    svr.Get("/dara/store", guard([=](const httplib::Request&, httplib::Response& res) { reply(res, 200, store->info()); }));

    svr.Get("/dara/settings", guard([=](const httplib::Request&, httplib::Response& res) { reply(res, 200, store->settings_all()); }));
    svr.Patch("/dara/settings", guard([=](const httplib::Request& req, httplib::Response& res) { store->settings_set(body_of(req)); reply(res, 200, store->settings_all()); }));
    svr.Delete(R"(/dara/settings/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->settings_delete(req.matches[1])) { error(res, 404, "setting not found", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));

    svr.Get("/dara/conversations", guard([=](const httplib::Request& req, httplib::Response& res) {
        reply(res, 200, store->conversations_list(intp(req, "limit", 50), intp(req, "offset", 0)));
    }));
    svr.Post("/dara/conversations", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 201, store->conversation_create(body_of(req))); }));
    svr.Get(R"(/dara/conversations/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->conversation_get(req.matches[1])); }));
    svr.Patch(R"(/dara/conversations/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->conversation_patch(req.matches[1], body_of(req))); }));
    svr.Delete(R"(/dara/conversations/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->conversation_delete(req.matches[1])) { error(res, 404, "conversation not found", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));
    svr.Post(R"(/dara/conversations/([^/]+)/messages)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 201, store->message_append(req.matches[1], body_of(req))); }));
    svr.Delete(R"(/dara/conversations/([^/]+)/messages)", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!req.has_param("from")) fail(400, "'from' (the first position to delete) is required");
        reply(res, 200, {{"deleted", store->messages_truncate(req.matches[1], intp(req, "from", 0))}});
    }));
    svr.Patch(R"(/dara/messages/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->message_patch(req.matches[1], body_of(req))); }));

    // image bytes: JSON {"data": "<base64>"} (a data: URI prefix is accepted); the type is detected from the bytes
    svr.Post(R"(/dara/messages/([^/]+)/attachments)", guard([=](const httplib::Request& req, httplib::Response& res) {
        const json b = body_of(req);
        std::string data = b.value("data", std::string());
        if (const auto c = data.find(','); data.rfind("data:", 0) == 0 && c != std::string::npos) data = data.substr(c + 1);
        static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string bytes;
        int val = 0, bits = -8;
        for (unsigned char ch : data) {
            if (ch == '=' || std::isspace(ch)) continue;
            const auto p = chars.find(static_cast<char>(ch));
            if (p == std::string::npos) fail(400, "'data' is not valid base64");
            val = (val << 6) + static_cast<int>(p);
            bits += 6;
            if (bits >= 0) { bytes.push_back(static_cast<char>((val >> bits) & 0xFF)); bits -= 8; }
        }
        reply(res, 201, store->attachment_add(req.matches[1], b.value("mime", std::string()), bytes));
    }));
    // the app fetches this with its Authorization header and shows the bytes through a blob: URL (an <img src> cannot send the header)
    svr.Get(R"(/dara/attachments/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        std::string mime, bytes;
        if (!store->attachment_open(req.matches[1], mime, bytes)) { error(res, 404, "attachment not found", "not_found"); return; }
        res.set_header("Cache-Control", "private, max-age=31536000, immutable");
        res.set_content(bytes, mime);
    }));
    svr.Delete(R"(/dara/attachments/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->attachment_delete(req.matches[1])) { error(res, 404, "attachment not found", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));

    svr.Get("/dara/artifacts", guard([=](const httplib::Request& req, httplib::Response& res) {
        reply(res, 200, store->artifacts_list(req.has_param("kind") ? req.get_param_value("kind") : std::string(), intp(req, "limit", 100), intp(req, "offset", 0)));
    }));
    svr.Post("/dara/artifacts", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 201, store->artifact_create(body_of(req))); }));
    svr.Get(R"(/dara/artifacts/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_get(req.matches[1])); }));
    svr.Patch(R"(/dara/artifacts/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_patch(req.matches[1], body_of(req))); }));
    svr.Get(R"(/dara/artifacts/([^/]+)/versions)", guard([=](const httplib::Request& req, httplib::Response& res) {
        reply(res, 200, store->artifact_versions_list(req.matches[1], intp(req, "limit", 100), intp(req, "offset", 0)));
    }));
    svr.Get(R"(/dara/artifacts/([^/]+)/versions/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        const std::string text = req.matches[2];
        int64_t rev = 0;
        try { size_t used = 0; rev = std::stoll(text, &used); if (used != text.size()) throw std::invalid_argument("rev"); } catch (...) { fail(404, "version not found", "not_found"); }
        reply(res, 200, store->artifact_version_get(req.matches[1], rev));
    }));
    svr.Post(R"(/dara/artifacts/([^/]+)/restore)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_restore(req.matches[1], body_of(req))); }));
    svr.Post(R"(/dara/artifacts/([^/]+)/duplicate)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 201, store->artifact_duplicate(req.matches[1], body_of(req))); }));

    // the chat that belongs to an artifact
    svr.Get(R"(/dara/artifacts/([^/]+)/chat)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_chat_list(req.matches[1])); }));
    svr.Put(R"(/dara/artifacts/([^/]+)/chat/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_chat_put(req.matches[1], req.matches[2], body_of(req))); }));
    svr.Delete(R"(/dara/artifacts/([^/]+)/chat)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, {{"deleted", store->artifact_chat_clear(req.matches[1])}}); }));

    // comments
    svr.Get(R"(/dara/artifacts/([^/]+)/comments)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_comments_list(req.matches[1])); }));
    svr.Post(R"(/dara/artifacts/([^/]+)/comments)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 201, store->artifact_comment_create(req.matches[1], body_of(req))); }));
    svr.Patch(R"(/dara/artifacts/([^/]+)/comments/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_comment_patch(req.matches[1], req.matches[2], body_of(req))); }));
    svr.Delete(R"(/dara/artifacts/([^/]+)/comments/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        const int64_t n = store->artifact_comment_delete(req.matches[1], req.matches[2]);
        if (n == 0) { error(res, 404, "comment not found", "not_found"); return; }
        reply(res, 200, {{"deleted", n}});
    }));

    // images and video: the body is the raw file, its type is the Content-Type, its name the (url-encoded) X-Dara-Name header
    svr.Post(R"(/dara/artifacts/([^/]+)/assets)", guard([=](const httplib::Request& req, httplib::Response& res) {
        std::string type = req.get_header_value("Content-Type");
        if (const auto semi = type.find(';'); semi != std::string::npos) type.resize(semi);
        while (!type.empty() && type.back() == ' ') type.pop_back();
        std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        reply(res, 201, store->artifact_asset_add(req.matches[1], type, url_decode(req.get_header_value("X-Dara-Name")), req.body));
    }));
    svr.Get(R"(/dara/artifacts/([^/]+)/assets)", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->artifact_assets_list(req.matches[1])); }));
    // fetched with the Authorization header and shown through a blob: URL (an <img src> cannot send the header)
    svr.Get(R"(/dara/artifacts/([^/]+)/assets/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        std::string mime, bytes;
        if (!store->artifact_asset_open(req.matches[1], req.matches[2], mime, bytes)) { error(res, 404, "asset not found", "not_found"); return; }
        res.set_header("Cache-Control", "private, max-age=31536000, immutable");
        res.set_header("X-Content-Type-Options", "nosniff");
        res.set_content(bytes, mime);
    }));
    svr.Delete(R"(/dara/artifacts/([^/]+)/assets/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->artifact_asset_delete(req.matches[1], req.matches[2])) { error(res, 404, "asset not found", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));
    svr.Delete(R"(/dara/artifacts/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->artifact_delete(req.matches[1])) { error(res, 404, "artifact not found", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));
    svr.Get("/dara/search", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->search(req.get_param_value("q"), intp(req, "limit", 20))); }));

    svr.Get("/dara/model-configs", guard([=](const httplib::Request&, httplib::Response& res) { reply(res, 200, store->model_configs_all()); }));
    svr.Put(R"(/dara/model-configs/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) { reply(res, 200, store->model_config_put(req.matches[1], body_of(req))); }));
    svr.Delete(R"(/dara/model-configs/([^/]+))", guard([=](const httplib::Request& req, httplib::Response& res) {
        if (!store->model_config_delete(req.matches[1])) { error(res, 404, "no saved configuration for this model", "not_found"); return; }
        reply(res, 200, {{"deleted", true}});
    }));

    svr.Get("/dara/data/export", guard([=](const httplib::Request&, httplib::Response& res) {
        res.set_header("Content-Disposition", "attachment; filename=\"dara-export.json\"");
        reply(res, 200, store->export_all());
    }));
    svr.Delete("/dara/data", guard([=](const httplib::Request& req, httplib::Response& res) {
        const json b = body_of(req);
        if (b.value("confirm", std::string()) != "delete-all-data") fail(400, "send {\"confirm\": \"delete-all-data\"} to delete all conversations, attachments and saved model configurations");
        store->delete_all();
        reply(res, 200, {{"deleted", true}});
    }));
}

}  // namespace dara::serve
