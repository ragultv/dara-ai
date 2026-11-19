// The app database: chat history, settings and per-model configuration, in one SQLite file inside the Dara data folder
// (%LOCALAPPDATA%\Dara\dara.db; --data-dir / DARA_DATA_DIR override). Owned by the server and served under /dara/ (see docs/ui/08-database.md);
// the desktop app and any other client use the HTTP endpoints, none opens the file. Images are files in <data dir>\attachments\.
// Schema: docs/ui/db/schema.sql (embedded as store_schema.h by tools/gen_store_schema.py).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

struct sqlite3;
namespace httplib { class Server; }

namespace dara::serve {

struct StoreError : std::runtime_error {
    int status;
    std::string type;
    StoreError(int s, const std::string& msg, const char* t = "invalid_request_error") : std::runtime_error(msg), status(s), type(t) {}
};

class Store {
public:
    // Opens (or creates) <data_dir>/dara.db and applies migrations. A database that fails its integrity check is set aside as
    // dara.db.corrupt-<time> and a fresh one is created. Throws std::runtime_error when the file cannot be opened/created
    // or comes from a newer Dara.
    explicit Store(const std::string& data_dir);
    ~Store();
    Store(const Store&) = delete;
    Store& operator=(const Store&) = delete;

    static std::string default_data_dir();   // DARA_DATA_DIR, else %LOCALAPPDATA%\Dara
    const std::string& dir() const { return dir_; }
    nlohmann::json info();                    // path, size, schema version, row counts, recovery note

    // settings: key/value JSON
    nlohmann::json settings_all();                                  // {"theme": "system", ...}
    void settings_set(const nlohmann::json& object);                // bulk upsert
    bool settings_delete(const std::string& key);

    // conversations and messages
    nlohmann::json conversations_list(int limit, int offset, bool pinned_first = true);
    nlohmann::json conversation_get(const std::string& id);         // with messages and their attachments
    nlohmann::json conversation_create(const nlohmann::json& body);
    nlohmann::json conversation_patch(const std::string& id, const nlohmann::json& body);
    bool conversation_delete(const std::string& id);
    nlohmann::json message_append(const std::string& conversation_id, const nlohmann::json& body);
    nlohmann::json message_patch(const std::string& id, const nlohmann::json& body);
    int messages_truncate(const std::string& conversation_id, int from_position);   // deletes messages at positions >= from; returns how many

    // attachments (images)
    nlohmann::json attachment_add(const std::string& message_id, const std::string& mime_hint, const std::string& bytes);
    bool attachment_open(const std::string& id, std::string& mime, std::string& bytes);
    bool attachment_delete(const std::string& id);

    nlohmann::json search(const std::string& query, int limit);

    // per-model configuration chosen by the user (absent = Auto)
    nlohmann::json model_configs_all();
    nlohmann::json model_config_put(const std::string& model, const nlohmann::json& body);
    bool model_config_delete(const std::string& model);

    nlohmann::json export_all();    // everything as JSON (attachments by id; fetch bytes separately)
    void delete_all();              // rows, VACUUM, attachment files

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string dir_;
    std::mutex mu_;
};

// Registers the /dara/ store endpoints. store == nullptr (it could not be opened) makes them answer 503 with `unavailable`.
void register_store_routes(httplib::Server& svr, Store* store, const std::string& unavailable);

}  // namespace dara::serve
