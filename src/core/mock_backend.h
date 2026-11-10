#pragma once
// Deterministic stand-ins for the Qwen and Gemma backends. They exercise the IModelBackend
// contract (session ownership, checkpoint/restore, strong guarantee) with the two different
// state shapes the real backends will have. No model math.
#include <algorithm>
#include <stdexcept>
#include <string>

#include "core/backend.h"

namespace dara::core {

template <class State, class Derived>
class MockBackendBase : public IModelBackend {
public:
    void load(const ModelManifest& m, const PlacementPlan& p) override {
        if (m.family != Derived::kFamily)
            throw std::runtime_error(std::string("backend ") + Derived::kFamily + " rejects family " + m.family);
        if (!p.loadable) throw std::runtime_error("placement is not loadable");
        loaded_ = true;
    }
    void unload() override { loaded_ = false; }

    void prefill(Session& s, const TokenBatch& b) override {
        State st = current(s);
        for (auto t : b.tokens) Derived::absorb(st, t);
        if (!s.append(b.tokens)) throw std::runtime_error("out of KV blocks");
        s.backend_state = std::make_unique<State>(std::move(st));
    }

    DecodeResult decode(Session& s) override {
        State st = current(s);
        DecodeResult r;
        r.token = Derived::next(st);
        r.eos = r.token % 97 == 0;
        Derived::absorb(st, r.token);
        if (!s.append({r.token})) throw std::runtime_error("out of KV blocks");
        s.backend_state = std::make_unique<State>(std::move(st));
        return r;
    }

    Checkpoint checkpoint(const Session& s) override {
        Checkpoint cp;
        cp.n_tokens = s.tokens().size();
        cp.prefix_hash = hash_tokens(s.tokens(), cp.n_tokens);
        cp.state = std::make_shared<State>(current(s));
        return cp;
    }

    void restore(Session& s, const Checkpoint& cp) override {
        // Validate everything before touching the session.
        auto* st = dynamic_cast<const State*>(cp.state.get());
        if (!st || st->owner() != Derived::kOwner) throw std::runtime_error("checkpoint belongs to another backend");
        if (cp.n_tokens > s.tokens().size()) throw std::runtime_error("checkpoint is ahead of session");
        if (hash_tokens(s.tokens(), cp.n_tokens) != cp.prefix_hash)
            throw std::runtime_error("checkpoint does not match session history");
        auto restored = std::make_unique<State>(*st);
        s.truncate(cp.n_tokens);
        s.backend_state = std::move(restored);
    }

private:
    State current(const Session& s) const {
        if (!loaded_) throw std::runtime_error("backend not loaded");
        if (!s.backend_state) {
            if (!s.tokens().empty()) throw std::runtime_error("session has tokens but no backend state");
            return State{};
        }
        auto* p = dynamic_cast<const State*>(s.backend_state.get());
        if (!p) throw std::runtime_error("session state belongs to another backend");
        return *p;
    }
    bool loaded_ = false;
};

// Qwen-like: fixed-size recurrent state folded over every token.
struct MockRecurrentState : BackendState {
    uint64_t h = 0;
    std::unique_ptr<BackendState> clone() const override { return std::make_unique<MockRecurrentState>(*this); }
    std::string owner() const override { return "mock-qwen35"; }
};

class MockQwenBackend : public MockBackendBase<MockRecurrentState, MockQwenBackend> {
public:
    static constexpr const char* kFamily = "qwen35";
    static constexpr const char* kOwner = "mock-qwen35";
    static void absorb(MockRecurrentState& s, int32_t t) { s.h = s.h * 1315423911ull + static_cast<uint32_t>(t) + 1; }
    static int32_t next(const MockRecurrentState& s) { return static_cast<int32_t>(s.h >> 33) % 50000 + 1; }
};

// Gemma-like: only a bounded window of recent tokens matters.
struct MockWindowState : BackendState {
    std::vector<int32_t> window;
    std::unique_ptr<BackendState> clone() const override { return std::make_unique<MockWindowState>(*this); }
    std::string owner() const override { return "mock-gemma4"; }
};

class MockGemmaBackend : public MockBackendBase<MockWindowState, MockGemmaBackend> {
public:
    static constexpr const char* kFamily = "gemma4";
    static constexpr const char* kOwner = "mock-gemma4";
    static constexpr size_t kWindow = 4;
    static void absorb(MockWindowState& s, int32_t t) {
        s.window.push_back(t);
        if (s.window.size() > kWindow) s.window.erase(s.window.begin());
    }
    static int32_t next(const MockWindowState& s) {
        int64_t a = 7;
        for (auto t : s.window) a = a * 31 + t;
        return static_cast<int32_t>(a % 50000 + 50000) % 50000 + 1;
    }
};

}  // namespace dara::core
