#pragma once
// Backend-independent CUDA-graph decode tests, run against Nano and Qwen alike. A backend supplies an adapter A with:
//   using Model; using Snap;
//   std::unique_ptr<Model> make();                       // constructs a fresh model (may throw)
//   int vocab(Model&);
//   Snap snap(Model&);  void rewind(Model&, const Snap&); // restore point (Nano: KV length; Qwen: DeltaNet state + KV length)
//   std::vector<float> fwd(Model&, const std::vector<int32_t>& tokens, int logits_from);
// The Model must provide set_graphs, graph_captures/replays/failures, graphs_enabled, invalidate_graph, position.
//
// Scenarios (each compared graphs-on vs graphs-off, bitwise):
//   stress          scratch buffers move (a longer prefill) -> re-capture; restore/rewind then replay; 560 replays crossing position 512
//   stage_faults    the driver "refuses" graph begin-capture / end-capture / instantiate / launch in turn -> eager fallback, then recovery
//   repeated_faults three consecutive failures disable graphs; decoding continues eagerly and exactly
//   growth_failure  a device allocation fails while a longer prefill grows the scratch buffers -> clean error, session unchanged, decode exact
//   ctor_failure    the Nth device allocation fails during construction -> exception, zero leaked device bytes, a fresh model works afterwards
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "model/decode_graph.h"
#include "model/device_buffer.h"

namespace graphtest {

using dara::model::DecodeGraph;

inline int argmax(const std::vector<float>& v) {
    int b = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[b]) b = static_cast<int>(i);
    return b;
}
inline bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Result {
    std::vector<std::vector<float>> rows;
    long captures = 0, replays = 0, failures = 0;
    bool graphs_enabled = true;
    bool restore_ok = true;   // restoring and re-decoding reproduced the first pass bitwise
    bool threw = false;       // an injected failure surfaced as an exception where one was expected
    bool position_kept = true;
};

template <class A>
struct Tests {
    A& a;
    explicit Tests(A& adapter) : a(adapter) {}

    using Model = typename A::Model;

    static std::vector<int32_t> prompt6() { return {2, 818, 5279, 529, 7001, 563}; }
    static std::vector<int32_t> prompt200() {
        std::vector<int32_t> p(200);
        for (int i = 0; i < 200; ++i) p[i] = 1000 + (i * 37) % 5000;
        return p;
    }

    void decode(Model& m, int32_t& tok, int n, Result& r) {
        for (int i = 0; i < n; ++i) {
            std::vector<float> row = a.fwd(m, {tok}, 0);
            tok = argmax(row);
            r.rows.push_back(std::move(row));
        }
    }
    void finish(Model& m, Result& r) {
        r.captures = m.graph_captures();
        r.replays = m.graph_replays();
        r.failures = m.graph_failures();
        r.graphs_enabled = m.graphs_enabled();
    }
    int32_t prefill(Model& m, const std::vector<int32_t>& p, Result& r) {
        r.rows.push_back(a.fwd(m, p, static_cast<int>(p.size()) - 1));
        return argmax(r.rows.back());
    }

    Result stress(bool graphs) {
        auto m = a.make();
        m->set_graphs(graphs);
        Result r;
        int32_t tok = prefill(*m, prompt6(), r);
        decode(*m, tok, 5, r);                                   // graph captured here (small scratch buffers)
        {   // restore point, decode, restore, decode again: the replayed graph must see the restored recurrent state / KV length
            const auto s = a.snap(*m);
            const int32_t t0 = tok;
            Result first;
            decode(*m, tok, 4, first);
            a.rewind(*m, s);
            int32_t t1 = t0;
            Result second;
            decode(*m, t1, 4, second);
            for (size_t i = 0; i < 4; ++i) r.restore_ok = r.restore_ok && same(first.rows[i], second.rows[i]);
            r.rows.insert(r.rows.end(), first.rows.begin(), first.rows.end());
            tok = t1;
        }
        tok = prefill(*m, prompt200(), r);                       // longer prefill: scratch buffers grow and move -> re-capture
        decode(*m, tok, 5, r);
        {
            const auto s = a.snap(*m);
            const int32_t t0 = tok;
            Result first;
            decode(*m, tok, 4, first);
            a.rewind(*m, s);
            int32_t t1 = t0;
            Result second;
            decode(*m, t1, 4, second);
            for (size_t i = 0; i < 4; ++i) r.restore_ok = r.restore_ok && same(first.rows[i], second.rows[i]);
            r.rows.insert(r.rows.end(), first.rows.begin(), first.rows.end());
            tok = t1;
        }
        decode(*m, tok, 560, r);                                 // hundreds of replays across position 512
        finish(*m, r);
        return r;
    }

    Result stage_faults(bool graphs) {
        auto m = a.make();
        m->set_graphs(graphs);
        Result r;
        int32_t tok = prefill(*m, prompt6(), r);
        decode(*m, tok, 3, r);
        using S = DecodeGraph::Stage;
        for (S stage : {S::BeginCapture, S::EndCapture, S::Instantiate, S::Launch}) {
            if (graphs) {
                if (stage != S::Launch) m->invalidate_graph();   // force a (re-)capture so the capture-time stages are reached
                DecodeGraph::inject_fault(stage, 1);
            }
            decode(*m, tok, 3, r);
        }
        DecodeGraph::inject_fault(S::None, 0);
        finish(*m, r);
        return r;
    }

    Result repeated_faults(bool graphs) {
        auto m = a.make();
        m->set_graphs(graphs);
        Result r;
        int32_t tok = prefill(*m, prompt6(), r);
        decode(*m, tok, 2, r);
        if (graphs) {
            m->invalidate_graph();
            DecodeGraph::inject_fault(DecodeGraph::Stage::BeginCapture, 100);
        }
        decode(*m, tok, 8, r);                                   // 3 consecutive failures -> graphs off, the rest runs eagerly
        DecodeGraph::inject_fault(DecodeGraph::Stage::None, 0);
        finish(*m, r);
        return r;
    }

    Result growth_failure(bool graphs, bool attempt) {
        auto m = a.make();
        m->set_graphs(graphs);
        Result r;
        int32_t tok = prefill(*m, prompt6(), r);
        decode(*m, tok, 3, r);
        if (attempt) {
            const size_t pos = m->position();
            dara::model::alloc_fault_countdown().store(1);       // the next device allocation fails: the long prefill must grow a scratch buffer
            try {
                (void)a.fwd(*m, prompt200(), 199);
            } catch (const std::exception&) {
                r.threw = true;
            }
            dara::model::alloc_fault_countdown().store(0);
            r.position_kept = (m->position() == pos);
        }
        decode(*m, tok, 6, r);                                   // the session must continue exactly as if nothing had happened
        finish(*m, r);
        return r;
    }

    // Returns true when construction failed cleanly at every injected point.
    bool ctor_failure() {
        bool ok = true;
        const size_t baseline = dara::model::device_stats().current.load();
        for (long n : {1L, 3L, 9L, 25L, 60L}) {
            dara::model::alloc_fault_countdown().store(n);
            bool threw = false;
            try {
                auto m = a.make();
            } catch (const std::exception&) {
                threw = true;
            }
            dara::model::alloc_fault_countdown().store(0);
            const size_t now = dara::model::device_stats().current.load();
            if (!threw) std::printf("    allocation #%ld: construction unexpectedly succeeded (fewer allocations than that)\n", n);
            if (now != baseline) { std::printf("    allocation #%ld: %zu device bytes leaked\n", n, now - baseline); ok = false; }
        }
        auto m = a.make();  // after all the failures a normal construction must work
        Result r;
        int32_t tok = prefill(*m, prompt6(), r);
        decode(*m, tok, 2, r);
        return ok && r.rows.size() == 3;
    }

    static bool rows_equal(const Result& x, const Result& y, const char* what) {
        if (x.rows.size() != y.rows.size()) { std::printf("    %s: row counts differ (%zu vs %zu)\n", what, x.rows.size(), y.rows.size()); return false; }
        size_t bad = 0;
        for (size_t i = 0; i < x.rows.size(); ++i) if (!same(x.rows[i], y.rows[i])) ++bad;
        if (bad) std::printf("    %s: %zu of %zu logits rows differ\n", what, bad, x.rows.size());
        return bad == 0;
    }

    int run_all(const char* name) {
        int failures = 0;
        auto check = [&](bool ok, const char* what) { std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what); if (!ok) ++failures; };
        std::printf("%s graph tests\n", name);

        const Result se = stress(false), sg = stress(true);
        check(rows_equal(se, sg, "stress"), "stress: graphs-on logits bitwise identical to graphs-off (re-capture, restore, 560 replays)");
        check(sg.captures >= 2 && sg.replays > 500 && se.captures == 0, "stress: graph re-captured after the scratch buffers moved");
        check(se.restore_ok && sg.restore_ok, "stress: restore then replay reproduces the first pass bitwise (recurrent state / KV length restored)");

        const Result fe = stage_faults(false), fg = stage_faults(true);
        check(rows_equal(fe, fg, "stage_faults"), "faults: begin/end-capture, instantiate and launch failures each fall back to eager with identical logits");
        check(fg.failures == 4 && fg.captures >= 5 && fg.graphs_enabled, "faults: every failure counted, graph re-captured after each, graphs still enabled");

        const Result re = repeated_faults(false), rg = repeated_faults(true);
        check(rows_equal(re, rg, "repeated_faults"), "repeated faults: logits identical while failing");
        check(!rg.graphs_enabled && rg.failures == 3, "repeated faults: three consecutive failures switch graphs off (no retry storm)");

        const Result ge = growth_failure(false, false), gg = growth_failure(true, true), gge = growth_failure(false, true);
        check(gg.threw && gge.threw, "allocation failure: the long prefill raised an error instead of corrupting state");
        check(gg.position_kept && gge.position_kept, "allocation failure: position unchanged after the failed forward");
        check(rows_equal(ge, gg, "growth_failure(graph)") && rows_equal(ge, gge, "growth_failure(eager)"),
              "allocation failure: decoding afterwards is bitwise identical to a session that never saw the failure");
        check(gg.graphs_enabled, "allocation failure: graphs still enabled and working afterwards");

        check(ctor_failure(), "construction failure at 5 different allocations: exception, zero leaked device bytes, fresh model works");
        std::printf("%s\n", failures ? "FAILED" : "ok");
        return failures;
    }
};

}  // namespace graphtest
