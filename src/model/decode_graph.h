#pragma once
// Reusable CUDA-graph execution for single-token decode, shared by every model backend.
//
// Decode launches hundreds of small kernels per token; on Windows the per-launch cost of the driver, not the GPU, then sets the pace
// (measured: host enqueue time ~= GPU time). A backend describes ONE decode step as "enqueue this work on stream()"; DecodeGraph captures
// it once, instantiates it, and replays it with a single launch. The backend's contract:
//   * everything that changes from token to token (the position) lives in DEVICE memory the kernels read, never in a launch argument
//   * no allocation, host synchronization or host-side state change happens inside the captured body
//   * the body is re-captured when the device buffers it was built for move (`key` = those pointers)
//   * graph failure is never fatal: every failure in capture / instantiate / launch discards the graph, releases every CUDA resource it
//     created, leaves the stream usable, and surfaces as a GraphError; the backend then runs the step on its ordinary eager path
// The work stream is a blocking stream, so it orders against the legacy default stream the rest of the runtime still uses.
#include <cuda_runtime.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/device_buffer.h"

namespace dara::model {

// Thrown by DecodeGraph for any failure of graph preparation or replay. Callers catch it and fall back to eager execution.
class GraphError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class DecodeGraph {
public:
    enum class Stage { None = 0, BeginCapture, EndCapture, Instantiate, Launch };

    DecodeGraph() = default;
    ~DecodeGraph() { reset(); }
    DecodeGraph(const DecodeGraph&) = delete;
    DecodeGraph& operator=(const DecodeGraph&) = delete;

    void create_stream() {
        if (!stream_) { cuda_check(cudaStreamCreate(&stream_), "create work stream"); owns_stream_ = true; }
    }
    // Captures on (and replays on) another DecodeGraph's stream, which it does not own (several graphs of different shapes, one work stream).
    void share_stream(cudaStream_t s) { stream_ = s; owns_stream_ = false; }
    cudaStream_t stream() const { return stream_; }
    void* stream_ptr() const { return stream_; }

    bool ready_for(const std::vector<void*>& key) const { return exec_ != nullptr && key == key_; }
    bool valid() const { return exec_ != nullptr; }

    // Captures body() (which must enqueue only on stream()) and instantiates it, replacing any previous graph. On ANY failure the partial
    // capture is aborted, the graph and exec are destroyed, valid() is false, and GraphError is thrown.
    template <class F>
    void capture(const std::vector<void*>& key, F&& body) {
        invalidate();
        bool capturing = false;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        try {
            fault(Stage::BeginCapture);
            check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "begin graph capture");
            capturing = true;
            body();
            fault(Stage::EndCapture);
            capturing = false;
            check(cudaStreamEndCapture(stream_, &graph), "end graph capture");
            if (!graph) throw GraphError("graph capture produced no graph");
            fault(Stage::Instantiate);
            check(cudaGraphInstantiate(&exec, graph, 0), "instantiate graph");
            cudaGraphDestroy(graph);
            graph = nullptr;
        } catch (const std::exception& e) {
            if (capturing) {  // abort the capture so the stream is usable again
                cudaGraph_t dead = nullptr;
                cudaStreamEndCapture(stream_, &dead);
                if (dead) cudaGraphDestroy(dead);
            }
            if (graph) cudaGraphDestroy(graph);
            if (exec) cudaGraphExecDestroy(exec);
            cudaGetLastError();  // clear any non-sticky error the failure left behind
            invalidate();
            ++failures_;
            if (dynamic_cast<const GraphError*>(&e)) throw;
            throw GraphError(std::string("graph capture failed: ") + e.what());
        }
        exec_ = exec;
        key_ = key;
        ++captures_;
    }

    void launch() {
        try {
            fault(Stage::Launch);
            check(cudaGraphLaunch(exec_, stream_), "launch graph");
        } catch (const std::exception& e) {
            cudaGetLastError();
            invalidate();  // a graph that failed to launch is not trusted again; the next step re-captures
            ++failures_;
            if (dynamic_cast<const GraphError*>(&e)) throw;
            throw GraphError(std::string("graph launch failed: ") + e.what());
        }
        ++replays_;
    }

    void invalidate() {
        if (exec_) { cudaGraphExecDestroy(exec_); exec_ = nullptr; }
        key_.clear();
    }

    long captures() const { return captures_; }
    long replays() const { return replays_; }
    long failures() const { return failures_; }

    // Test hook: the next `count` operations at `stage` fail as if the driver had refused them (count 0 clears it).
    static void inject_fault(Stage stage, int count = 1) {
        fault_stage().store(static_cast<int>(stage));
        fault_count().store(count);
    }

private:
    static std::atomic<int>& fault_stage() { static std::atomic<int> s{0}; return s; }
    static std::atomic<int>& fault_count() { static std::atomic<int> c{0}; return c; }
    static void fault(Stage at) {
        if (fault_stage().load() == static_cast<int>(at) && fault_count().load() > 0 && fault_count().fetch_sub(1) > 0)
            throw GraphError("injected graph failure");
    }
    static void check(cudaError_t e, const char* what) {
        if (e != cudaSuccess) throw GraphError(std::string(what) + ": " + cudaGetErrorString(e));
    }
    void reset() {
        invalidate();
        if (stream_ && owns_stream_) cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    cudaGraphExec_t exec_ = nullptr;
    std::vector<void*> key_;
    long captures_ = 0, replays_ = 0, failures_ = 0;
};

}  // namespace dara::model
