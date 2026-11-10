#pragma once
#include <cstdint>
#include <vector>

#include "core/manifest.h"
#include "core/placement.h"
#include "core/session.h"

namespace dara::core {

struct TokenBatch {
    std::vector<int32_t> tokens;
    // Image spans and position ids join here when the vision path lands.
};

struct DecodeResult {
    int32_t token = 0;
    bool eos = false;
};

// Execution contract shared by CPU and CUDA backends and by the Qwen and Gemma families.
//  - Session owns sequence state (tokens, page table, opaque backend state slot).
//  - Backend owns model computation and decides what its BackendState contains.
//  - BlockPool owns KV memory.
// Errors are reported by exception. prefill, decode and restore give the strong guarantee: if
// they throw, the session is exactly as it was before the call.
class IModelBackend {
public:
    virtual ~IModelBackend() = default;

    // Rejects a manifest that does not match what this backend implements.
    virtual void load(const ModelManifest& manifest, const PlacementPlan& placement) = 0;
    virtual void unload() = 0;

    virtual void prefill(Session& session, const TokenBatch& batch) = 0;
    virtual DecodeResult decode(Session& session) = 0;

    // Captures KV position and architecture-specific state. Supports speculative rollback later.
    virtual Checkpoint checkpoint(const Session& session) = 0;
    // Validates first (owner, length, prefix hash), then rewinds the session to the checkpoint.
    virtual void restore(Session& session, const Checkpoint& cp) = 0;
};

}  // namespace dara::core
