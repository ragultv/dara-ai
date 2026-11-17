#pragma once
// CPU execution of host-resident Qwen layers for single-token decode.
//
// Why: a model that does not fit in VRAM (Qwen3.5-9B on a 4 GB GPU: 19 of 32 layers in host RAM) used to stream every hosted layer's weights over PCIe
// for every generated token (2.4 GB/token -> 2.4 tok/s on the GTX 1650 laptop). Reading the same weights from system RAM on the CPU is several times faster
// than PCIe, and is what llama.cpp does for layers it does not offload. Prompts (many tokens) keep the GPU streaming path: there each weight is reused
// across hundreds of tokens, so the copy amortises.
//
// The CPU path reads the layer's packed GGUF image (PackedLayer::host, already in RAM) directly: AVX2 int8 dot products for Q4_K / Q5_K / Q6_K / Q8_0 with
// activations quantised per 32 values (the same scheme as the GPU int8 path), FP32 for everything else (norms, conv, DeltaNet recurrence, attention),
// following the validated CPU reference (deltanet.cpp, attention.cpp). Work is split over a persistent spinning thread pool.
//
// State: a hosted layer's recurrent state and K/V cache live in CpuLayerState in EXACTLY the GPU layout and format (FP32, or INT8 codes + per-32 scales),
// so handing the state between the GPU (prompt chunks) and the CPU (decode) is a plain memcpy (QwenModel tracks which side is current).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "kernels/quant.h"
#include "model/layer_cuda.h"
#include "model/packed_layer.h"

namespace dara::model {

// A quantised matrix in host memory (rows of GGUF blocks).
struct HostMatrix {
    const uint8_t* data = nullptr;
    dara::kernels::QuantType qtype = dara::kernels::QuantType::Q4_K;
    size_t rows = 0, cols = 0, row_bytes = 0;
};

struct CpuLayerWeights {
    LayerKind kind = LayerKind::DeltaNet;
    int hidden = 0, inter = 0;
    float eps = 1e-6f;
    std::vector<float> input_norm, post_norm;
    HostMatrix gate, up, down;
    DeltaNetDims dn;
    VHeadOrder order = VHeadOrder::Checkpoint;
    HostMatrix in_qkv, in_z, in_b, in_a, dn_out;
    std::vector<float> conv_w, ssm_a, dt_bias, norm_w;
    AttentionDims at;
    HostMatrix q_proj, k_proj, v_proj, o_proj;
    std::vector<float> q_norm, k_norm, inv_freq;
};

struct CpuLayerState {
    std::vector<float> conv, S;      // DeltaNet: same layout as GpuLayerState::conv_state / S
    KvFormat kv_format = KvFormat::F32;
    std::vector<float> K, V;         // F32 cache [position][kv_heads][head_dim]
    std::vector<int8_t> Kq, Vq;      // Q8 cache codes, same layout
    std::vector<float> Kn, Vn;       // Q8 scales [position][kv_heads][head_dim/32]
    size_t kv_cap = 0;
    const int64_t* rope3 = nullptr;   // multimodal RoPE: host [T][3] rotary positions for the next call (null: token positions)
    int mrope_h = 11, mrope_w = 10;
};

// True when this machine and build can run the CPU decode path (AVX2 + F16C), and the format/layer is supported.
bool cpu_decode_available();
bool cpu_decode_supports(KvFormat f);

class CpuDecoder {
public:
    explicit CpuDecoder(int threads = 0);   // 0: physical cores (hardware threads / 2), at least 1
    ~CpuDecoder();
    CpuDecoder(const CpuDecoder&) = delete;
    CpuDecoder& operator=(const CpuDecoder&) = delete;

    // Views the packed host image of `p` and downloads its small device parameters (norms, conv, gates, rope table). p.host must stay alive.
    static CpuLayerWeights make_weights(const PackedLayer& p);
    static CpuLayerState make_state(const CpuLayerWeights& w, size_t kv_capacity, KvFormat fmt);

    // One token through one layer: x [hidden] -> y [hidden] (y may alias x), at absolute position `pos` (the KV length before this token).
    void forward(const CpuLayerWeights& w, CpuLayerState& s, const float* x, float* y, int64_t pos);
    // T consecutive tokens (a speculative verification step): x/y [T][hidden] (y may alias x), first token at position pos0. Each weight row is read
    // from RAM once for all T tokens; the result is bitwise that of T single-token forwards. tape: optional DeltaNet recurrence inputs (tape_floats).
    void forward_multi(const CpuLayerWeights& w, CpuLayerState& s, const float* x, float* y, int64_t pos0, int T, float* tape);
    static size_t tape_floats(const CpuLayerWeights& w, int T);
    // After restoring a DeltaNet layer's state to its value before a T-token forward_multi, re-applies the first `keep` tokens from the tape.
    void replay(const CpuLayerWeights& w, CpuLayerState& s, const float* tape, int T, int keep);
    // y [T][M.rows] = M x_t for T vectors x [T][M.cols] (the host-RAM output head in a verification step): M read once.
    void matvec_multi(const HostMatrix& M, const float* x, int T, float* y);

    // y = M x for one vector (the output head when it lives in host RAM). M.cols % 256 == 0 for K-quants.
    void matvec(const HostMatrix& M, const float* x, float* y);

    // State hand-over with the GPU copy of the same layer. kv_len: live K/V positions to copy.
    static void download(const GpuLayerState& g, CpuLayerState& c, size_t kv_len);
    static void upload(const CpuLayerState& c, GpuLayerState& g, size_t kv_len);

    int threads() const;
    // Keep the worker threads spinning (true) ahead of a burst of CPU work -- e.g. while the GPU runs its layers -- so the first CPU jobs do not wait
    // for threads to wake from sleep; false lets them back off to yield/sleep.
    void set_hot(bool on);

    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace dara::model
