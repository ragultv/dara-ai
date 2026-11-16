#pragma once
// Dara Flash: the trained speculative drafter (EAGLE-style), runtime side. Trained by tools/flash/train_drafter.py, exported by tools/flash/export_drafter.py.
//
// Per position t the drafter reads the target's final (normed) hidden state f_t and the embedding of the token that followed, x_{t+1}:
//     y = Layer(fc([f_t ; emb(x_{t+1})]))      one Qwen3.5-style attention layer + SwiGLU MLP (executed by gpu_layer_forward_dev: same kernels as the target)
//     y ~ f_{t+1},  draft token = argmax over a reduced vocabulary of (head . y), head rows = the target's own output head rows (raw GGUF blocks)
// Drafting k tokens feeds y back as the next f. After verification the drafter is rewound and EXTENDED with the target's true states of the accepted
// positions, so its KV cache always describes the real sequence.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "io/gguf_checkpoint.h"
#include "model/decode_graph.h"
#include "model/device_buffer.h"
#include "model/layer_cuda.h"
#include "model/packed_layer.h"

namespace dara::model {

class FlashDrafter {
public:
    // path: the exported drafter file; g: the target GGUF (for the output-head rows); embed: the target's dequantised token_embd row (unscaled).
    using EmbedFn = std::function<std::vector<float>(int32_t)>;
    FlashDrafter(const std::string& path, const dara::io::GgufCheckpoint& g, EmbedFn embed, size_t kv_capacity);
    ~FlashDrafter();

    // Feeds n real positions: feats [n][hidden] = target states f_t, next [n] = x_{t+1}; appends to the drafter's cache at its current position.
    // Returns the drafter's prediction for the position after the last one (feature + token), ready to start drafting.
    void extend(const float* feats, const int32_t* next, int n, std::vector<float>& y_last, int32_t& tok_last);
    float last_prob() const { return last_p_; }   // the drafter's probability of its last picked token (reduced-vocabulary softmax)
    // Drafts up to k tokens after (y, tok) -- tok is the first draft token (already predicted, with probability p_first). Stops early once the
    // running product of the drafter's probabilities falls below min_prob (a likely rejection is not worth a verification slot).
    void draft(std::vector<float> y, int32_t tok, float p_first, int k, float min_prob, std::vector<int32_t>& draft);
    size_t position() const;
    void rewind(size_t pos);
    int hidden() const { return H_; }

private:
    void step(const float* d_in2h, int n, float* d_y);   // fc + layer for n positions (inputs on the device), d_y [n][hidden]
    int32_t pick(const float* d_y_row);                   // reduced-vocab argmax -> target token id
    // One drafted position: input [g_y_ (the drafter's previous predicted state, on the device) ; emb(tok)], output g_y_. Replayed as a CUDA graph
    // (DARA_FLASH_GRAPH=0: eager); only the embedding row goes up and the picked id + probability come back.
    int32_t draft_step(int32_t tok);
    void draft_body();
    DecodeGraph graph_;
    DevBuf<float> g_in_, g_x_, g_y_, g_logits_, g_out_;
    DevBuf<int64_t> g_pos_;
    float* h_buf_ = nullptr;     // pinned: embedding row [H] + {id bits, probability}
    int64_t* h_pos_ = nullptr;   // pinned position
    bool graphs_ = true;
    int graph_failures_ = 0;
    int H_ = 0, K_ = 0;
    float last_p_ = 1.0f;
    std::vector<int32_t> keep_;
    DeviceMatrix fc_, head_;
    PackedLayer layer_;
    GpuLayerState state_;
    ScratchPool pool_;
    EmbedFn embed_;
    DevBuf<unsigned char> d_argmax_;
};

}  // namespace dara::model
