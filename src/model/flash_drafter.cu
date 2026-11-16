#include "model/flash_drafter.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "kernels/quant.h"
#include "model/ops.h"
#include "model/ops_cuda.h"

namespace dara::model {

namespace {
struct Reader {
    std::ifstream f;
    explicit Reader(const std::string& p) : f(p, std::ios::binary) { if (!f) throw std::runtime_error("cannot open drafter file " + p); }
    template <class T> T get() { T v; f.read(reinterpret_cast<char*>(&v), sizeof v); if (!f) throw std::runtime_error("truncated drafter file"); return v; }
    template <class T> std::vector<T> vec(size_t n) { std::vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T))); if (!f) throw std::runtime_error("truncated drafter file"); return v; }
};
size_t q80_bytes(size_t rows, size_t cols) { return rows * (cols / 32) * 34; }

// The file stores Q8_0; the runtime re-quantises to the format with the fastest multi-token kernels (Q5_K int8 path by default:
// DARA_FLASH_Q=q8|q5|q4). Q8_0 has only a generic per-token kernel, which made the drafter ~3.5 ms per position.
dara::kernels::QuantType runtime_qtype() {
    const char* e = std::getenv("DARA_FLASH_Q");
    const std::string v = e ? e : "q5";
    if (v == "q8") return dara::kernels::QuantType::Q8_0;
    if (v == "q4") return dara::kernels::QuantType::Q4_K;
    return dara::kernels::QuantType::Q5_K;
}
std::vector<uint8_t> requant(const std::vector<uint8_t>& q80, size_t rows, size_t cols, dara::kernels::QuantType to) {
    if (to == dara::kernels::QuantType::Q8_0) return q80;
    const auto& src = dara::kernels::quant_format(dara::kernels::QuantType::Q8_0);
    const auto& dst = dara::kernels::quant_format(to);
    const size_t drow = cols / dst.block_weights * dst.block_bytes, srow = cols / 32 * 34;
    std::vector<uint8_t> out(rows * drow);
    std::vector<float> x(cols);
    for (size_t r = 0; r < rows; ++r) {
        src.dequant_f32(q80.data() + r * srow, cols / 32, x.data());
        dst.quantize(x.data(), cols, out.data() + r * drow);
    }
    return out;
}
// One block: argmax and softmax probability of the max over n logits (n <= a few 100K). out = {index as int bits, probability}.
__global__ void argmax_prob_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    __shared__ float sv[32];
    __shared__ int si[32];
    __shared__ float smax;
    float best = -INFINITY; int bi = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float v = x[i]; if (v > best) { best = v; bi = i; } }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, best, off); const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) { sv[w] = best; si[w] = bi; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float b = sv[0]; int i0 = si[0];
        for (int k = 1; k < static_cast<int>(blockDim.x >> 5); ++k) if (sv[k] > b || (sv[k] == b && si[k] < i0)) { b = sv[k]; i0 = si[k]; }
        smax = b; reinterpret_cast<int*>(out)[0] = i0;
    }
    __syncthreads();
    const float m = smax;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(x[i] - m);
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
    __syncthreads();
    if (l == 0) sv[w] = s;
    __syncthreads();
    if (threadIdx.x == 0) { float t = 0; for (int k = 0; k < static_cast<int>(blockDim.x >> 5); ++k) t += sv[k]; out[1] = 1.0f / t; }
}
}  // namespace

// File layout (tools/flash/export_drafter.py): "DFLASH01", int32 hidden heads kv_heads head_dim rot inter K, float32 theta eps, int32 keep[K],
// then Q8_0 matrices fc [H][2H], q [2*heads*hd][H], k [kv*hd][H], v [kv*hd][H], o [H][heads*hd], gate [inter][H], up [inter][H], down [H][inter],
// then float32 in_norm[H], q_norm[hd], k_norm[hd], post_norm[H].
FlashDrafter::FlashDrafter(const std::string& path, const dara::io::GgufCheckpoint& g, EmbedFn embed, size_t kv_capacity) : embed_(std::move(embed)) {
    Reader r(path);
    char magic[8];
    r.f.read(magic, 8);
    if (std::memcmp(magic, "DFLASH01", 8) != 0) throw std::runtime_error("not a Dara Flash drafter file: " + path);
    H_ = r.get<int32_t>();
    const int heads = r.get<int32_t>(), kvh = r.get<int32_t>(), hd = r.get<int32_t>(), rot = r.get<int32_t>(), inter = r.get<int32_t>();
    K_ = r.get<int32_t>();
    const float theta = r.get<float>(), eps = r.get<float>();
    keep_ = r.vec<int32_t>(static_cast<size_t>(K_));
    auto mat = [&](size_t rows, size_t cols) { return r.vec<uint8_t>(q80_bytes(rows, cols)); };
    const auto fc = mat(H_, 2 * static_cast<size_t>(H_));
    struct M { DeviceMatrix GpuLayerWeights::* member; size_t rows, cols; std::vector<uint8_t> bytes; };
    std::vector<M> ms;
    ms.push_back({&GpuLayerWeights::q_proj, static_cast<size_t>(2 * heads * hd), static_cast<size_t>(H_), {}});
    ms.push_back({&GpuLayerWeights::k_proj, static_cast<size_t>(kvh * hd), static_cast<size_t>(H_), {}});
    ms.push_back({&GpuLayerWeights::v_proj, static_cast<size_t>(kvh * hd), static_cast<size_t>(H_), {}});
    ms.push_back({&GpuLayerWeights::o_proj, static_cast<size_t>(H_), static_cast<size_t>(heads * hd), {}});
    ms.push_back({&GpuLayerWeights::gate, static_cast<size_t>(inter), static_cast<size_t>(H_), {}});
    ms.push_back({&GpuLayerWeights::up, static_cast<size_t>(inter), static_cast<size_t>(H_), {}});
    ms.push_back({&GpuLayerWeights::down, static_cast<size_t>(H_), static_cast<size_t>(inter), {}});
    const auto rq = runtime_qtype();
    for (M& m : ms) m.bytes = requant(mat(m.rows, m.cols), m.rows, m.cols, rq);
    const auto in_norm = r.vec<float>(H_), q_norm = r.vec<float>(hd), k_norm = r.vec<float>(hd), post_norm = r.vec<float>(H_);

    GpuLayerWeights& w = layer_.weights;
    w.kind = LayerKind::Attention;
    w.hidden = H_; w.inter = inter; w.eps = eps;
    w.at.hidden = H_; w.at.heads = heads; w.at.kv_heads = kvh; w.at.head_dim = hd; w.at.rot_dim = rot; w.at.eps = eps; w.at.rope_theta = theta;
    w.input_norm = DevBuf<float>(in_norm); w.post_norm = DevBuf<float>(post_norm);
    w.q_norm = DevBuf<float>(q_norm); w.k_norm = DevBuf<float>(k_norm);
    w.inv_freq = DevBuf<float>(rope_inv_freq(rot, theta));
    size_t off = 0;
    for (const M& m : ms) { layer_.slots.push_back({m.member, rq, m.rows, m.cols, off, m.bytes.size()}); off = (off + m.bytes.size() + 255) / 256 * 256; }
    layer_.matrix_bytes = off;
    layer_.host = HostBuffer(off);
    for (size_t i = 0; i < ms.size(); ++i) std::memcpy(layer_.host.data() + layer_.slots[i].offset, ms[i].bytes.data(), ms[i].bytes.size());
    make_resident(layer_);
    state_ = make_gpu_layer_state(w, kv_capacity, KvFormat::F32);
    fc_ = upload_quant_raw(rq, requant(fc, static_cast<size_t>(H_), 2 * static_cast<size_t>(H_), rq).data(), static_cast<size_t>(H_), 2 * static_cast<size_t>(H_));

    // reduced-vocabulary head: the target's own output-head rows (raw GGUF blocks, so the int8 kernels apply)
    const bool tied = !g.has("output.weight");
    const std::string hn = tied ? "token_embd.weight" : "output.weight";
    const auto& hi = g.info(hn);
    const uint32_t type = hi.type;
    const dara::kernels::QuantType hq = type == 12 ? dara::kernels::QuantType::Q4_K : type == 13 ? dara::kernels::QuantType::Q5_K
                                       : type == 14 ? dara::kernels::QuantType::Q6_K : dara::kernels::QuantType::Q8_0;
    const auto& f = dara::kernels::quant_format(hq);
    const size_t row_bytes = hi.cols() / f.block_weights * f.block_bytes;
    std::vector<uint8_t> rows(static_cast<size_t>(K_) * row_bytes);
    for (int i = 0; i < K_; ++i) {
        const auto b = g.read_rows(hn, static_cast<uint64_t>(keep_[i]), 1);
        std::memcpy(rows.data() + static_cast<size_t>(i) * row_bytes, b.data(), row_bytes);
    }
    head_ = upload_quant_raw(hq, rows.data(), static_cast<size_t>(K_), hi.cols());
    d_argmax_ = DevBuf<unsigned char>(argmax_scratch_bytes() + 64);
    // single-position drafting buffers (graph)
    graph_.create_stream();
    g_in_ = DevBuf<float>(2 * static_cast<size_t>(H_)); g_x_ = DevBuf<float>(static_cast<size_t>(H_)); g_y_ = DevBuf<float>(static_cast<size_t>(H_));
    g_logits_ = DevBuf<float>(static_cast<size_t>(K_)); g_out_ = DevBuf<float>(2); g_pos_ = DevBuf<int64_t>(1);
    void* hp = nullptr;
    cuda_check(cudaHostAlloc(&hp, (static_cast<size_t>(H_) + 4) * sizeof(float) + sizeof(int64_t), cudaHostAllocDefault), "flash pinned");
    h_buf_ = static_cast<float*>(hp);
    h_pos_ = reinterpret_cast<int64_t*>(h_buf_ + H_ + 2);
    if (const char* e = std::getenv("DARA_FLASH_GRAPH")) graphs_ = e[0] != '0';
}

FlashDrafter::~FlashDrafter() { if (h_buf_) cudaFreeHost(h_buf_); }
size_t FlashDrafter::position() const { return state_.pos; }
void FlashDrafter::rewind(size_t pos) { if (pos < state_.pos) state_.pos = pos; }

void FlashDrafter::step(const float* d_in2h, int n, float* d_y) {
    float* x = pool_.get<float>(50, static_cast<size_t>(n) * H_);
    cuda_check(matvec_multi_cuda(fc_, d_in2h, 2 * static_cast<size_t>(H_), x, static_cast<size_t>(H_), n), "flash fc");
    gpu_layer_forward_dev(layer_.weights, state_, x, d_y, n, pool_, nullptr);
}

int32_t FlashDrafter::pick(const float* d_y_row) {
    float* logits = pool_.get<float>(51, static_cast<size_t>(K_));
    cuda_check(matvec_multi_cuda(head_, d_y_row, static_cast<size_t>(H_), logits, static_cast<size_t>(K_), 1), "flash head");
    float* d_out = reinterpret_cast<float*>(d_argmax_.p);
    argmax_prob_kernel<<<1, 1024>>>(logits, K_, d_out);
    cuda_check(cudaGetLastError(), "flash argmax");
    float hv[2];
    cuda_check(cudaMemcpy(hv, d_out, sizeof hv, cudaMemcpyDeviceToHost), "flash id");
    int id = 0;
    std::memcpy(&id, &hv[0], sizeof id);
    last_p_ = hv[1];
    return keep_[static_cast<size_t>(id)];
}

void FlashDrafter::extend(const float* feats, const int32_t* next, int n, std::vector<float>& y_last, int32_t& tok_last) {
    std::vector<float> in(static_cast<size_t>(n) * 2 * H_);
    for (int t = 0; t < n; ++t) {
        std::memcpy(&in[static_cast<size_t>(t) * 2 * H_], feats + static_cast<size_t>(t) * H_, static_cast<size_t>(H_) * sizeof(float));
        const std::vector<float> e = embed_(next[t]);
        std::memcpy(&in[static_cast<size_t>(t) * 2 * H_ + H_], e.data(), static_cast<size_t>(H_) * sizeof(float));
    }
    float* d_in = pool_.get<float>(52, in.size());
    float* d_y = pool_.get<float>(53, static_cast<size_t>(n) * H_);
    cuda_check(cudaMemcpy(d_in, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice), "flash in");
    step(d_in, n, d_y);
    const float* last = d_y + static_cast<size_t>(n - 1) * H_;
    cuda_check(cudaMemcpyAsync(g_y_.p, last, static_cast<size_t>(H_) * sizeof(float), cudaMemcpyDeviceToDevice, 0), "flash y keep");   // drafting starts here
    tok_last = pick(last);
    y_last.resize(H_);
    cuda_check(cudaMemcpy(y_last.data(), last, static_cast<size_t>(H_) * sizeof(float), cudaMemcpyDeviceToHost), "flash y");
}

void FlashDrafter::draft(std::vector<float> y, int32_t tok, float p_first, int k, float min_prob, std::vector<int32_t>& out) {
    out.clear();
    if (k <= 0 || p_first < min_prob) return;
    out.push_back(tok);
    float run = p_first;
    (void)y;   // the predicted state stays on the device (g_y_)
    for (int i = 1; i < k; ++i) {
        tok = draft_step(tok);   // the drafter's own predicted state stands in for the target's
        run *= last_p_;
        if (run < min_prob) break;
        out.push_back(tok);
    }
}

void FlashDrafter::draft_body() {
    cudaStream_t s = graph_.stream();
    cuda_check(cudaMemcpyAsync(g_in_.p, g_y_.p, static_cast<size_t>(H_) * sizeof(float), cudaMemcpyDeviceToDevice, s), "flash y in");
    cuda_check(matvec_multi_cuda(fc_, g_in_.p, 2 * static_cast<size_t>(H_), g_x_.p, static_cast<size_t>(H_), 1, s), "flash fc");
    gpu_layer_forward_dev(layer_.weights, state_, g_x_.p, g_y_.p, 1, pool_, nullptr, s, g_pos_.p);
    cuda_check(matvec_multi_cuda(head_, g_y_.p, static_cast<size_t>(H_), g_logits_.p, static_cast<size_t>(K_), 1, s), "flash head");
    argmax_prob_kernel<<<1, 1024, 0, s>>>(g_logits_.p, K_, g_out_.p);
    cuda_check(cudaGetLastError(), "flash argmax");
}

int32_t FlashDrafter::draft_step(int32_t tok) {
    if (state_.pos + 1 > state_.kv_cap) throw std::length_error("drafter K/V capacity exceeded");
    cudaStream_t s = graph_.stream();
    const std::vector<float> e = embed_(tok);
    std::memcpy(h_buf_, e.data(), static_cast<size_t>(H_) * sizeof(float));
    *h_pos_ = static_cast<int64_t>(state_.pos);
    cuda_check(cudaMemcpyAsync(g_in_.p + H_, h_buf_, static_cast<size_t>(H_) * sizeof(float), cudaMemcpyHostToDevice, s), "flash emb");
    cuda_check(cudaMemcpyAsync(g_pos_.p, h_pos_, sizeof(int64_t), cudaMemcpyHostToDevice, s), "flash pos");
    bool done = false;
    if (graphs_) {
        try {
            std::vector<void*> key;
            for (size_t slot = 1; slot <= 20; ++slot) key.push_back(pool_.get<unsigned char>(slot, 1));
            if (!graph_.ready_for(key)) graph_.capture(key, [&] { draft_body(); });
            graph_.launch();
            done = true;
        } catch (const GraphError&) {
            graph_.invalidate();   // typically a scratch buffer that had to grow: the eager run below sizes it, the next step re-captures
            if (++graph_failures_ >= 3) graphs_ = false;
        }
    }
    if (!done) draft_body();
    cuda_check(cudaMemcpyAsync(h_buf_ + H_, g_out_.p, 2 * sizeof(float), cudaMemcpyDeviceToHost, s), "flash id");
    cuda_check(cudaStreamSynchronize(s), "flash step");
    state_.pos += 1;   // the device-position path does not advance it
    int id = 0;
    std::memcpy(&id, &h_buf_[H_], sizeof id);
    last_p_ = h_buf_[H_ + 1];
    return keep_[static_cast<size_t>(id)];
}

}  // namespace dara::model
