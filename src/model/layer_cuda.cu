#include "model/layer_cuda.h"
#include "model/attn_prefill.h"
#include "model/attn_splitkv.h"
#include "model/tq_kv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "kernels/gemv.h"
#include "kernels/half.h"
#include "model/attention_cuda.h"
#include "model/deltanet_cuda.h"
#include "model/ops_cuda.h"

namespace dara::model {

static DeviceMatrix make_matrix(const void* host_bytes, size_t bytes, size_t rows, size_t cols, size_t ld) {
    void* d = counted_malloc(bytes);
    DeviceMatrix m;
    m.data = std::shared_ptr<void>(d, [bytes](void* p) { counted_free(p, bytes); });
    cuda_check(cudaMemcpy(d, host_bytes, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(matrix)");
    m.rows = rows; m.cols = cols; m.ld = ld; m.bytes = bytes;
    return m;
}

DeviceMatrix upload_bf16_matrix(const std::vector<float>& host, size_t rows, size_t cols) {
    if (host.size() != rows * cols) throw std::invalid_argument("matrix size mismatch");
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = dara::kernels::float_to_bf16(host[i]);
    return make_matrix(bits.data(), bits.size() * 2, rows, cols, cols);
}

DeviceMatrix upload_bf16_raw(const uint16_t* bits, size_t rows, size_t cols) {
    return make_matrix(bits, rows * cols * 2, rows, cols, cols);
}

DeviceMatrix upload_quant_raw(dara::kernels::QuantType t, const void* blocks, size_t rows, size_t cols) {
    const auto& f = dara::kernels::quant_format(t);
    if (cols % f.block_weights != 0) throw std::invalid_argument("quantized row is not a whole number of blocks");
    const size_t ld = cols / f.block_weights;
    DeviceMatrix m = make_matrix(blocks, rows * ld * f.block_bytes, rows, cols, ld);
    m.quantized = true;
    m.qtype = t;
    return m;
}

int matvec_cuda(const DeviceMatrix& m, const float* d_x, float* d_y, void* stream) {
    if (!m.data) return static_cast<int>(cudaErrorInvalidValue);
    if (m.quantized) return dara::kernels::gemv_quant_cuda(m.qtype, m.data.get(), m.ld, d_x, d_y, m.rows, m.cols, stream);
    return dara::kernels::gemv_cuda(dara::kernels::WeightType::BF16, m.data.get(), m.ld, d_x, d_y, m.rows, m.cols, stream);
}

// Opt-in lossy fast decode (DARA_INT8=1): int8 activations + dp4a for Q4_K/Q5_K single-vector matvecs (see kernels/gemv_q8.cu). The scratch is allocated up
// front by int8_init() (called from the model constructors, outside any graph capture) so a captured graph never allocates.
bool int8_enabled() {
    // DEFAULT ON (quality-gated: teacher-forced top-1 >= 126/128 on English / Tamil / code / JSON / 1.5K / 8K for 4B and Nano). DARA_INT8=0 selects the
    // FP32 reference path (bitwise regression against the RC baselines).
    static const bool on = [] { const char* e = std::getenv("DARA_INT8"); return !(e && e[0] == '0'); }();
    return on;
}
static void* g_q8_scratch = nullptr;
static constexpr size_t kQ8ScratchBytes = 256 * 1024;   // covers K up to ~50k
int int8_init() {
    if (!int8_enabled() || g_q8_scratch) return 0;
    return static_cast<int>(cudaMalloc(&g_q8_scratch, kQ8ScratchBytes));
}

static const float* g_share_x = nullptr;   // input declared shared by the active Int8SharedInput scope
static size_t g_share_cols = 0;            // non-zero once g_q8_scratch holds g_share_x quantised (for this many columns)
static size_t g_share_big_key[3] = {0, 0, 0};   // prefill: (cols, nt, ldx) for which g_q8_big holds g_share_x quantised
Int8SharedInput::Int8SharedInput(const float* x) { g_share_x = x; g_share_cols = 0; g_share_big_key[0] = 0; }
Int8SharedInput::~Int8SharedInput() { g_share_x = nullptr; g_share_cols = 0; g_share_big_key[0] = 0; }

// Prefill (nt tokens): int8 activations for the dp4a GEMM, in a grow-only buffer. Prefill is never graph-captured; while a capture is active this path is
// skipped anyway so it can never allocate inside a graph.
static void* g_q8_big = nullptr;
static size_t g_q8_big_bytes = 0;
static int int8_gemm(const DeviceMatrix& m, const float* d_x, size_t ldx, float* d_y, size_t ldy, int nt, void* stream) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) { cudaGetLastError(); return -1; }
    const size_t need = dara::kernels::q8act_bytes(m.cols, nt);
    if (need > g_q8_big_bytes) {
        if (g_q8_big) { cudaStreamSynchronize(static_cast<cudaStream_t>(stream)); counted_free(g_q8_big, g_q8_big_bytes); g_q8_big = nullptr; g_q8_big_bytes = 0; }
        const size_t bytes = std::max(need, static_cast<size_t>(8) << 20);
        g_q8_big = counted_malloc(bytes);
        g_q8_big_bytes = bytes;
        g_share_big_key[0] = 0;
    }
    const bool reuse = d_x == g_share_x && g_share_big_key[0] == m.cols && g_share_big_key[1] == static_cast<size_t>(nt) && g_share_big_key[2] == ldx;
    if (!reuse) {
        if (dara::kernels::quantize_activations_cuda(d_x, ldx, m.cols, nt, g_q8_big, stream) != 0) { g_share_big_key[0] = 0; return -1; }
        if (d_x == g_share_x) { g_share_big_key[0] = m.cols; g_share_big_key[1] = static_cast<size_t>(nt); g_share_big_key[2] = ldx; }
        else g_share_big_key[0] = 0;
    }
    return dara::kernels::gemm_q8_cuda(m.qtype, m.data.get(), m.ld, g_q8_big, m.cols, nt, d_y, ldy, m.rows, stream);
}

int matvec_multi_cuda(const DeviceMatrix& m, const float* d_x, size_t ldx, float* d_y, size_t ldy, int nt, void* stream) {
    if (!m.data) return static_cast<int>(cudaErrorInvalidValue);
    if (m.quantized) {
        // Q6_K has an int8 kernel too, but on its 2-byte-aligned blocks it has not yet measured faster than the FP32 kernel: DARA_INT8_Q6=1 opts in.
        static const bool q6 = [] { const char* e = std::getenv("DARA_INT8_Q6"); return e && e[0] == '1'; }();
        if (nt == 1 && g_q8_scratch && dara::kernels::q8_supported(m.qtype) && (m.qtype != dara::kernels::QuantType::Q6_K || q6) && m.cols % 256 == 0 && dara::kernels::q8act_bytes(m.cols, 1) <= kQ8ScratchBytes) {
            // Inside an Int8SharedInput scope for d_x the vector is quantised once and reused by every matrix that reads it; any other input invalidates it.
            const bool reuse = d_x == g_share_x && g_share_cols == m.cols;
            bool ok = reuse;
            if (!reuse) {
                ok = dara::kernels::quantize_activations_cuda(d_x, ldx, m.cols, 1, g_q8_scratch, stream) == 0;
                g_share_cols = (ok && d_x == g_share_x) ? m.cols : 0;
            }
            if (ok && dara::kernels::gemv_q8_cuda(m.qtype, m.data.get(), m.ld, g_q8_scratch, m.cols, 1, 0, d_y, m.rows, stream) == 0) return 0;
        }
        if (int8_enabled() && nt >= dara::kernels::kGemmMinTokens && ldy % 4 == 0 && dara::kernels::gemm_q8_available(m.qtype) && m.cols % 256 == 0) {
            if (int8_gemm(m, d_x, ldx, d_y, ldy, nt, stream) == 0) return 0;   // otherwise fall through to the FP32 paths
        }
        // Prefill chunks: a tiled GEMM decodes each weight tile once and reuses it for 64 tokens (about 1.6-2.2x the multi-vector GEMV from ~48 tokens up).
        // DARA_NO_GEMM=1 keeps the GEMV path (bitwise regression against the RC1/RC2 baselines).
        static const bool no_gemm = [] { const char* e = std::getenv("DARA_NO_GEMM"); return e && e[0] == '1'; }();
        if (!no_gemm && nt >= dara::kernels::kGemmMinTokens && m.rows >= 128 && ldy % 4 == 0 && dara::kernels::gemm_available(m.qtype)) {
            const int e = dara::kernels::gemm_quant_cuda(m.qtype, m.data.get(), m.ld, d_x, ldx, d_y, ldy, m.rows, m.cols, nt, stream);
            if (e == 0) return 0;  // otherwise (e.g. an alignment the GEMM refuses) fall through to the GEMV path
        }
        return dara::kernels::gemv_quant_multi_cuda(m.qtype, m.data.get(), m.ld, d_x, ldx, d_y, ldy, m.rows, m.cols, nt, stream);
    }
    {   // BF16 prefill: tiled GEMM from kGemmMinTokens tokens (DARA_NO_GEMM=1 keeps the GEMV path, bitwise identical to the per-token loop)
        static const bool no_gemm = [] { const char* e = std::getenv("DARA_NO_GEMM"); return e && e[0] == '1'; }();
        if (!no_gemm && nt >= dara::kernels::kGemmMinTokens && ldy % 4 == 0 &&
            dara::kernels::gemm_bf16_cuda(m.data.get(), m.ld, d_x, ldx, d_y, ldy, m.rows, m.cols, nt, stream) == 0)
            return 0;
    }
    if (nt > 1) {  // BF16: read each weight once per group of 8 tokens (bitwise identical to the per-token loop)
        const int e = dara::kernels::gemv_multi_cuda(dara::kernels::WeightType::BF16, m.data.get(), m.ld, d_x, ldx, d_y, ldy, m.rows, m.cols, nt, stream);
        if (e == 0) return 0;  // otherwise fall back to the per-token loop below
    }
    for (int t = 0; t < nt; ++t) {
        const int e = matvec_cuda(m, d_x + static_cast<size_t>(t) * ldx, d_y + static_cast<size_t>(t) * ldy, stream);
        if (e) return e;
    }
    return 0;
}

size_t GpuLayerWeights::total_bytes() const {
    size_t b = (input_norm.n + post_norm.n + conv_w.n + ssm_a.n + dt_bias.n + norm_w.n + q_norm.n + k_norm.n + inv_freq.n) * sizeof(float);
    for (const DeviceMatrix* m : {&gate, &up, &down, &in_qkv, &in_z, &in_b, &in_a, &dn_out, &q_proj, &k_proj, &v_proj, &o_proj}) b += m->bytes;
    return b;
}

GpuLayerWeights upload_layer_weights(const DecoderLayerWeights& w) {
    GpuLayerWeights s;
    s.kind = w.kind; s.hidden = w.hidden; s.inter = w.mlp.inter; s.eps = w.eps;
    s.input_norm = DevBuf<float>(w.input_norm);
    s.post_norm = DevBuf<float>(w.post_norm);
    s.gate = upload_bf16_matrix(w.mlp.gate, w.mlp.inter, w.hidden);
    s.up = upload_bf16_matrix(w.mlp.up, w.mlp.inter, w.hidden);
    s.down = upload_bf16_matrix(w.mlp.down, w.hidden, w.mlp.inter);
    if (w.kind == LayerKind::DeltaNet) {
        s.dn = w.dn.dims;
        s.order = w.dn.order;
        s.in_qkv = upload_bf16_matrix(w.dn.in_qkv, s.dn.conv_dim(), s.dn.hidden);
        s.in_z = upload_bf16_matrix(w.dn.in_z, s.dn.value_dim(), s.dn.hidden);
        s.in_b = upload_bf16_matrix(w.dn.in_b, s.dn.v_heads, s.dn.hidden);
        s.in_a = upload_bf16_matrix(w.dn.in_a, s.dn.v_heads, s.dn.hidden);
        s.dn_out = upload_bf16_matrix(w.dn.out, s.dn.hidden, s.dn.value_dim());
        s.conv_w = DevBuf<float>(w.dn.conv_w);
        s.ssm_a = DevBuf<float>(w.dn.ssm_a);
        s.dt_bias = DevBuf<float>(w.dn.dt_bias);
        s.norm_w = DevBuf<float>(w.dn.norm_w);
    } else {
        s.at = w.at.dims;
        s.q_proj = upload_bf16_matrix(w.at.q_proj, s.at.q_rows(), s.at.hidden);
        s.k_proj = upload_bf16_matrix(w.at.k_proj, s.at.kv_width(), s.at.hidden);
        s.v_proj = upload_bf16_matrix(w.at.v_proj, s.at.kv_width(), s.at.hidden);
        s.o_proj = upload_bf16_matrix(w.at.o_proj, s.at.hidden, s.at.out_width());
        s.q_norm = DevBuf<float>(w.at.q_norm);
        s.k_norm = DevBuf<float>(w.at.k_norm);
        s.inv_freq = DevBuf<float>(w.at.inv_freq);
    }
    return s;
}

// DARA_ATTN_REF=1 selects the original single-block attention kernels (bitwise regression against the RC1/RC2 baselines); the default is split-KV.
static bool attention_reference_mode() {
    static const bool r = [] { const char* e = std::getenv("DARA_ATTN_REF"); return e && e[0] == '1'; }();
    return r;
}

namespace {
struct LayerProf {
    bool on = false;
    std::vector<cudaEvent_t> ev;
    std::vector<const char*> label;
    size_t used = 0;
    std::vector<std::pair<std::string, double>> total_ms;
    bool decode_only = false;   // DARA_LAYER_PROF=2: time single-token steps only (decode at long context without the prompt's events)
    LayerProf() { const char* e = std::getenv("DARA_LAYER_PROF"); on = e && (e[0] == '1' || e[0] == '2'); decode_only = e && e[0] == '2'; }
    void mark(const char* l, void* stream, int T) {
        if (!on || (decode_only && T != 1)) return;
        if (used >= ev.size()) { cudaEvent_t e; cudaEventCreate(&e); ev.push_back(e); }
        cudaEventRecord(ev[used], static_cast<cudaStream_t>(stream));
        if (label.size() <= used) label.resize(used + 1);
        label[used++] = l;
    }
    void flush() {
        if (!on || used < 2) { used = 0; return; }
        for (size_t i = 1; i < used; ++i) {
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[i - 1], ev[i]);
            auto it = std::find_if(total_ms.begin(), total_ms.end(), [&](const auto& kv) { return kv.first == label[i]; });
            if (it == total_ms.end()) { total_ms.push_back({label[i], 0.0}); it = total_ms.end() - 1; }
            it->second += ms;
        }
        used = 0;
    }
};
LayerProf& lprof() { static LayerProf p; return p; }
}  // namespace
void layer_prof_flush() { lprof().flush(); }
std::string layer_prof_report() {
    LayerProf& p = lprof();
    double sum = 0;
    for (auto& kv : p.total_ms) sum += kv.second;
    std::string r;
    char b[160];
    std::snprintf(b, sizeof b, "layer-path GPU time by operation (all forward calls, %.1f ms total)\n", sum);
    r += b;
    for (auto& kv : p.total_ms) { std::snprintf(b, sizeof b, "  %-28s %9.1f ms  %5.1f%%\n", kv.first.c_str(), kv.second, 100.0 * kv.second / sum); r += b; }
    return r;
}
#define LP(label) lprof().mark(label, stream, T)

GpuLayerState make_gpu_layer_state(const GpuLayerWeights& w, size_t kv_capacity, KvFormat fmt) {
    GpuLayerState s;
    s.kind = w.kind;
    if (w.kind == LayerKind::DeltaNet) {
        s.conv_state = DevBuf<float>(deltanet_conv_state_floats(w.dn));
        s.S = DevBuf<float>(deltanet_S_floats(w.dn));
    } else {
        s.kv_cap = kv_capacity;
        s.kv_format = fmt;
        if (fmt == KvFormat::K16V8) {
            s.Kq = DevBuf<uint8_t>(kv_capacity * w.at.kv_width() * 2);
            s.Vq = DevBuf<uint8_t>(kv_capacity * w.at.kv_width());
            s.Vn = DevBuf<float>(kv_capacity * w.at.kv_width() / 32);
        } else if (fmt == KvFormat::Q8) {
            s.Kq = DevBuf<uint8_t>(kv_capacity * w.at.kv_width());
            s.Vq = DevBuf<uint8_t>(kv_capacity * w.at.kv_width());
            s.Kn = DevBuf<float>(kv_capacity * w.at.kv_width() / 32);
            s.Vn = DevBuf<float>(kv_capacity * w.at.kv_width() / 32);
        } else if (fmt == KvFormat::TQ4) {
            s.Kq = DevBuf<uint8_t>(kv_capacity * w.at.kv_heads * (w.at.head_dim / 2));
            s.Vq = DevBuf<uint8_t>(kv_capacity * w.at.kv_heads * (w.at.head_dim / 2));
            s.Kn = DevBuf<float>(kv_capacity * w.at.kv_heads);
            s.Vn = DevBuf<float>(kv_capacity * w.at.kv_heads);
        } else {
            s.Kc = DevBuf<float>(kv_capacity * w.at.kv_width());
            s.Vc = DevBuf<float>(kv_capacity * w.at.kv_width());
        }
    }
    return s;
}

GpuLayerCheckpoint gpu_layer_checkpoint(const GpuLayerState& s) {
    GpuLayerCheckpoint c;
    c.length = s.pos;
    if (s.kind == LayerKind::DeltaNet) { c.conv = s.conv_state.get(); c.S = s.S.get(); }
    return c;
}

void gpu_layer_restore(GpuLayerState& s, const GpuLayerCheckpoint& c) {
    if (s.kind == LayerKind::DeltaNet) {
        if (c.conv.size() != s.conv_state.n || c.S.size() != s.S.n) throw std::invalid_argument("checkpoint does not match this layer");
        s.conv_state.put(c.conv.data(), c.conv.size());
        s.S.put(c.S.data(), c.S.size());
    } else if (c.length > s.pos) {
        throw std::invalid_argument("KV checkpoint is ahead of the layer");  // the cache can only be rewound
    }
    s.pos = c.length;
}

namespace {
__global__ void fill_positions_kernel(int64_t* p, int64_t first, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = first + i;
}
}  // namespace

// Device-to-device layer step: reads x (device, [T][hidden]) and writes `out` (device); never touches the host and never
// synchronizes, so a whole token can be queued without a host round trip. Scratch slots 1..7 and 10..16 are used.
size_t deltanet_tape_floats(const GpuLayerWeights& w, int T) {
    const DeltaNetDims& d = w.dn;
    return static_cast<size_t>(T) * (static_cast<size_t>(d.conv_dim()) + static_cast<size_t>(d.value_dim()) + 2 * static_cast<size_t>(d.v_heads));
}

void gpu_deltanet_replay(const GpuLayerWeights& w, GpuLayerState& s, const float* tape, int T, int keep, ScratchPool& sp) {
    const DeltaNetDims& d = w.dn;
    const size_t CD = static_cast<size_t>(d.conv_dim()), VD = static_cast<size_t>(d.value_dim()), VH = static_cast<size_t>(d.v_heads);
    const float* qkv = tape;
    const float* z = qkv + static_cast<size_t>(T) * CD;
    const float* b = z + static_cast<size_t>(T) * VD;
    const float* a = b + static_cast<size_t>(T) * VH;
    float* scratch = sp.get<float>(14, CD);
    float* sink = sp.get<float>(15, VD);  // the per-token outputs are not needed for a replay
    DeltaNetDeviceParams params{w.conv_w.p, w.ssm_a.p, w.dt_bias.p, w.norm_w.p};
    for (int t = 0; t < keep; ++t)
        cuda_check(deltanet_step_cuda(d, w.order, params, s.conv_state.p, s.S.p, qkv + t * CD, z + t * VD, b + t * VH, a + t * VH, scratch, sink), "deltanet replay");
}

void gpu_layer_forward_dev(const GpuLayerWeights& w, GpuLayerState& s, const float* dx, float* out, int T, ScratchPool& sp, float* tape, void* stream,
                           const int64_t* dev_pos) {
    if (T <= 0) return;
    if (s.kind != w.kind) throw std::invalid_argument("layer state kind does not match the layer weights");
    const int H = w.hidden, I = w.inter;
    const size_t TH = static_cast<size_t>(T) * H;
    float* h1 = sp.get<float>(1, TH);
    float* m = sp.get<float>(2, TH);
    float* x1 = sp.get<float>(3, TH);
    float* h2 = sp.get<float>(4, TH);
    float* g = sp.get<float>(5, static_cast<size_t>(T) * I);
    float* u = sp.get<float>(6, static_cast<size_t>(T) * I);
    float* f = sp.get<float>(7, TH);

    LP("start");
    cuda_check(rmsnorm_cuda(dx, w.input_norm.p, T, H, w.eps, h1, stream), "input norm");
    LP("norms / residuals");

    if (w.kind == LayerKind::DeltaNet) {
        const DeltaNetDims& d = w.dn;
        const size_t CD = static_cast<size_t>(d.conv_dim()), VD = static_cast<size_t>(d.value_dim()), VH = static_cast<size_t>(d.v_heads);
        float* qkv = sp.get<float>(10, static_cast<size_t>(T) * CD);
        float* z = sp.get<float>(11, static_cast<size_t>(T) * VD);
        float* b = sp.get<float>(12, static_cast<size_t>(T) * VH);
        float* a = sp.get<float>(13, static_cast<size_t>(T) * VH);
        float* scratch = sp.get<float>(14, CD);
        float* nm = sp.get<float>(15, static_cast<size_t>(T) * VD);
        DeltaNetDeviceParams params{w.conv_w.p, w.ssm_a.p, w.dt_bias.p, w.norm_w.p};
        // The projections depend only on the normed input, not on the recurrent state: batch them over all T tokens so each weight
        // matrix is read once per group of 8 tokens, then run the (inherently sequential) recurrence token by token.
        {
        Int8SharedInput share(h1);
        cuda_check(matvec_multi_cuda(w.in_qkv, h1, H, qkv, CD, T, stream), "in_qkv");
        cuda_check(matvec_multi_cuda(w.in_z, h1, H, z, VD, T, stream), "in_z");
        cuda_check(matvec_multi_cuda(w.in_b, h1, H, b, VH, T, stream), "in_b");
        cuda_check(matvec_multi_cuda(w.in_a, h1, H, a, VH, T, stream), "in_a");
        }
        LP("dn projections qkv/z/b/a");
        if (tape) {  // record the recurrence inputs (device-to-device, ~0.2 MB per token per layer)
            float* tp = tape;
            cuda_check(cudaMemcpyAsync(tp, qkv, static_cast<size_t>(T) * CD * 4, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "tape qkv"); tp += static_cast<size_t>(T) * CD;
            cuda_check(cudaMemcpyAsync(tp, z, static_cast<size_t>(T) * VD * 4, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "tape z"); tp += static_cast<size_t>(T) * VD;
            cuda_check(cudaMemcpyAsync(tp, b, static_cast<size_t>(T) * VH * 4, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "tape b"); tp += static_cast<size_t>(T) * VH;
            cuda_check(cudaMemcpyAsync(tp, a, static_cast<size_t>(T) * VH * 4, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)), "tape a");
        }
        for (int t = 0; t < T; ++t)
            cuda_check(deltanet_step_cuda(d, w.order, params, s.conv_state.p, s.S.p, qkv + t * CD, z + t * VD, b + t * VH, a + t * VH, scratch,
                                          nm + t * VD, stream), "deltanet step");
        LP("deltanet recurrence");
        cuda_check(matvec_multi_cuda(w.dn_out, nm, VD, m, H, T, stream), "out_proj");
        LP("dn out_proj");
    } else {
        const AttentionDims& d = w.at;
        if (s.pos + static_cast<size_t>(T) > s.kv_cap) throw std::length_error("KV cache capacity exceeded");
        float* qg = sp.get<float>(10, static_cast<size_t>(T) * d.q_rows());
        float* k = sp.get<float>(11, static_cast<size_t>(T) * d.kv_width());
        float* v = sp.get<float>(12, static_cast<size_t>(T) * d.kv_width());
        float* q = sp.get<float>(13, static_cast<size_t>(T) * d.heads * d.head_dim);
        float* gate = sp.get<float>(14, static_cast<size_t>(T) * d.out_width());
        float* att = sp.get<float>(15, static_cast<size_t>(T) * d.out_width());
        int64_t* dpos = sp.get<int64_t>(16, static_cast<size_t>(T));
        {
        Int8SharedInput share(h1);
        cuda_check(matvec_multi_cuda(w.q_proj, h1, H, qg, static_cast<size_t>(d.q_rows()), T, stream), "q_proj");
        cuda_check(matvec_multi_cuda(w.k_proj, h1, H, k, static_cast<size_t>(d.kv_width()), T, stream), "k_proj");
        cuda_check(matvec_multi_cuda(w.v_proj, h1, H, v, static_cast<size_t>(d.kv_width()), T, stream), "v_proj");
        }
        LP("attn q/k/v proj");
        cuda_check(attn_split_qgate_cuda(d, qg, T, q, gate, stream), "split q|gate");
        cuda_check(rmsnorm_cuda(q, w.q_norm.p, T * d.heads, d.head_dim, d.eps, q, stream), "q_norm");
        cuda_check(rmsnorm_cuda(k, w.k_norm.p, T * d.kv_heads, d.head_dim, d.eps, k, stream), "k_norm");
        const int64_t* pos_dev = dev_pos;
        if (!dev_pos) {
            fill_positions_kernel<<<(T + 127) / 128, 128, 0, static_cast<cudaStream_t>(stream)>>>(dpos, static_cast<int64_t>(s.pos), T);
            cuda_check(cudaGetLastError(), "fill positions");
            pos_dev = dpos;
        }
        cuda_check(rope_cuda(q, T, d.heads, d.head_dim, d.rot_dim, pos_dev, w.inv_freq.p, stream), "rope q");
        cuda_check(rope_cuda(k, T, d.kv_heads, d.head_dim, d.rot_dim, pos_dev, w.inv_freq.p, stream), "rope k");
        LP("attn split/norm/rope");
        const bool ref = attention_reference_mode();
        const int maxpos = dev_pos ? static_cast<int>(s.kv_cap) : static_cast<int>(s.pos) + T;  // grid bound: fixed for a captured graph
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(d.head_dim));
        const AttnFmt afmt = s.kv_format == KvFormat::TQ4 ? AttnFmt::TQ4 : s.kv_format == KvFormat::Q8 ? AttnFmt::Q8 : s.kv_format == KvFormat::K16V8 ? AttnFmt::K16V8 : AttnFmt::F32;
        // prefill chunks (positions on the host, many tokens) on an FP32 or INT8 cache: GEMM -> masked softmax -> GEMM, gate applied after
        const bool prefill_attn = !ref && !dev_pos && T >= kPrefillAttnMinTokens && s.kv_format != KvFormat::TQ4;
        auto prefill_attention = [&] {
            const int p0 = static_cast<int>(s.pos);
            const size_t pf = attn_prefill_scratch_floats(T, p0, 0, d.heads);
            float* ps = sp.get<float>(18, pf);
            const bool q8 = kv_q8_values(s.kv_format);   // K/V in Kq/Vq (Q8 or K16V8)
            cuda_check(attn_prefill_cuda(afmt, q, gate, q8 ? static_cast<const void*>(s.Kq.p) : s.Kc.p, q8 ? s.Kn.p : nullptr,
                                         q8 ? static_cast<const void*>(s.Vq.p) : s.Vc.p, q8 ? s.Vn.p : nullptr, T, p0, d.heads, d.kv_heads, d.head_dim, 0, attn_scale, ps, pf,
                                         att, stream),
                       "prefill attention");
        };
        auto split_attention = [&] {
            const size_t per = attn_splitkv_scratch_floats_per_token(afmt, d.heads, d.head_dim, maxpos);
            const size_t floats = per * static_cast<size_t>(std::min(T, 8));
            float* scr = sp.get<float>(17, floats);
            cuda_check(afmt != AttnFmt::F32
                           ? attn_splitkv_cuda(afmt, q, gate, s.Kq.p, s.Kn.p, s.Vq.p, s.Vn.p, pos_dev, T, d.heads, d.kv_heads, d.head_dim, 0, attn_scale, maxpos, scr, floats, att, stream)
                           : attn_splitkv_cuda(afmt, q, gate, s.Kc.p, nullptr, s.Vc.p, nullptr, pos_dev, T, d.heads, d.kv_heads, d.head_dim, 0, attn_scale, maxpos, scr, floats, att, stream),
                       "split-KV attention");
        };
        if (kv_q8_values(s.kv_format)) {   // int8 values (+ int8 or fp16 keys): quantise-and-append, split-KV attention reads them directly (no reference kernel)
            if (s.kv_format == KvFormat::K16V8) cuda_check(f16kv_append_cuda(k, pos_dev, T, d.kv_heads, d.head_dim, s.Kq.p, stream), "f16 append K");
            else cuda_check(q8kv_append_cuda(k, pos_dev, T, d.kv_heads, d.head_dim, s.Kq.p, s.Kn.p, stream), "q8 append K");
            cuda_check(q8kv_append_cuda(v, pos_dev, T, d.kv_heads, d.head_dim, s.Vq.p, s.Vn.p, stream), "q8 append V");
            LP("kv append");
            if (prefill_attn) prefill_attention(); else split_attention();
        } else if (s.kv_format == KvFormat::TQ4) {  // 4-bit cache: quantise-and-append, then attend in the rotated domain without expanding the cache
            cuda_check(tq_append_cuda(k, pos_dev, T, d.kv_heads, d.head_dim, s.Kq.p, s.Kn.p, stream), "tq append K");
            cuda_check(tq_append_cuda(v, pos_dev, T, d.kv_heads, d.head_dim, s.Vq.p, s.Vn.p, stream), "tq append V");
            if (ref) {
                const int smem = dev_pos ? static_cast<int>(std::min<size_t>(s.kv_cap, kTqMaxPositions)) : static_cast<int>(s.pos) + T;
                cuda_check(tq_attention_cuda(q, gate, s.Kq.p, s.Kn.p, s.Vq.p, s.Vn.p, pos_dev, T, d.heads, d.kv_heads, d.head_dim, 0, attn_scale, smem, att, stream), "tq attention");
            } else {
                split_attention();
            }
        } else {
            if (dev_pos) cuda_check(attn_kv_append_dev_cuda(d, s.Kc.p, s.Vc.p, k, v, dev_pos, T, stream), "kv append");
            else cuda_check(attn_kv_append_cuda(d, s.Kc.p, s.Vc.p, k, v, static_cast<int>(s.pos), T, stream), "kv append");
            if (prefill_attn) prefill_attention();
            else if (!ref) split_attention();
            else if (dev_pos) cuda_check(attention_core_dev_cuda(d, q, gate, s.Kc.p, s.Vc.p, dev_pos, T, static_cast<int>(std::min<size_t>(s.kv_cap, kAttentionMaxKvLen)), att, stream), "attention");
            else cuda_check(attention_core_cuda(d, q, gate, s.Kc.p, s.Vc.p, static_cast<int>(s.pos), T, att, stream), "attention");
        }
        LP("attention core + kv append");
        cuda_check(matvec_multi_cuda(w.o_proj, att, static_cast<size_t>(d.out_width()), m, H, T, stream), "o_proj");
        LP("attn o_proj");
    }
    if (!dev_pos) s.pos += T;  // graph decode: the caller advances positions on the host

    cuda_check(add_cuda(dx, m, x1, TH, stream), "residual 1");
    cuda_check(rmsnorm_cuda(x1, w.post_norm.p, T, H, w.eps, h2, stream), "post norm");
    {
    Int8SharedInput share(h2);
    cuda_check(matvec_multi_cuda(w.gate, h2, H, g, static_cast<size_t>(I), T, stream), "gate_proj");
    cuda_check(matvec_multi_cuda(w.up, h2, H, u, static_cast<size_t>(I), T, stream), "up_proj");
    }
    LP("mlp gate/up");
    cuda_check(swiglu_cuda(g, u, g, static_cast<size_t>(T) * I, stream), "swiglu");
    LP("swiglu");
    cuda_check(matvec_multi_cuda(w.down, g, static_cast<size_t>(I), f, H, T, stream), "down_proj");
    LP("mlp down");
    cuda_check(add_cuda(x1, f, out, TH, stream), "residual 2");
    LP("norms / residuals");
}

// Host-vector convenience wrapper (single-layer tests and traces): upload, run, download (the download synchronizes).
std::vector<float> gpu_layer_forward(const GpuLayerWeights& w, GpuLayerState& s, const float* x, int T, LayerTrace* trace, ScratchPool* pool) {
    if (T <= 0) return {};
    ScratchPool local;
    ScratchPool& sp = pool ? *pool : local;
    const size_t TH = static_cast<size_t>(T) * w.hidden;
    float* dx = sp.get<float>(0, TH);
    float* out = sp.get<float>(8, TH);
    float* h1 = sp.get<float>(1, TH);
    float* m = sp.get<float>(2, TH);
    float* x1 = sp.get<float>(3, TH);
    float* h2 = sp.get<float>(4, TH);
    float* f = sp.get<float>(7, TH);
    cuda_check(cudaMemcpy(dx, x, TH * sizeof(float), cudaMemcpyHostToDevice), "upload hidden");
    gpu_layer_forward_dev(w, s, dx, out, T, sp);
    std::vector<float> result(TH);
    cuda_check(cudaMemcpy(result.data(), out, TH * sizeof(float), cudaMemcpyDeviceToHost), "download hidden");
    if (trace) {
        auto dl = [&](float* p) { std::vector<float> v(TH); cuda_check(cudaMemcpy(v.data(), p, TH * sizeof(float), cudaMemcpyDeviceToHost), "trace"); return v; };
        trace->normed1 = dl(h1); trace->mixer = dl(m); trace->resid1 = dl(x1);
        trace->normed2 = dl(h2); trace->mlp = dl(f); trace->out = result;
    }
    return result;
}

}  // namespace dara::model
