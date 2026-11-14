#pragma once
// Quantization format registry and GEMV dispatch. The model backend deals only in QuantType; bit layouts,
// decoders and kernels stay behind this interface so each format can be optimized independently.
#include <cstddef>
#include <cstdint>
#include <string>

namespace dara::kernels {

enum class QuantType { Q4_K, Q5_K, Q6_K, Q8_0 };

struct QuantFormat {
    QuantType type;
    const char* name;      // GGUF type name
    int block_weights;     // weights per block; row length must be a multiple
    int block_bytes;
    int weight_align;      // required alignment of the weight pointer, in bytes
    // CPU references and a simple (valid, not error-minimizing) quantizer, on raw block bytes.
    void (*dequant_f32)(const void* blocks, size_t nblocks, float* out);   // ggml float arithmetic
    void (*dequant_f64)(const void* blocks, size_t nblocks, double* out);  // exact
    void (*quantize)(const float* x, size_t cols, void* blocks_out);
    void (*set_scales)(void* block, float d, float dmin);                  // dmin ignored by formats without one
};

const QuantFormat& quant_format(QuantType t);
bool quant_type_from_name(const std::string& gguf_name, QuantType& out);  // "Q4_K" -> Q4_K; false if unsupported

// y[r] = sum_c W[r][c] * x[c]; row r starts at W + r*ld_blocks*block_bytes. FP32 accumulation, sequential.
void gemv_quant_cpu_f32(QuantType t, const void* W, size_t ld_blocks, const float* x, float* y, size_t rows, size_t cols);
// Double-precision truth plus sum|w*x| per row (the scale of acceptable FP32 error).
void gemv_quant_cpu_f64(QuantType t, const void* W, size_t ld_blocks, const float* x, double* y, double* sum_abs,
                        size_t rows, size_t cols);

// Device pointers; returns a cudaError_t as int (0 on success). Asynchronous on `stream` (cudaStream_t or null).
// Validates cols, stride and alignment for the format and refuses bad input instead of faulting.
int gemv_quant_cuda(QuantType t, const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows,
                    size_t cols, void* stream = nullptr);

// Prefill GEMM (Q4_K only for now): y[t*ldy + r] = sum_c W[r][c] * x[t*ldx + c] for t < T, with 64x64 output tiles that decode each weight tile once and
// reuse it for 64 tokens. Same weight decoding as the GEMV kernels; accumulation order differs (relative difference ~1e-6). ldx multiple of 4.
int gemm_q4k_cuda(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream = nullptr);
int gemm_q5k_cuda(const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream = nullptr);
// Format dispatch. gemm_available(t) says whether a GEMM exists for the format (Q4_K, Q5_K); gemm_quant_cuda returns cudaErrorNotSupported otherwise.
// The GEMM pays off from about kGemmMinTokens tokens per call (below that the multi-vector GEMV is faster).
constexpr int kGemmMinTokens = 48;
bool gemm_available(QuantType t);
int gemm_quant_cuda(QuantType t, const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream = nullptr);

// int8-activation (dp4a) decode path for Q4_K / Q5_K (lossy: 8-bit activations; ~1.7x faster than the FP32 kernels on the GTX 1650, see gemv_q8.cu).
// quantize_activations_cuda fills `scratch` (q8act_bytes(K, T) bytes, 16-byte aligned) with T quantised vectors; gemv_q8_cuda multiplies a weight matrix
// by the `vec`-th of them. q8_supported(t) says which formats have a kernel.
size_t q8act_bytes(size_t K, int T);
// Prefill: Y[t*ldy + r] = W[r] . x_t for T int8 vectors in `scratch` (quantize_activations_cuda with T vectors); Q4_K / Q5_K. ~2.8x the FP32 GEMM.
bool gemm_q8_available(QuantType t);
// A few vectors (2..47; speculative verification): y[t*ldy + r] = W[r] . x_t for T int8 vectors in `scratch` (quantize_activations_cuda), Q4_K/Q5_K/Q6_K.
int gemv_q8_multi_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream = nullptr);
// Prefill BF16 GEMM (exact weights, FP32 math): W [rows][ld] bf16. Same tile as the quantised GEMM. Returns non-zero for unsupported shapes.
int gemm_bf16_cuda(const void* dW, size_t ld, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int T, void* stream = nullptr);
int gemm_q8_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, float* dy, size_t ldy, size_t rows, void* stream = nullptr);
int quantize_activations_cuda(const float* dx, size_t ldx, size_t K, int T, void* scratch, void* stream = nullptr);
bool q8_supported(QuantType t);
int gemv_q8_cuda(QuantType t, const void* dW, size_t ld_blocks, const void* scratch, size_t K, int T, int vec, float* dy, size_t rows, void* stream = nullptr);

// Multi-vector: y[t*ldy + r] = sum_c W[r][c] * x[t*ldx + c] for t < nt. Q4_K/Q6_K read each weight block once per group of up to 8
// vectors; every output is bitwise identical to gemv_quant_cuda on that vector. ldx must be a multiple of 4 floats and >= cols.
int gemv_quant_multi_cuda(QuantType t, const void* dW, size_t ld_blocks, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows,
                          size_t cols, int nt, void* stream = nullptr);

}  // namespace dara::kernels
