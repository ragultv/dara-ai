#pragma once
// Decode GEMV:  y[r] = sum_c W[r*ld + c] * x[c]   (W is [rows x cols], row stride ld >= cols)
// W is stored as 16-bit (FP16 or BF16), x and y are FP32, accumulation is FP32.
#include <cstddef>
#include <cstdint>

#include "kernels/half.h"

namespace dara::kernels {

// CPU references. Sequential FP32 accumulation (what a naive engine does) and a double-precision
// "truth" used to bound both CPU and GPU error.
void gemv_cpu_f32(WeightType t, const uint16_t* W, size_t ld, const float* x, float* y, size_t rows, size_t cols);
void gemv_cpu_f64(WeightType t, const uint16_t* W, size_t ld, const float* x, double* y, double* sum_abs,
                  size_t rows, size_t cols);

#ifdef __CUDACC__
// Device pointers. Returns a cudaError_t (as int); 0 on success. Asynchronous on `stream`.
int gemv_cuda(WeightType t, const void* dW, size_t ld, const float* dx, float* dy, size_t rows, size_t cols,
              void* stream = nullptr);
// Multi-vector: y[k*ldy + r] = sum_c W[r*ld + c] * x[k*ldx + c] for k < nt, reading each weight once per group of up to 8 vectors; every output is
// bitwise identical to gemv_cuda on that vector. Needs 16-byte aligned W/x, ld % 8 == 0, ldx % 4 == 0 (else cudaErrorMisalignedAddress: use gemv_cuda).
int gemv_multi_cuda(WeightType t, const void* dW, size_t ld, const float* dx, size_t ldx, float* dy, size_t ldy, size_t rows, size_t cols, int nt,
                    void* stream = nullptr);
#endif

}  // namespace dara::kernels
