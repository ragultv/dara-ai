// GEMV dispatch: QuantType -> the format's fused dequantize + GEMV kernel.
#include <cuda_runtime.h>

#include "kernels/q4k.h"
#include "kernels/q5k.h"
#include "kernels/q6k.h"
#include "kernels/q8_0.h"
#include "kernels/quant.h"

namespace dara::kernels {

int gemv_quant_cuda(QuantType t, const void* dW, size_t ld_blocks, const float* dx, float* dy, size_t rows,
                    size_t cols, void* stream) {
    switch (t) {
        case QuantType::Q4_K: return gemv_q4k_cuda(dW, ld_blocks, dx, dy, rows, cols, stream);
        case QuantType::Q5_K: return gemv_q5k_cuda(dW, ld_blocks, dx, dy, rows, cols, stream);
        case QuantType::Q6_K: return gemv_q6k_cuda(dW, ld_blocks, dx, dy, rows, cols, stream);
        case QuantType::Q8_0: return gemv_q80_cuda(dW, ld_blocks, dx, dy, rows, cols, stream);
    }
    return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace dara::kernels
