// Does this GPU execute the tensor-core mma instructions Dara's kernels use, and do they give the exact integer / expected FP16 results?
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

// int8: m8n8k16 (sm_75+). A 8x16 row-major, B 16x8 col-major, C 8x8 s32. One warp.
__global__ void mma_s8_m8n8k16(const int8_t* A, const int8_t* B, int* C) {
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    const uint32_t a = *reinterpret_cast<const uint32_t*>(A + gid * 16 + tig * 4);
    const uint32_t b = *reinterpret_cast<const uint32_t*>(B + gid * 16 + tig * 4);   // column gid of B, k = tig*4..+3
    int c0 = 0, c1 = 0;
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c0), "+r"(c1) : "r"(a), "r"(b));
    C[gid * 8 + tig * 2] = c0;
    C[gid * 8 + tig * 2 + 1] = c1;
}

// fp16: m16n8k8 (sm_75+). A 16x8 row-major, B 8x8 col-major, C 16x8 f32.
__global__ void mma_f16_m16n8k8(const __half* A, const __half* B, float* C) {
    const int lane = threadIdx.x, gid = lane >> 2, tig = lane & 3;
    const uint32_t a0 = *reinterpret_cast<const uint32_t*>(A + gid * 8 + tig * 2);
    const uint32_t a1 = *reinterpret_cast<const uint32_t*>(A + (gid + 8) * 8 + tig * 2);
    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(B + gid * 8 + tig * 2);
    float c[4] = {0, 0, 0, 0};
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
    C[gid * 8 + tig * 2] = c[0]; C[gid * 8 + tig * 2 + 1] = c[1];
    C[(gid + 8) * 8 + tig * 2] = c[2]; C[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

int main() {
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, 0);
    std::printf("%s sm_%d%d\n", p.name, p.major, p.minor);
    int8_t hA[8 * 16], hB[8 * 16];   // B stored column-major: column n = hB[n*16 .. +15]
    for (int i = 0; i < 128; ++i) { hA[i] = static_cast<int8_t>((i * 37) % 255 - 127); hB[i] = static_cast<int8_t>((i * 53) % 251 - 125); }
    int8_t *dA, *dB; int* dC;
    cudaMalloc(&dA, 128); cudaMalloc(&dB, 128); cudaMalloc(&dC, 64 * 4);
    cudaMemcpy(dA, hA, 128, cudaMemcpyHostToDevice); cudaMemcpy(dB, hB, 128, cudaMemcpyHostToDevice);
    mma_s8_m8n8k16<<<1, 32>>>(dA, dB, dC);
    cudaError_t e = cudaDeviceSynchronize();
    int hC[64]; cudaMemcpy(hC, dC, 256, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int m = 0; m < 8; ++m) for (int n = 0; n < 8; ++n) { int s = 0; for (int k = 0; k < 16; ++k) s += hA[m * 16 + k] * hB[n * 16 + k]; bad += s != hC[m * 8 + n]; }
    std::printf("int8 mma m8n8k16: %s, %d of 64 wrong\n", e == cudaSuccess ? "ran" : cudaGetErrorString(e), e == cudaSuccess ? bad : 64);
    __half hA2[16 * 8], hB2[8 * 8]; float ref[16 * 8];
    for (int i = 0; i < 128; ++i) hA2[i] = __float2half(static_cast<float>((i % 7) - 3) * 0.5f);
    for (int i = 0; i < 64; ++i) hB2[i] = __float2half(static_cast<float>((i % 5) - 2) * 0.25f);
    for (int m = 0; m < 16; ++m) for (int n = 0; n < 8; ++n) { float s = 0; for (int k = 0; k < 8; ++k) s += __half2float(hA2[m * 8 + k]) * __half2float(hB2[n * 8 + k]); ref[m * 8 + n] = s; }
    __half *dA2, *dB2; float* dC2;
    cudaMalloc(&dA2, sizeof hA2); cudaMalloc(&dB2, sizeof hB2); cudaMalloc(&dC2, sizeof ref);
    cudaMemcpy(dA2, hA2, sizeof hA2, cudaMemcpyHostToDevice); cudaMemcpy(dB2, hB2, sizeof hB2, cudaMemcpyHostToDevice);
    mma_f16_m16n8k8<<<1, 32>>>(dA2, dB2, dC2);
    e = cudaDeviceSynchronize();
    float hC2[128]; cudaMemcpy(hC2, dC2, sizeof hC2, cudaMemcpyDeviceToHost);
    bad = 0;
    for (int i = 0; i < 128; ++i) bad += hC2[i] != ref[i];
    std::printf("fp16 mma m16n8k8: %s, %d of 128 wrong\n", e == cudaSuccess ? "ran" : cudaGetErrorString(e), e == cudaSuccess ? bad : 128);
    return 0;
}
