// CUDA bring-up check: compile (nvcc + MSVC), allocate, host->device, launch, device->host, verify.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::printf("CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void saxpy(int n, float a, const float* x, const float* y, float* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::puts("no CUDA device"); return 1; }
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    std::printf("device: %s  sm_%d%d  VRAM free %.0f / %.0f MiB\n", p.name, p.major, p.minor,
                free_b / 1048576.0, total_b / 1048576.0);

    const int n = 1 << 20;
    std::vector<float> x(n), y(n), out(n, -1.0f);
    for (int i = 0; i < n; ++i) { x[i] = static_cast<float>(i % 1000); y[i] = 0.5f * (i % 7); }

    float *dx = nullptr, *dy = nullptr, *dout = nullptr;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dout, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, y.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    saxpy<<<(n + 255) / 256, 256>>>(n, 2.0f, dx, dy, dout);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), dout, n * sizeof(float), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < n; ++i)
        if (out[i] != 2.0f * x[i] + y[i]) ++bad;   // exact: all values are small multiples of 0.5
    CUDA_CHECK(cudaFree(dx)); CUDA_CHECK(cudaFree(dy)); CUDA_CHECK(cudaFree(dout));
    std::printf("saxpy over %d elements: %d mismatches\n", n, bad);
    std::puts(bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
