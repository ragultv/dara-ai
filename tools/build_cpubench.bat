@echo off
setlocal
set CUDA=D:\tools\cuda\13.1.1
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
set NVFLAGS=-std=c++20 -O3 -Xcompiler /O2 -arch=sm_75 -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\cpu_kernel_bench.cpp src\model\cpu_decode.cpp src\model\ops.cpp src\model\deltanet.cpp src\model\packed_layer.cu src\model\layer_cuda.cu src\model\kv4.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\attention.cpp src\model\layer.cpp src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\kernels\q4k_gemv.cu src\kernels\q4k_cpu.cpp src\kernels\q5k_gemv.cu src\kernels\q80_gemv.cu src\kernels\q5k_q80_cpu.cpp src\kernels\q6k_gemv.cu src\kernels\q6k_cpu.cpp src\kernels\quant_gemv.cu src\kernels\gemv_multi.cu src\kernels\gemv16.cu src\kernels\gemv_q8.cu src\kernels\gemv_q6kp.cu src\kernels\gemv_q8_multi.cu src\kernels\gemm_q8.cu src\kernels\gemm_q8_q6k.cu src\kernels\gemm_bf16.cu src\kernels\gemm_f16.cu src\kernels\gemm_q4k.cu src\kernels\quant.cpp -o build\cuda\cpu_kernel_bench.exe -lcudart || exit /b 1
echo built build\cuda\cpu_kernel_bench.exe
