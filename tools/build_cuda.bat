@echo off
rem Builds the CUDA tests with the pinned redist toolkit and MSVC 2022 Build Tools (outside CMake for now).
setlocal
set CUDA=D:\tools\cuda\13.1.1
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
where cl >nul 2>nul || (echo cl.exe not on PATH after vcvars & exit /b 1)
cd /d %~dp0..
if not exist build\cuda mkdir build\cuda
rem Target GPUs: every consumer NVIDIA laptop/desktop architecture CUDA 13 supports -- Turing (GTX 16xx, RTX 20xx: sm_75), Ampere
rem (RTX 30xx: sm_86), Ada (RTX 40xx: sm_89), Blackwell (RTX 50xx: sm_120) -- plus compute_120 PTX so newer cards still run.
rem Pascal and older (GTX 10xx) are not supported by CUDA 13. For fast development builds set DARA_ARCH=75 (or 86, 89, 120).
if not defined DARA_ARCH set DARA_ARCH=all
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
echo CUDA targets: %GENCODE%
set NVFLAGS=-std=c++20 %GENCODE% -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
set K=src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\kernels\q4k_gemv.cu src\kernels\q4k_cpu.cpp src\kernels\q5k_gemv.cu src\kernels\q80_gemv.cu src\kernels\q5k_q80_cpu.cpp src\kernels\q6k_gemv.cu src\kernels\q6k_cpu.cpp src\kernels\quant_gemv.cu src\kernels\gemv_multi.cu src\kernels\gemv16.cu src\kernels\gemv_q8.cu src\kernels\gemm_q8.cu src\kernels\gemm_q8_q6k.cu src\kernels\gemm_bf16.cu src\kernels\gemm_q4k.cu src\kernels\quant.cpp

"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\cuda_smoke.cu -o build\cuda\cuda_smoke.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\gemv_test.cu %K% src\core\manifest.cpp -o build\cuda\gemv_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\q4k_test.cu %K% -o build\cuda\q4k_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\q6k_test.cu %K% -o build\cuda\q6k_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\gemv_multi_test.cu %K% -o build\cuda\gemv_multi_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\quant_test.cu %K% -o build\cuda\quant_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\workload_bench.cu %K% src\core\gpu_telemetry.cpp -o build\cuda\workload_bench.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\deltanet_cuda_test.cu src\model\deltanet_cuda.cu src\model\deltanet.cpp src\core\gpu_telemetry.cpp -o build\cuda\deltanet_cuda_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\ops_cuda_test.cu src\model\ops_cuda.cu src\model\ops.cpp -o build\cuda\ops_cuda_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\attention_cuda_test.cu src\model\attention_cuda.cu src\model\attention.cpp src\model\ops_cuda.cu src\model\ops.cpp src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\core\gpu_telemetry.cpp -o build\cuda\attention_cuda_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\layer_cuda_test.cu src\model\layer_cuda.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\attention.cpp src\model\deltanet.cpp src\model\ops.cpp src\model\layer.cpp %K% -o build\cuda\layer_cuda_test.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tools\qwen_generate.cu src\tokenizer\unicode.cpp src\tokenizer\tokenizer.cpp src\model\qwen_model.cu src\model\qwen_weights.cu src\model\packed_layer.cu src\io\gguf_checkpoint.cpp src\io\gguf.cpp src\model\layer_cuda.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\ops.cpp src\model\deltanet.cpp src\model\attention.cpp src\model\layer.cpp src\core\system.cpp src\core\placement.cpp src\core\budget.cpp src\core\manifest.cpp %K% -o build\cuda\qwen_generate.exe -lcudart || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\qwen_graph_test.cu src\tokenizer\unicode.cpp src\tokenizer\tokenizer.cpp src\model\qwen_model.cu src\model\qwen_weights.cu src\model\packed_layer.cu src\io\gguf_checkpoint.cpp src\io\gguf.cpp src\model\layer_cuda.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\ops.cpp src\model\deltanet.cpp src\model\attention.cpp src\model\layer.cpp src\core\system.cpp src\core\placement.cpp src\core\budget.cpp src\core\manifest.cpp %K% -o build\cuda\qwen_graph_test.exe -lcudart || exit /b 1
copy /y "%CUDA%\bin\x64\cudart64_13.dll" build\cuda\ >nul
rem Nano tier (Gemma 4 E2B) driver and the Gemma tokenizer test: same toolchain and GPU targets (DARA_ARCH applies to both).
call "%~dp0build_nano.bat" || exit /b 1
call "%~dp0build_tokenizer_tests.bat" || exit /b 1
echo built CUDA tests in build\cuda
