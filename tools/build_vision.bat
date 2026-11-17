@echo off
rem Builds the vision parity test (Gemma 4 / Nano image encoder vs the transformers reference). Same toolchain and GPU targets as
rem build_cuda.bat (set DARA_ARCH=75 for a fast single-architecture development build).
setlocal
set CUDA=D:\tools\cuda\13.1.1
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not exist build\cuda mkdir build\cuda
if not defined DARA_ARCH set DARA_ARCH=all
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
echo CUDA targets: %GENCODE%
set NVFLAGS=-std=c++20 -O3 -Xcompiler /O2 %GENCODE% -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
set K=src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\kernels\q4k_gemv.cu src\kernels\q4k_cpu.cpp src\kernels\q5k_gemv.cu src\kernels\q80_gemv.cu src\kernels\q5k_q80_cpu.cpp src\kernels\q6k_gemv.cu src\kernels\q6k_cpu.cpp src\kernels\quant_gemv.cu src\kernels\gemv_multi.cu src\kernels\gemv16.cu src\kernels\gemv_q8.cu src\kernels\gemv_q8_multi.cu src\kernels\gemm_q8.cu src\kernels\gemm_q8_q6k.cu src\kernels\gemm_bf16.cu src\kernels\gemm_q4k.cu src\kernels\quant.cpp
set M=src\model\layer_cuda.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\ops.cpp src\model\deltanet.cpp src\model\attention.cpp src\model\layer.cpp src\model\packed_layer.cu
set C=src\io\gguf_checkpoint.cpp src\io\gguf.cpp src\core\system.cpp src\core\placement.cpp src\core\budget.cpp src\core\manifest.cpp
set V=src\model\gemma_vision.cu src\model\vision_ops_cuda.cu src\model\gemma_ops_cuda.cu src\io\image.cpp

"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\vision_parity.cu %V% %M% %C% %K% -o build\cuda\vision_parity.exe -lcudart || exit /b 1
echo built build\cuda\vision_parity.exe
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\qwen_vision_parity.cu src\model\qwen_vision.cu %V% %M% %C% %K% -o build\cuda\qwen_vision_parity.exe -lcudart || exit /b 1
echo built build\cuda\qwen_vision_parity.exe
