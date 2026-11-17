@echo off
rem Builds dara.exe: one executable for every tier. The architecture in the GGUF picks the driver (qwen35 -> Core 4B / Pro 9B,
rem gemma4 -> Nano). The two drivers are compiled with main() renamed and linked with the union of the model sources.
rem Same toolchain and GPU targets as build_cuda.bat (set DARA_ARCH=75 for a fast single-architecture development build).
rem The CUDA runtime is linked statically: dara.exe needs only the NVIDIA driver (no cudart DLL next to it).
rem   DARA_CUDA  CUDA toolkit folder (default D:\tools\cuda\13.1.1; tools\install_cuda_redist.py installs one without admin rights)
rem   Visual Studio 2022 (any edition with the C++ x64 tools) is found with vswhere.
setlocal
if not defined DARA_CUDA set DARA_CUDA=D:\tools\cuda\13.1.1
set "CUDA=%DARA_CUDA%"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set VSDIR=
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (echo Visual Studio with the C++ x64 tools was not found & exit /b 1)
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not exist build\cuda\dara_obj mkdir build\cuda\dara_obj
if not defined DARA_ARCH set DARA_ARCH=all
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
echo CUDA targets: %GENCODE%
set NVFLAGS=-std=c++20 -O3 -Xcompiler /O2 %GENCODE% --threads 0 -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
set K=src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\kernels\q4k_gemv.cu src\kernels\q4k_cpu.cpp src\kernels\q5k_gemv.cu src\kernels\q80_gemv.cu src\kernels\q5k_q80_cpu.cpp src\kernels\q6k_gemv.cu src\kernels\q6k_cpu.cpp src\kernels\quant_gemv.cu src\kernels\gemv_multi.cu src\kernels\gemv16.cu src\kernels\gemv_q8.cu src\kernels\gemv_q8_multi.cu src\kernels\gemm_q8.cu src\kernels\gemm_q8_q6k.cu src\kernels\gemm_bf16.cu src\kernels\gemm_q4k.cu src\kernels\quant.cpp
set M=src\model\layer_cuda.cu src\model\tq_kv.cu src\model\attn_splitkv.cu src\model\attn_prefill.cu src\model\attention_cuda.cu src\model\deltanet_cuda.cu src\model\ops_cuda.cu src\model\ops.cpp src\model\deltanet.cpp src\model\attention.cpp src\model\layer.cpp src\model\packed_layer.cu
set C=src\io\gguf_checkpoint.cpp src\io\gguf.cpp src\core\system.cpp src\core\placement.cpp src\core\budget.cpp src\core\manifest.cpp
set Q=src\model\qwen_model.cu src\model\cpu_decode.cpp src\model\qwen_weights.cu src\tokenizer\tokenizer.cpp src\model\qwen_vision.cu
set G=src\model\gemma_model.cu src\model\gemma_ops_cuda.cu src\tokenizer\gemma_tokenizer.cpp src\tokenizer\gemma_chat.cpp src\model\gemma_vision.cu src\model\vision_ops_cuda.cu src\io\image.cpp

"%CUDA%\bin\nvcc.exe" %NVFLAGS% -c tools\qwen_generate.cu -Dmain=dara_qwen_main -o build\cuda\dara_obj\qwen_driver.obj || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% -c tools\nano_generate.cu -Dmain=dara_nano_main -o build\cuda\dara_obj\nano_driver.obj || exit /b 1
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tools\dara.cu build\cuda\dara_obj\qwen_driver.obj build\cuda\dara_obj\nano_driver.obj src\model\flash_drafter.cu src\tokenizer\unicode.cpp %Q% %G% %M% %C% %K% -o build\cuda\dara.exe -cudart static || exit /b 1
echo built build\cuda\dara.exe
