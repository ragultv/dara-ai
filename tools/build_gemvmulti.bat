@echo off
rem Builds the multi-vector GEMV test (single architecture by default).
setlocal
set CUDA=D:\tools\cuda\13.1.1
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not defined DARA_ARCH set DARA_ARCH=75
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
set NVFLAGS=-std=c++20 %GENCODE% -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
set K=src\kernels\gemv.cu src\kernels\gemv_cpu.cpp src\kernels\q4k_gemv.cu src\kernels\q4k_cpu.cpp src\kernels\q5k_gemv.cu src\kernels\q80_gemv.cu src\kernels\q5k_q80_cpu.cpp src\kernels\q6k_gemv.cu src\kernels\q6k_cpu.cpp src\kernels\quant_gemv.cu src\kernels\gemv_multi.cu src\kernels\gemv16.cu src\kernels\quant.cpp
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\gemv_multi_test.cu %K% -o build\cuda\gemv_multi_test.exe -lcudart || exit /b 1
echo built build\cuda\gemv_multi_test.exe
