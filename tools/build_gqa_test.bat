@echo off
rem Builds the grouped-query decode attention test (bitwise A/B against the per-head kernels + speed). DARA_ARCH=all for every consumer architecture.
setlocal
set CUDA=D:\tools\cuda\13.1.1
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not exist build\cuda mkdir build\cuda
if not defined DARA_ARCH set DARA_ARCH=75
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
set NVFLAGS=-std=c++20 -O3 %GENCODE% -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\attn_gqa_test.cu src\model\attn_splitkv.cu src\model\tq_kv.cu -o build\cuda\attn_gqa_test.exe -lcudart || exit /b 1
echo built build\cuda\attn_gqa_test.exe
