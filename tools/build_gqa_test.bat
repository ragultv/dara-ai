@echo off
rem Builds the grouped-query decode attention test (bitwise A/B against the per-head kernels + speed) -> build\cuda\attn_gqa_test.exe
rem   DARA_CUDA  CUDA toolkit folder (default D:\tools\cuda\13.1.1; tools\install_cuda_redist.py installs one without admin rights)
rem   DARA_ARCH  75 = GTX 16xx / RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx, 120 = RTX 50xx (default 75 here; "all" builds every consumer architecture)
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
if not exist build\cuda mkdir build\cuda
if not defined DARA_ARCH set DARA_ARCH=75
set GENCODE=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120
if /i not "%DARA_ARCH%"=="all" set GENCODE=-arch=sm_%DARA_ARCH%
set NVFLAGS=-std=c++20 -O3 %GENCODE% -lineinfo -I"%CUDA%\include" -Isrc -Ithird_party -Itests -L"%CUDA%\lib\x64"
"%CUDA%\bin\nvcc.exe" %NVFLAGS% tests\attn_gqa_test.cu src\model\attn_splitkv.cu src\model\tq_kv.cu -o build\cuda\attn_gqa_test.exe -lcudart || exit /b 1
echo built build\cuda\attn_gqa_test.exe
