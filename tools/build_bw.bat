@echo off
setlocal
set CUDA=D:\tools\cuda\13.1.1
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not exist build\cuda mkdir build\cuda
"%CUDA%\bin\nvcc.exe" -std=c++20 -arch=sm_75 -I"%CUDA%\include" -L"%CUDA%\lib\x64" tests\bw_probe.cu -o build\cuda\bw_probe.exe -lcudart || exit /b 1
echo built build\cuda\bw_probe.exe
