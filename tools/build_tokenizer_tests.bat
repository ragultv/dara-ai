@echo off
rem Builds the Gemma tokenizer test with MSVC (no CUDA needed).
setlocal
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cd /d %~dp0..
if not exist build\cuda mkdir build\cuda
cl /nologo /std:c++20 /EHsc /O2 /utf-8 /permissive- /Isrc /Ithird_party tests\gemma_tokenizer_test.cpp src\tokenizer\gemma_tokenizer.cpp src\tokenizer\unicode.cpp src\io\gguf.cpp /Fo:build\cuda\ /Fe:build\cuda\gemma_tokenizer_test.exe || exit /b 1
echo built build\cuda\gemma_tokenizer_test.exe
cl /nologo /std:c++20 /EHsc /O2 /utf-8 /permissive- /Isrc /Ithird_party tests\gemma_chat_test.cpp src\tokenizer\gemma_chat.cpp src\tokenizer\gemma_tokenizer.cpp src\tokenizer\unicode.cpp src\io\gguf.cpp /Fo:build\cuda\ /Fe:build\cuda\gemma_chat_test.exe || exit /b 1
echo built build\cuda\gemma_chat_test.exe
