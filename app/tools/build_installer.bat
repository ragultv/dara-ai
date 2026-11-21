@echo off
rem Builds the Windows installer: stages the engine, then runs the Tauri build. Output: app\src-tauri\target\release\bundle\nsis\Dara_*_x64-setup.exe
rem Set CARGO_TARGET_DIR first to build somewhere else (keeps a running "tauri dev" undisturbed).
setlocal
cd /d "%~dp0\..\.."
powershell -NoProfile -ExecutionPolicy Bypass -File tools\stage_engine.ps1 || exit /b 1
cd app
call npm run tauri -- build || exit /b 1
echo.
echo Installer:
dir /b "%CARGO_TARGET_DIR%\release\bundle\nsis\*.exe" 2>nul
dir /b src-tauri\target\release\bundle\nsis\*.exe 2>nul
