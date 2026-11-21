# Puts the engine where the installer bundles it from: app\src-tauri\resources\engine\dara.exe and \flash\*.dflash.
#   tools\stage_engine.ps1                              uses build\cuda\dara.exe and build\flash\*.dflash
#   tools\stage_engine.ps1 -Engine X\dara.exe -Flash X\flash
# A CUDA runtime DLL next to the engine is copied too (a local build links it dynamically; the release build links it statically).
param(
  [string]$Engine = "build\cuda\dara.exe",
  [string]$Flash = "build\flash",
  [string]$Dest = "app\src-tauri\resources\engine"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path $Engine)) { throw "Engine not found: $Engine (build it with tools\build_dara.bat)" }
New-Item -ItemType Directory -Force -Path $Dest, "$Dest\flash" | Out-Null
Get-ChildItem $Dest -Recurse -File | Where-Object { $_.Name -ne "README.txt" } | Remove-Item -Force

Copy-Item $Engine "$Dest\dara.exe"
Get-ChildItem (Split-Path -Parent $Engine) -Filter "cudart*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $Dest
if (Test-Path $Flash) {  Get-ChildItem $Flash -Filter "*_v2.dflash" | Copy-Item -Destination "$Dest\flash" }

$files = Get-ChildItem $Dest -Recurse -File | Where-Object { $_.Name -ne "README.txt" }
$files | ForEach-Object { "{0,8:N1} MB  {1}" -f ($_.Length / 1MB), $_.FullName.Substring((Resolve-Path $Dest).Path.Length + 1) }
if (-not (Test-Path "$Dest\flash\*.dflash")) { Write-Warning "No Flash drafters staged: Flash will be offered as a download instead." }
