# Releasing Dara (Windows installer)

What users get: one file, `Dara_<version>_x64-setup.exe` (about 125 MB). It installs the app and its engine; **models are not in it**, they are downloaded inside the app (several GB each).

## What the installer contains

```
<install folder>\dara.exe                      the desktop app
<install folder>\engine\dara.exe               the engine (CUDA runtime linked in)
<install folder>\engine\flash\*_v2.dflash      Flash drafters for Core 4B and Nano
<install folder>\uninstall.exe
```

- **Per user**: installs to `%LOCALAPPDATA%\Dara` by default, no administrator prompt. The installer lets the user pick another folder; `Dara_..._setup.exe /S /D=<folder>` installs silently there.
- **Models folder**: default `%LOCALAPPDATA%\Dara\models`; change it in Settings > Engine. There is no welcome screen: the app opens on Chat, and models are downloaded from the Models page. The folder is kept when the app is uninstalled.
- The app finds the engine in `engine\` next to itself (`installed_engine` in `app/src-tauri/src/sidecar.rs`); a development run still finds `build\cuda\dara.exe`.
- Needs an NVIDIA GPU from Turing (GTX 16xx / RTX 20xx) on and a driver with CUDA 13 (R580+). The Models page says plainly when the engine did not start, there is no NVIDIA card, or it is older than Turing (`startupCheck.ts`).

## Building it

Locally (about 6 minutes the first time):

```
tools\build_dara.bat                 # the engine, build\cuda\dara.exe
app\tools\build_installer.bat        # stages the engine (tools\stage_engine.ps1), then the Tauri build
```

Set `CARGO_TARGET_DIR` to build elsewhere and leave a running `tauri dev` alone. The installer lands in `<target>\release\bundle\nsis\`.

On GitHub: `.github/workflows/build-release.yml` has two jobs. `windows-cuda` builds the engine (all supported architectures) and keeps `dara.exe` and the drafters as an artifact; `installer` builds the app from them (tests first) and uploads the installer. A push to `main` (or a manual run with "release" ticked) publishes release `b<run number>` with the engine zip and the installer, each with a `.sha256`. A manual run can also tick "sign" to sign the installer without publishing (SignPath's test policy).

## Checked

Built the installer, installed it silently into a chosen folder, started the installed app: it launched `engine\dara.exe` and `/health` answered 200; the uninstaller left no registry entry or files, and the models folder stayed.

## Not done yet

- **Code signing.** The installer is still unsigned, so Windows SmartScreen warns on download. The workflow is prepared to sign it through SignPath Foundation's free programme for open-source projects (it signs once the repository variable `SIGNPATH_ORGANIZATION_ID` is set; nothing changes before). What is left is applying to SignPath and setting the secret and variables: [CODE_SIGNING.md](CODE_SIGNING.md). The signing step itself has not run yet.
- **Auto-update.** Users reinstall to update. Tauri's updater plugin needs a signing key and a hosted update manifest.
- **The release workflow's new job has not run on GitHub yet.** First run: use "Run workflow" without ticking "release", and check the `dara-installer` artifact.
- **Clean-machine test.** Tried on the development machine only. Try a Windows install that never had Dara, WebView2 absent (the installer downloads it), and a machine without an NVIDIA card.
- **Version number** is 0.1.0 in `tauri.conf.json` and the app's `package.json`; bump both per release.
