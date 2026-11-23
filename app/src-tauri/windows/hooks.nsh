; Installer hooks: put the folder that holds dara.exe on the user's PATH, so `dara pull`, `dara serve` and `dara list`
; work from any terminal. The installer is per-user, so this edits only the current user's PATH (no administrator needed).
; Terminals that are already open keep the old PATH: open a new one after installing.
;
; The PATH edit is done by dara-path.ps1, not by NSIS itself: NSIS strings are cut at 1024 characters, which would truncate a long PATH.

; captured here: inside a macro, __FILEDIR__ would be the folder of the installer script that uses it
!define DARA_HOOK_DIR "${__FILEDIR__}"

!macro NSIS_HOOK_POSTINSTALL
  File "/oname=$INSTDIR\engine\dara-path.ps1" "${DARA_HOOK_DIR}\dara-path.ps1"
  nsExec::Exec '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\engine\dara-path.ps1" add "$INSTDIR\engine"'
  Pop $0
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  nsExec::Exec '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\engine\dara-path.ps1" remove "$INSTDIR\engine"'
  Pop $0
!macroend
