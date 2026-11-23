# Adds or removes one folder in the current user's PATH (HKCU\Environment), for the Dara installer.
#   dara-path.ps1 add <folder>      makes `dara` work from any new terminal
#   dara-path.ps1 remove <folder>   undoes that when Dara is uninstalled
# It edits the registry value directly so the other entries keep their %VARIABLES% (the value stays an expandable string) and so a long
# PATH is never cut short.
param([Parameter(Mandatory)][ValidateSet('add', 'remove')][string]$Action, [Parameter(Mandatory)][string]$Dir)
$ErrorActionPreference = 'Stop'
$Dir = $Dir.TrimEnd('\')
$key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
try {
    $old = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
    $kept = @($old -split ';' | Where-Object { $_ -ne '' -and $_.TrimEnd('\') -ne $Dir })
    if ($Action -eq 'add') { $kept += $Dir }
    $new = $kept -join ';'
    if ($new -ne $old) { $key.SetValue('Path', $new, [Microsoft.Win32.RegistryValueKind]::ExpandString) }
} finally {
    $key.Close()
}
# tell running programs (Explorer, so new terminals pick it up) that the environment changed
Add-Type -Namespace Dara -Name Win -MemberDefinition '[DllImport("user32.dll", CharSet = CharSet.Auto)] public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, UIntPtr w, string l, uint f, uint t, out UIntPtr r);'
$r = [UIntPtr]::Zero
[void][Dara.Win]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$r)
