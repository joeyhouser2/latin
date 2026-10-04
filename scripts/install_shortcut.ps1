<#
.SYNOPSIS
    Put a "Latin Library" shortcut on the Desktop (and optionally the Start Menu).

.DESCRIPTION
    Creates a .lnk pointing at "Latin Library.bat" in the repo root, so the web
    app starts from an icon rather than a terminal. Re-running overwrites the
    existing shortcut, which is how you move it after moving the repo.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -StartMenu
    powershell -ExecutionPolicy Bypass -File scripts\install_shortcut.ps1 -Remove
#>
param(
    [switch]$StartMenu,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repo   = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repo 'Latin Library.bat'
$name   = 'Latin Library.lnk'

$paths = @([System.IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), $name))
if ($StartMenu) {
    $programs = [Environment]::GetFolderPath('Programs')
    $paths += [System.IO.Path]::Combine($programs, $name)
}

if ($Remove) {
    foreach ($p in $paths) {
        if (Test-Path $p) { Remove-Item $p -Force; Write-Host "Removed $p" }
        else { Write-Host "Not there: $p" }
    }
    return
}

if (-not (Test-Path $target)) { throw "Cannot find $target - run this from a checkout of the repo." }

# The generated mark (scripts\make_logo.py writes every size Windows asks for,
# 16px for the taskbar up to 256px for large Desktop icons). Falls back to a
# shell icon if the .ico has not been generated yet.
$icon = Join-Path $repo 'web\static\latin-library.ico'
if (-not (Test-Path $icon)) { $icon = "$env:SystemRoot\System32\SHELL32.dll,13" }

$shell = New-Object -ComObject WScript.Shell
foreach ($p in $paths) {
    $lnk = $shell.CreateShortcut($p)
    $lnk.TargetPath       = $target
    $lnk.WorkingDirectory = $repo
    $lnk.IconLocation     = $icon
    $lnk.Description      = 'Browse the Latin/Greek corpus and queue translation jobs'
    $lnk.WindowStyle      = 7          # start minimised: the console is just a log
    $lnk.Save()
    Write-Host "Created $p"
}

Write-Host ''
Write-Host 'Double-click it to start the app; it opens http://127.0.0.1:8000 in your browser.'
