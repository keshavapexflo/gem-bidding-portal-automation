<# Run one incremental maintenance cycle interactively. #>
param(
    [switch]$ForceExpiry,
    [switch]$SkipExpiry,
    [switch]$ApplyExpiry
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run .\setup_new_laptop.ps1 first.' }
[string[]]$arguments = @('.\portal_pipeline.py', 'maintain')
if ($ForceExpiry) { $arguments += '--force-expiry' }
if ($SkipExpiry) { $arguments += '--skip-expiry' }
if ($ApplyExpiry) { $arguments += '--apply-expiry' }
Push-Location $ProjectDir
try { & $Python @arguments } finally { Pop-Location }
