<# Validate code, runtime data, Chroma, and lexical state. #>
param([switch]$Full, [switch]$AllowEmpty)
$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run .\setup_new_laptop.ps1 first.' }
$arguments = @((Join-Path $ProjectDir 'validate_installation.py'))
if ($Full) { $arguments += '--full' }
if ($AllowEmpty) { $arguments += '--allow-empty' }
& $Python @arguments
exit $LASTEXITCODE
