<# Run the initial download, chunking, embedding, boilerplate, and lexical build. #>
param(
    [switch]$SkipDownload,
    [switch]$ResetIndex,
    [switch]$ForceRechunk,
    [switch]$SkipBoilerplate,
    [int]$BatchSize = 64
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run .\setup_new_laptop.ps1 first.' }

[string[]]$arguments = @('.\portal_pipeline.py', 'initialise', '--batch-size', "$BatchSize")
if ($SkipDownload) { $arguments += '--skip-download' }
if ($ResetIndex) { $arguments += '--reset-index' }
if ($ForceRechunk) { $arguments += '--force-rechunk' }
if ($SkipBoilerplate) { $arguments += '--skip-boilerplate' }

Push-Location $ProjectDir
try { & $Python @arguments } finally { Pop-Location }
