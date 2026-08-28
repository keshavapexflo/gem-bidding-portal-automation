<# Task Scheduler entry point. Always establishes the correct working directory. #>
param([switch]$ApplyExpiry)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
$LogDir = Join-Path $ProjectDir 'downloads\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir 'daily_maintenance.log'

Push-Location $ProjectDir
try {
    [string[]]$arguments = @('.\portal_pipeline.py', 'maintain')
    if ($ApplyExpiry) { $arguments += '--apply-expiry' }
    & $Python @arguments *>> $LogFile
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
