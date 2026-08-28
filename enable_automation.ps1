<# Register daily incremental maintenance for the current Windows user. #>
param(
    [ValidatePattern('^([01]\d|2[0-3]):[0-5]\d$')]
    [string]$Time = '11:00',
    [switch]$ApplyExpiry
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
$TaskScript = Join-Path $ProjectDir 'scheduled_maintenance.ps1'
$TaskName = 'LetsBidDailyMaintenance'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run .\setup_new_laptop.ps1 first.' }
& $Python (Join-Path $ProjectDir 'validate_installation.py')
if ($LASTEXITCODE -ne 0) { throw 'Validation failed; automation was not enabled.' }

$expiryArgument = if ($ApplyExpiry) { ' -ApplyExpiry' } else { '' }
$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$TaskScript`"$expiryArgument"
schtasks.exe /Create /TN $TaskName /TR $taskCommand /SC DAILY /ST $Time /F | Out-Host
if ($LASTEXITCODE -ne 0) { throw "Task Scheduler registration failed with exit code $LASTEXITCODE" }
Write-Host "Scheduled $TaskName daily at $Time."
if (-not $ApplyExpiry) {
    Write-Host 'Expiry remains dry-run only. Re-enable with -ApplyExpiry after reviewing expiry reports.'
}
