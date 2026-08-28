<# Create a recoverable snapshot of the runtime data before upgrades or cleanup. #>
param([string]$DestinationRoot = '')

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$LockFile = Join-Path $ProjectDir 'downloads\.maintenance.lock'
if (Test-Path -LiteralPath $LockFile) {
    throw "Maintenance is active or left a lock file. Do not back up changing data: $LockFile"
}
if (-not $DestinationRoot) { $DestinationRoot = Join-Path $ProjectDir 'backups' }
New-Item -ItemType Directory -Force -Path $DestinationRoot | Out-Null
$DestinationRoot = (Resolve-Path -LiteralPath $DestinationRoot).Path
$Snapshot = Join-Path $DestinationRoot (Get-Date -Format 'yyyyMMdd_HHmmss')
New-Item -ItemType Directory -Path $Snapshot | Out-Null

function Copy-Tree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source)) { return }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy.exe $Source $Destination /E /COPY:DAT /DCOPY:T /R:2 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Host
    if ($LASTEXITCODE -gt 7) { throw "Backup copy failed with exit code ${LASTEXITCODE}: $Source" }
}

$ChunkFile = Join-Path $ProjectDir 'bid_chunks.json'
if (Test-Path -LiteralPath $ChunkFile) {
    Copy-Item -LiteralPath $ChunkFile -Destination (Join-Path $Snapshot 'bid_chunks.json')
}
Copy-Tree -Source (Join-Path $ProjectDir 'downloads') -Destination (Join-Path $Snapshot 'downloads')
Copy-Tree -Source (Join-Path $ProjectDir 'chroma_db') -Destination (Join-Path $Snapshot 'chroma_db')
Write-Host "Backup completed: $Snapshot"
