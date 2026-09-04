<# Review and optionally archive bids that are no longer listed as ongoing on GeM. #>
param(
    [switch]$Apply
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Run .\setup_new_laptop.ps1 first.' }

Push-Location $ProjectDir
try {
    Write-Host 'Scanning every current GeM ongoing-bid page and preparing an expiry report...'
    & $Python .\gem_expiry_cleanup.py
    if ($LASTEXITCODE -ne 0) { throw 'The expiry report failed. No local data was changed.' }

    if (-not $Apply) {
        Write-Host ''
        Write-Host 'Dry run complete. Nothing was removed from search or moved on disk.'
        Write-Host 'After reviewing the counts, rerun with -Apply to archive the reported expired bids.'
        exit 0
    }

    Write-Warning 'Apply mode removes expired chunks/vectors from search and moves their PDFs into downloads\expired.'
    Write-Warning 'Stop the Streamlit portal before continuing. The PDFs are archived, not permanently deleted.'
    $confirmation = Read-Host 'Type APPLY to continue'
    if ($confirmation -cne 'APPLY') {
        Write-Host 'Cleanup cancelled. No local data was changed.'
        exit 0
    }

    # The apply command deliberately performs a fresh GeM scan. It refuses a
    # zero/implausibly-small response and blocks unexpectedly large removals.
    & $Python .\gem_expiry_cleanup.py --apply
    if ($LASTEXITCODE -ne 0) { throw 'Expiry cleanup failed. Review the error before trying again.' }

    Write-Host 'Rebuilding the lexical index after expiry cleanup...'
    & $Python .\gem_hybrid_retrieval.py --build-lexical-index --rebuild
    if ($LASTEXITCODE -ne 0) { throw 'Expiry cleanup succeeded, but lexical-index rebuilding failed.' }

    Write-Host 'Validating the updated searchable corpus...'
    & $Python .\validate_installation.py --full
    if ($LASTEXITCODE -ne 0) { throw 'Cleanup completed, but validation found a problem.' }

    Write-Host 'Weekly expiry cleanup complete. You can restart the portal.'
} finally {
    Pop-Location
}
