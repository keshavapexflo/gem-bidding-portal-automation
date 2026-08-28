<# Install Python dependencies and optionally import an existing Phase 1 dataset. #>
param(
    [string]$DataSource = ''
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $PSCommandPath
$VenvPython = Join-Path $ProjectDir '.venv\Scripts\python.exe'

function Test-Python311 {
    param([string[]]$Command)
    $executable = $Command[0]
    $pythonArgs = if ($Command.Length -gt 1) { $Command[1..($Command.Length - 1)] } else { @() }
    $version = & $executable @pythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    return $LASTEXITCODE -eq 0 -and "$version".Trim() -eq '3.11'
}

function Copy-DataDirectory {
    param([string]$Source, [string]$Destination)
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy.exe $Source $Destination /E /COPY:DAT /DCOPY:T /R:2 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Host
    if ($LASTEXITCODE -gt 7) {
        throw "Data copy failed with robocopy exit code ${LASTEXITCODE}: $Source"
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCommand = @('py', '-3.11')
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCommand = @('python')
} else {
    throw 'Python 3.11 was not found. Install 64-bit Python 3.11, then rerun setup.'
}
if (-not (Test-Python311 $PythonCommand)) {
    throw 'Python 3.11 is required. Install it from python.org, including the Python Launcher.'
}

Push-Location $ProjectDir
try {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host 'Creating the Python 3.11 virtual environment...'
        $executable = $PythonCommand[0]
        $pythonArgs = if ($PythonCommand.Length -gt 1) { $PythonCommand[1..($PythonCommand.Length - 1)] } else { @() }
        & $executable @pythonArgs -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the Python 3.11 virtual environment.' }
    }
    Write-Host 'Installing pinned dependencies...'
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    Write-Host 'Downloading and caching the pinned BGE query model...'
    & $VenvPython -c "from sentence_transformers import SentenceTransformer; from create_embeddings import DEFAULT_MODEL, DEFAULT_MODEL_REVISION; SentenceTransformer(DEFAULT_MODEL, revision=DEFAULT_MODEL_REVISION); print('Model cache ready')"

    foreach ($folder in @('downloads\bids', 'downloads\expired', 'downloads\logs', 'backups', 'chroma_db', 'static')) {
        New-Item -ItemType Directory -Force -Path (Join-Path $ProjectDir $folder) | Out-Null
    }

    if ($DataSource) {
        $SourceRoot = (Resolve-Path -LiteralPath $DataSource).Path
        $required = @('bid_chunks.json', 'downloads', 'chroma_db')
        $missing = @($required | Where-Object { -not (Test-Path -LiteralPath (Join-Path $SourceRoot $_)) })
        if ($missing.Count -gt 0) {
            throw "Data source is incomplete. Missing: $($missing -join ', ')"
        }
        Write-Host "Importing runtime data from $SourceRoot ..."
        Copy-Item -LiteralPath (Join-Path $SourceRoot 'bid_chunks.json') -Destination (Join-Path $ProjectDir 'bid_chunks.json') -Force
        Copy-DataDirectory -Source (Join-Path $SourceRoot 'downloads') -Destination (Join-Path $ProjectDir 'downloads')
        Copy-DataDirectory -Source (Join-Path $SourceRoot 'chroma_db') -Destination (Join-Path $ProjectDir 'chroma_db')
    }

    $allowEmpty = if (Test-Path -LiteralPath (Join-Path $ProjectDir 'bid_chunks.json')) { @() } else { @('--allow-empty') }
    & $VenvPython .\validate_installation.py @allowEmpty
    if ($LASTEXITCODE -ne 0) {
        throw 'Setup completed, but deployment validation failed. Review the messages above.'
    }
    Write-Host 'Setup complete. Run .\start_portal.ps1 or .\initialize_phase1.ps1.'
} finally {
    Pop-Location
}
