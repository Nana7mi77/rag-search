param(
    [string]$DataPath = "",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

function Test-Python {
    param([string]$Exe, [string[]]$Args = @())
    try {
        & $Exe @Args -c "import sys; sys.exit(0)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

function Resolve-Python {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if ((Test-Path -LiteralPath $VenvPython) -and (Test-Python $VenvPython)) {
        return @{ Exe = $VenvPython; Args = @() }
    }

    $CodexPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if ((Test-Path -LiteralPath $CodexPython) -and (Test-Python $CodexPython)) {
        return @{ Exe = $CodexPython; Args = @() }
    }

    if (Test-Python "py" @("-3")) {
        return @{ Exe = "py"; Args = @("-3") }
    }

    if (Test-Python "python") {
        return @{ Exe = "python"; Args = @() }
    }

    throw "No working Python found. Install Python 3.9+ or run inside Codex Desktop."
}

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath = Join-Path $ProjectRoot "data\local_subtitles.csv"
}

$PythonInfo = Resolve-Python

if (!(Test-Path -LiteralPath $DataPath)) {
    powershell -ExecutionPolicy Bypass -File scripts\export_legacy_data.ps1 -Out $DataPath -Limit $Limit
}

if ($Limit -gt 0) {
    & $PythonInfo.Exe @($PythonInfo.Args) -m rag_search build --data $DataPath --graph data\sample_kg.csv --index data\index.json --limit $Limit
} else {
    & $PythonInfo.Exe @($PythonInfo.Args) -m rag_search build --data $DataPath --graph data\sample_kg.csv --index data\index.json
}

Write-Host ""
Write-Host "Install complete. Start Web demo:"
Write-Host "  scripts\start_demo.bat"
