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

    throw "No working Python found."
}

if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker is not installed. Run scripts\install_docker_desktop.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    $DataPath = Join-Path $ProjectRoot "data\local_subtitles.csv"
}

if (!(Test-Path -LiteralPath $DataPath)) {
    powershell -ExecutionPolicy Bypass -File scripts\export_legacy_data.ps1 -Out $DataPath -Limit $Limit
}

$PythonInfo = Resolve-Python

Write-Host "Importing subtitles into Elasticsearch..."
if ($Limit -gt 0) {
    & $PythonInfo.Exe @($PythonInfo.Args) -m rag_search import-es --data $DataPath --limit $Limit
} else {
    & $PythonInfo.Exe @($PythonInfo.Args) -m rag_search import-es --data $DataPath
}

Write-Host ""
Write-Host "Importing sample knowledge graph into Neo4j..."
docker compose exec -T neo4j cypher-shell -u neo4j -p ragsearch123 -f /var/lib/neo4j/import/seed_kg.cypher

Write-Host ""
Write-Host "Service import complete."
Write-Host "Try:"
Write-Host "  python -m rag_search search-es `"自然光源`""
