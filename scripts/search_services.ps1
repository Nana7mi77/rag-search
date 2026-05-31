param(
    [string]$Query = "自然光源",
    [int]$TopK = 5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$Python = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (!(Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

& $Python -m rag_search search-es $Query --top-k $TopK
