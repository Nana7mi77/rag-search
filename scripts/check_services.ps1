$ErrorActionPreference = "Continue"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose ps
} else {
    Write-Host "Docker command not found."
}

Write-Host ""
Write-Host "Checking Elasticsearch..."
try {
    (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:9200" -TimeoutSec 5).Content
} catch {
    Write-Host $_.Exception.Message
}

Write-Host ""
Write-Host "Checking Neo4j HTTP..."
try {
    (Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:7474" -TimeoutSec 5).StatusCode
} catch {
    Write-Host $_.Exception.Message
}
