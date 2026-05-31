$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed or not on PATH."
    Write-Host "Run scripts\install_docker_desktop.ps1, then restart PowerShell."
    exit 1
}

docker compose version | Out-Host
docker compose up -d

Write-Host ""
Write-Host "Elasticsearch: http://127.0.0.1:9200"
Write-Host "Neo4j Browser:  http://127.0.0.1:7474"
Write-Host "Neo4j login:    neo4j / ragsearch123"
