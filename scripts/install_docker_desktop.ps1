$ErrorActionPreference = "Stop"

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker --version
    Write-Host "Docker is already installed."
    exit 0
}

if (!(Get-Command winget -ErrorAction SilentlyContinue)) {
    Write-Host "winget was not found. Install Docker Desktop manually:"
    Write-Host "https://docs.docker.com/desktop/setup/install/windows-install/"
    exit 1
}

Write-Host "Installing Docker Desktop with winget. You may need administrator permission."
winget install --id Docker.DockerDesktop -e --accept-package-agreements --accept-source-agreements
Write-Host "Install requested. Restart PowerShell after Docker Desktop finishes installing."
