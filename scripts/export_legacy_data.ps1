param(
    [string]$Source = "E:\jupyter-notebook\wroks\algrothm\2023621.csv",
    [string]$Out = "",
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($Out)) {
    $Out = Join-Path $ProjectRoot "data\local_subtitles.csv"
}

if (!(Test-Path -LiteralPath $Source)) {
    throw "Legacy subtitle data not found: $Source"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Out) | Out-Null

$rows = Import-Csv -LiteralPath $Source -Encoding UTF8
if ($Limit -gt 0) {
    $rows = $rows | Select-Object -First $Limit
}
$rows | Export-Csv -LiteralPath $Out -NoTypeInformation -Encoding UTF8

Write-Host "Exported: $Out"
Write-Host "Rows:" ($rows | Measure-Object).Count
