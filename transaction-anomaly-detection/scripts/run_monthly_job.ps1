# Monthly anomaly detection job — schedule via Windows Task Scheduler
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

python -m src.main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Anomaly detection job completed. Check output/ folder."
