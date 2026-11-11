$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$stageRoot = Join-Path $scriptRoot "dist\frontend-site"

$frontendOut = Join-Path $projectRoot "frontend\out"
if (-not (Test-Path $frontendOut)) {
    throw "frontend/out was not found. Run 'powershell -ExecutionPolicy Bypass -Command ""cd frontend; npm.cmd run -s build""' first."
}

if (Test-Path $stageRoot) {
    Remove-Item $stageRoot -Recurse -Force
}

Copy-Item $frontendOut $stageRoot -Recurse
Write-Output "Frontend static site ready at $stageRoot"
