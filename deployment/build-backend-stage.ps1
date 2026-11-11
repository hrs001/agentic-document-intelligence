$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
$stageRoot = Join-Path $scriptRoot "dist\backend-src"
$packageRoot = Join-Path $stageRoot "agentic_document_intelligence"

if (Test-Path $stageRoot) {
    Remove-Item $stageRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $packageRoot | Out-Null
Copy-Item (Join-Path $projectRoot "__init__.py") $packageRoot
Copy-Item (Join-Path $projectRoot "backend") (Join-Path $packageRoot "backend") -Recurse
Copy-Item (Join-Path $projectRoot "scripts") (Join-Path $packageRoot "scripts") -Recurse
Copy-Item (Join-Path $scriptRoot "requirements-lambda.txt") (Join-Path $stageRoot "requirements.txt")

Write-Output "Backend stage ready at $stageRoot"
