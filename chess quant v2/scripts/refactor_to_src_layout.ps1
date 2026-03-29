$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force src | Out-Null
New-Item -ItemType Directory -Force configs, outputs, scripts, notebooks, tests | Out-Null
New-Item -ItemType Directory -Force configs\datasets, configs\experiments, configs\policies | Out-Null
New-Item -ItemType Directory -Force outputs\figures, outputs\metrics, outputs\models, outputs\reports, outputs\web_exports | Out-Null
New-Item -ItemType Directory -Force data\cache | Out-Null

if ((Test-Path chessquant_ml) -and -not (Test-Path src\chessquant_ml)) {
  Move-Item chessquant_ml src\chessquant_ml
}

if (Test-Path experiments\configs) {
  Copy-Item experiments\configs\* configs\experiments -Recurse -Force
}

Write-Host "Refactor skeleton created."
Write-Host "Next manual step: update pyproject.toml packages = [{ include = 'chessquant_ml', from = 'src' }]"
