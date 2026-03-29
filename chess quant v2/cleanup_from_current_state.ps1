param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$RemoveBackupFolder
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Info($msg) { Write-Host "[INFO] $msg" }
function Ok($msg) { Write-Host "[OK] $msg" }
function Warn($msg) { Write-Host "[WARN] $msg" }

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Write-Utf8NoBom([string]$Path, [string]$Content) {
    $enc = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $enc)
}

function Merge-MoveTree([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) {
        return
    }

    $srcFull = (Resolve-Path $Source).Path
    Ensure-Dir $Destination
    $dstFull = (Resolve-Path $Destination).Path

    if ($srcFull -eq $dstFull) {
        return
    }

    Info "Merging $srcFull -> $dstFull"
    robocopy $srcFull $dstFull /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache .ruff_cache .git | Out-Null

    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed: $srcFull -> $dstFull"
    }

    Remove-Item -Recurse -Force $srcFull
}

function Move-FileIfExists([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) {
        return
    }

    Ensure-Dir (Split-Path -Parent $Destination)
    Move-Item -Force $Source $Destination
}

function Copy-FileIfExists([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) {
        return
    }

    Ensure-Dir (Split-Path -Parent $Destination)
    Copy-Item -Force $Source $Destination
}

function Remove-IfExists([string]$Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
}

function Remove-IfEmpty([string]$Path) {
    if (-not (Test-Path $Path)) {
        return
    }

    $item = Get-Item $Path
    if (-not $item.PSIsContainer) {
        return
    }

    $children = Get-ChildItem -Force $Path -ErrorAction SilentlyContinue
    if ($null -eq $children -or $children.Count -eq 0) {
        Remove-Item -Force $Path
    }
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot

Info "Working in: $RepoRoot"

# Target structure
$TargetDirs = @(
    "configs",
    "configs\datasets",
    "configs\experiments",
    "configs\features",
    "configs\policies",
    "configs\serving",
    "data",
    "data\raw",
    "data\enriched",
    "data\features",
    "data\cache",
    "data\logs",
    "data\registry",
    "data\registry\snapshots",
    "data\samples",
    "artifacts",
    "artifacts\models",
    "artifacts\reports",
    "artifacts\figures",
    "artifacts\exports",
    "artifacts\exports\models",
    "artifacts\exports\runtime",
    "artifacts\exports\web",
    "artifacts\exports\production_handoff",
    "artifacts\metrics",
    "mlflow",
    "mlflow\db",
    "mlflow\runs",
    "mlflow\artifacts",
    "docs",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "tools",
    "tools\stockfish"
)

foreach ($dir in $TargetDirs) {
    Ensure-Dir (Join-Path $RepoRoot $dir)
}

# 1. config -> configs/features
Info "Moving feature schema"
Move-FileIfExists `
    (Join-Path $RepoRoot "config\feature_schema.json") `
    (Join-Path $RepoRoot "configs\features\feature_schema.json")

# 2. data cleanup
Info "Reorganizing data"
Merge-MoveTree (Join-Path $RepoRoot "data\snapshots") (Join-Path $RepoRoot "data\registry\snapshots")

# 3. outputs -> artifacts
Info "Moving outputs into artifacts"
Merge-MoveTree (Join-Path $RepoRoot "outputs\models") (Join-Path $RepoRoot "artifacts\models")
Merge-MoveTree (Join-Path $RepoRoot "outputs\reports") (Join-Path $RepoRoot "artifacts\reports")
Merge-MoveTree (Join-Path $RepoRoot "outputs\figures") (Join-Path $RepoRoot "artifacts\figures")
Merge-MoveTree (Join-Path $RepoRoot "outputs\metrics") (Join-Path $RepoRoot "artifacts\metrics")
Merge-MoveTree (Join-Path $RepoRoot "outputs\web_exports") (Join-Path $RepoRoot "artifacts\exports\web")

# 4. legacy runtime/export files out of data
Info "Moving runtime/export artifacts out of data"
Merge-MoveTree (Join-Path $RepoRoot "data\models") (Join-Path $RepoRoot "artifacts\exports\models")
Merge-MoveTree (Join-Path $RepoRoot "data\artifacts") (Join-Path $RepoRoot "artifacts\exports\runtime")

# 5. mlflow regrouping
Info "Reorganizing MLflow files"
Move-FileIfExists (Join-Path $RepoRoot "mlflow.db") (Join-Path $RepoRoot "mlflow\db\mlflow.db")
Merge-MoveTree (Join-Path $RepoRoot "mlruns") (Join-Path $RepoRoot "mlflow\runs")
Merge-MoveTree (Join-Path $RepoRoot "mlartifacts") (Join-Path $RepoRoot "mlflow\artifacts")

# 6. stockfish wrapper cleanup
Info "Reorganizing Stockfish"
$StockfishWrapped = Join-Path $RepoRoot "tools\stockfish-windows-x86-64-avx2\stockfish"
$StockfishWrappedAlt = Join-Path $RepoRoot "tools\stockfish-windows-x86-64-avx2"

if (Test-Path $StockfishWrapped) {
    Merge-MoveTree $StockfishWrapped (Join-Path $RepoRoot "tools\stockfish")
    Remove-IfEmpty (Join-Path $RepoRoot "tools\stockfish-windows-x86-64-avx2")
} elseif (Test-Path $StockfishWrappedAlt) {
    Merge-MoveTree $StockfishWrappedAlt (Join-Path $RepoRoot "tools\stockfish")
}

# 7. production handoff relocation if present
Info "Moving production handoff if present"
Merge-MoveTree (Join-Path $RepoRoot "production_handoff") (Join-Path $RepoRoot "artifacts\exports\production_handoff")

# 8. docs
Info "Writing docs"
$StartHere = @"
# START HERE

This is the single ChessQuant repository.

## Main folders
- configs/   editable YAML and schema files
- data/      canonical raw, enriched, and feature datasets
- artifacts/ generated models, reports, figures, exports
- mlflow/    experiment tracking database, runs, and artifacts
- scripts/   CLI entrypoints
- src/       project source code
- tests/     automated tests
- tools/     external tools like Stockfish

## Notes
- research and production live in one repo
- runtime/export outputs are in artifacts/exports
- train/eval outputs are in artifacts/models, artifacts/reports, artifacts/figures
"@

$RepoLayout = @"
# REPO LAYOUT

## configs/
- datasets/     dataset and split configs
- experiments/  model training configs
- features/     schema and feature-related config
- policies/     stop/continue threshold configs
- serving/      runtime serving configs

## data/
- raw/          raw downloaded games
- enriched/     engine-enriched games
- features/     parquet feature tables
- cache/        temporary caches
- logs/         data pipeline logs
- registry/     registries and snapshots
- samples/      small sample files

## artifacts/
- models/       train run outputs
- reports/      policy reports and analysis reports
- figures/      generated plots
- exports/      ONNX, manifests, handoff bundles
- metrics/      summary csv/json outputs

## mlflow/
- db/           MLflow sqlite database
- runs/         MLflow metadata store
- artifacts/    MLflow artifact store
"@

Write-Utf8NoBom (Join-Path $RepoRoot "docs\START_HERE.md") $StartHere.Trim()
Write-Utf8NoBom (Join-Path $RepoRoot "docs\REPO_LAYOUT.md") $RepoLayout.Trim()

# 9. update common path strings in YAMLs only
Info "Updating config paths"
$yamlFiles = Get-ChildItem (Join-Path $RepoRoot "configs") -Recurse -Include *.yaml,*.yml -File -ErrorAction SilentlyContinue
foreach ($yaml in $yamlFiles) {
    $content = Get-Content -Raw -Encoding UTF8 $yaml.FullName

    $content = $content.Replace("config/feature_schema.json", "configs/features/feature_schema.json")
    $content = $content.Replace("config\feature_schema.json", "configs/features/feature_schema.json")

    $content = $content.Replace("outputs/models", "artifacts/models")
    $content = $content.Replace("outputs/reports", "artifacts/reports")
    $content = $content.Replace("outputs/figures", "artifacts/figures")
    $content = $content.Replace("outputs/metrics", "artifacts/metrics")
    $content = $content.Replace("outputs/web_exports", "artifacts/exports/web")

    Write-Utf8NoBom $yaml.FullName $content
}

# 10. rebuild latest pointer in artifacts\models
Info "Rebuilding latest model pointer"
$ModelRoot = Join-Path $RepoRoot "artifacts\models"
$modelRuns = @(
    Get-ChildItem $ModelRoot -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^\d{8}_\d{6}_.+' } |
    Sort-Object Name
)

if ($modelRuns.Count -gt 0) {
    Write-Utf8NoBom (Join-Path $ModelRoot "latest") $modelRuns[-1].FullName
}

# 11. remove old now-empty containers
Info "Removing obsolete directories if empty"
$MaybeEmpty = @(
    (Join-Path $RepoRoot "config"),
    (Join-Path $RepoRoot "outputs"),
    (Join-Path $RepoRoot "data\artifacts"),
    (Join-Path $RepoRoot "data\models"),
    (Join-Path $RepoRoot "data\snapshots"),
    (Join-Path $RepoRoot "mlruns"),
    (Join-Path $RepoRoot "mlartifacts"),
    (Join-Path $RepoRoot "tools\stockfish-windows-x86-64-avx2")
)

foreach ($path in $MaybeEmpty) {
    Remove-IfEmpty $path
}

# 12. optional backup cleanup
if ($RemoveBackupFolder) {
    Info "Removing _pre_unify_backup"
    Remove-IfExists (Join-Path $RepoRoot "_pre_unify_backup")
}

Ok "Current-state restructure complete"
Write-Host ""
Write-Host "New top-level intent:"
Write-Host "  configs/   configuration"
Write-Host "  data/      canonical datasets"
Write-Host "  artifacts/ generated outputs"
Write-Host "  mlflow/    tracking internals"
Write-Host "  docs/      documentation"
Write-Host "  scripts/   CLI entrypoints"
Write-Host "  src/       source code"
Write-Host "  tests/     tests"
Write-Host "  tools/     external tools"
Write-Host ""
Write-Host "Recommended next checks:"
Write-Host "  tree /f"
Write-Host "  poetry check"
Write-Host "  poetry install"