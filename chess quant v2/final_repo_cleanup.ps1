param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$DeleteBackup,
    [switch]$DeleteCleanupScripts
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Info([string]$Message) {
    Write-Host "[INFO] $Message"
}

function Ok([string]$Message) {
    Write-Host "[OK] $Message"
}

function Warn([string]$Message) {
    Write-Host "[WARN] $Message"
}

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Remove-IfExists([string]$Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
}

function Remove-IfEmpty([string]$Path) {
    if (-not (Test-Path $Path)) { return }
    $item = Get-Item $Path
    if (-not $item.PSIsContainer) { return }

    $children = Get-ChildItem -Force $Path -ErrorAction SilentlyContinue
    if ($null -eq $children -or $children.Count -eq 0) {
        Remove-Item -Force $Path
    }
}

function Merge-MoveTree([string]$Source, [string]$Destination) {
    if ([string]::IsNullOrWhiteSpace($Source)) { return }
    if (-not (Test-Path $Source)) { return }

    $srcFull = (Resolve-Path $Source).Path
    Ensure-Dir $Destination
    $dstFull = (Resolve-Path $Destination).Path

    if ($srcFull -eq $dstFull) { return }

    Info "Merging $srcFull -> $dstFull"
    robocopy $srcFull $dstFull /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache .ruff_cache .git | Out-Null

    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed: $srcFull -> $dstFull"
    }

    Remove-Item -Recurse -Force $srcFull
}

function Move-FileIfExists([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) { return }
    Ensure-Dir (Split-Path -Parent $Destination)
    Move-Item -Force $Source $Destination
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot

Info "Working in $RepoRoot"

# Final target structure
$TargetDirs = @(
    "artifacts\training\models",
    "artifacts\training\reports",
    "artifacts\training\figures",
    "artifacts\training\metrics",
    "artifacts\training\logs",
    "artifacts\deployment\web",
    "artifacts\deployment\runtime",
    "artifacts\deployment\handoff",
    "artifacts\deployment\model_bundles",
    "data\samples",
    "docs",
    "src\chessquant_ml\data"
)

foreach ($dir in $TargetDirs) {
    Ensure-Dir (Join-Path $RepoRoot $dir)
}

# 1. Reorganize artifacts into training vs deployment
Info "Reorganizing artifacts into training vs deployment"

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\models") `
    (Join-Path $RepoRoot "artifacts\training\models")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\reports") `
    (Join-Path $RepoRoot "artifacts\training\reports")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\figures") `
    (Join-Path $RepoRoot "artifacts\training\figures")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\metrics") `
    (Join-Path $RepoRoot "artifacts\training\metrics")

Merge-MoveTree `
    (Join-Path $RepoRoot "data\logs") `
    (Join-Path $RepoRoot "artifacts\training\logs")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\exports\web") `
    (Join-Path $RepoRoot "artifacts\deployment\web")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\exports\runtime") `
    (Join-Path $RepoRoot "artifacts\deployment\runtime")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\exports\production_handoff") `
    (Join-Path $RepoRoot "artifacts\deployment\handoff")

Merge-MoveTree `
    (Join-Path $RepoRoot "artifacts\exports\models") `
    (Join-Path $RepoRoot "artifacts\deployment\model_bundles")

Remove-IfEmpty (Join-Path $RepoRoot "artifacts\exports")

# 2. Fix suspicious src\data folder
Info "Checking src\data"

$srcData = Join-Path $RepoRoot "src\data"
$pkgData = Join-Path $RepoRoot "src\chessquant_ml\data"
$sampleDump = Join-Path $RepoRoot "data\samples\src_data_legacy"

if (Test-Path $srcData) {
    $pythonFiles = Get-ChildItem $srcData -Recurse -File -Include *.py -ErrorAction SilentlyContinue
    if ($pythonFiles -and $pythonFiles.Count -gt 0) {
        Warn "src\data contains Python files. Merging them into src\chessquant_ml\data"
        Merge-MoveTree $srcData $pkgData
    } else {
        Warn "src\data does not look like package code. Moving it to data\samples\src_data_legacy"
        Merge-MoveTree $srcData $sampleDump
    }
}

# 3. Remove stale empty dirs
Info "Removing empty leftover directories"
$maybeEmpty = @(
    (Join-Path $RepoRoot "artifacts\models"),
    (Join-Path $RepoRoot "artifacts\reports"),
    (Join-Path $RepoRoot "artifacts\figures"),
    (Join-Path $RepoRoot "artifacts\metrics"),
    (Join-Path $RepoRoot "artifacts\exports"),
    (Join-Path $RepoRoot "data\logs"),
    (Join-Path $RepoRoot "src\data")
)

foreach ($path in $maybeEmpty) {
    Remove-IfEmpty $path
}

# 4. Write final structure docs
Info "Updating docs"

$startHere = @"
# START HERE

This is the single ChessQuant repository.

## Main folders
- configs/   all editable configuration
- data/      canonical datasets and samples
- artifacts/ generated outputs
- mlflow/    experiment tracking internals
- docs/      project documentation
- notebooks/ exploratory work
- scripts/   CLI entrypoints
- src/       source code
- tests/     tests
- tools/     external tools

## Artifact split
- artifacts/training/   model runs, reports, figures, logs, metrics
- artifacts/deployment/ ONNX/runtime/web/handoff outputs
"@

$repoLayout = @"
# REPO LAYOUT

## configs/
- datasets/
- experiments/
- features/
- policies/
- serving/

## data/
- raw/
- enriched/
- features/
- cache/
- registry/
- samples/

## artifacts/training/
- models/
- reports/
- figures/
- metrics/
- logs/

## artifacts/deployment/
- web/
- runtime/
- handoff/
- model_bundles/

## mlflow/
- db/
- runs/
- artifacts/
"@

$enc = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "docs\START_HERE.md"), $startHere.Trim(), $enc)
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "docs\REPO_LAYOUT.md"), $repoLayout.Trim(), $enc)

# 5. Optional cleanup of backup and helper scripts
if ($DeleteBackup) {
    Info "Deleting _pre_unify_backup"
    Remove-IfExists (Join-Path $RepoRoot "_pre_unify_backup")
}

if ($DeleteCleanupScripts) {
    Info "Deleting old cleanup scripts"
    Remove-IfExists (Join-Path $RepoRoot "cleanup_and_restructure.ps1")
    Remove-IfExists (Join-Path $RepoRoot "cleanup_from_current_state.ps1")
}

# 6. Final summary
Write-Host ""
Ok "Final structural cleanup complete"
Write-Host ""
Write-Host "Recommended root layout now:"
Write-Host "  configs/"
Write-Host "  data/"
Write-Host "  artifacts/training/"
Write-Host "  artifacts/deployment/"
Write-Host "  mlflow/"
Write-Host "  docs/"
Write-Host "  notebooks/"
Write-Host "  scripts/"
Write-Host "  src/"
Write-Host "  tests/"
Write-Host "  tools/"
Write-Host ""
Write-Host "Recommended checks:"
Write-Host "  tree"
Write-Host "  poetry check"
Write-Host "  poetry install"