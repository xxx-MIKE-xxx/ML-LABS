param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot
)

$ErrorActionPreference = 'Stop'

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

if (-not (Test-Path $RepoRoot)) {
    throw "Repo root not found: $RepoRoot"
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot

$docsDir = Join-Path $RepoRoot 'docs'
Ensure-Dir $docsDir

# Move lab-specific docs into docs/
$docMoves = @(
    'MIGRATION_GUIDE_LAB.md',
    'README_LAB.md',
    'pyproject_patch.md'
)
foreach ($name in $docMoves) {
    $src = Join-Path $RepoRoot $name
    $dst = Join-Path $docsDir $name
    if ((Test-Path $src) -and -not (Test-Path $dst)) {
        Move-Item $src $dst
    }
}

# Remove temporary patch folders
$tempDirs = @('PATCH', 'PATCH_VIS', 'PATCH_POLICY_VIZ')
foreach ($dir in $tempDirs) {
    $path = Join-Path $RepoRoot $dir
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

# Remove stale non-package data mirror under src/data if present
$staleSrcData = @(
    'src\\data\\raw',
    'src\\data\\enriched',
    'src\\data\\features'
)
foreach ($rel in $staleSrcData) {
    $path = Join-Path $RepoRoot $rel
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
    }
}

# Remove Python caches and backup files
Get-ChildItem -Path $RepoRoot -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -Path $RepoRoot -Recurse -File -Include '*.pyc', '*.bak', '*.pre_patch_backup' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Write documentation
@'
# ChessQuant ML Labs - Repository Layout

This repository is the main ML/product bridge for ChessQuant.

## Top-level directories

- `config/` - stable schemas used by the production-style ML package
- `configs/` - experiment, dataset, and policy YAMLs for lab work
- `data/` - local raw/enriched/features/models/artifacts/cache storage
- `docs/` - migration notes, lab notes, and repository orientation docs
- `notebooks/` - optional ad-hoc exploration
- `outputs/` - experiment outputs, reports, figures, and model snapshots
- `production_handoff/` - files prepared for implementation / deployment teams
- `scripts/` - repeatable operational scripts
- `src/chessquant_ml/` - Python package source code
- `tests/` - tests for reusable logic such as splitters

## Source package structure

- `src/chessquant_ml/cli/` - CLI entrypoints
- `src/chessquant_ml/data/` - splitting and dataset utilities
- `src/chessquant_ml/exporting/` - ONNX and web export logic
- `src/chessquant_ml/lab/` - policy, plotting, targets, tracking, evaluation helpers
- `src/chessquant_ml/pipeline/` - fetch, enrich, feature computation
- `src/chessquant_ml/training/` - production-style training code
- `src/chessquant_ml/utils/` - shared helpers

## Data flow

`data/raw` -> `data/enriched` -> `data/features` -> `data/models` + `data/artifacts`

## Outputs flow

`outputs/models/<run_id>` contains experiment artifacts for the research workflow.

## Rules

- Do not create new `PATCH*` folders in-repo.
- Temporary migrations should live outside the repo root.
- Generated files belong in `data/` or `outputs/`, not inside `src/`.
- Keep package code only under `src/chessquant_ml/`.
'@ | Set-Content -Path (Join-Path $docsDir 'REPO_LAYOUT.md') -Encoding UTF8

@'
# ChessQuant ML Labs - Start Here

## Daily workflow

1. Fetch / enrich / build features with the package CLI.
2. Train the production-style model from `src/chessquant_ml/training/`.
3. Export ONNX from `src/chessquant_ml/exporting/`.
4. Use the research sandbox for objective comparison and policy evaluation.

## Important locations

- Production-like model artifacts: `data/models/`
- Export manifests: `data/artifacts/`
- Research outputs copied or compared later: `outputs/`
- Handoff bundle for implementation team: `production_handoff/`

## Cleanup policy

- Remove cache-like Python artifacts (`__pycache__`, `*.pyc`) freely.
- Keep `data/` and `outputs/` unless you intentionally want to reset experiments.
- Keep `production_handoff/` because it is part of delivery.
'@ | Set-Content -Path (Join-Path $docsDir 'START_HERE.md') -Encoding UTF8

Write-Host "[OK] Main repo cleaned: $RepoRoot"
