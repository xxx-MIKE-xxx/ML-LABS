param(
    [string]$RepoRoot = (Get-Location).Path,
    [switch]$CreateBackup
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Step([string]$Message) {
    Write-Host "[INFO] $Message"
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message"
}

function Ensure-Dir([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Remove-IfExists([string]$Path) {
    if (Test-Path $Path) {
        Remove-Item -Recurse -Force $Path
    }
}

function Copy-Tree([string]$Source, [string]$Destination) {
    if (-not (Test-Path $Source)) {
        return
    }

    Ensure-Dir $Destination
    robocopy $Source $Destination /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP /XD __pycache__ .pytest_cache .ruff_cache .git | Out-Null

    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed: $Source -> $Destination"
    }
}

function Copy-File-IfExists([string]$Source, [string]$Destination) {
    if (Test-Path $Source) {
        Ensure-Dir (Split-Path -Parent $Destination)
        Copy-Item -Force $Source $Destination
    }
}

function Replace-InFile([string]$Path, [hashtable]$Replacements) {
    if (-not (Test-Path $Path)) {
        return
    }

    $content = Get-Content -Raw -Encoding UTF8 $Path
    foreach ($key in $Replacements.Keys) {
        $content = $content.Replace($key, $Replacements[$key])
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Resolve-Path $Path), $content, $utf8NoBom)
}

function Test-RepoLike([string]$Path) {
    if (-not (Test-Path $Path)) { return $false }

    return (
        (Test-Path (Join-Path $Path "pyproject.toml")) -or
        (Test-Path (Join-Path $Path "src")) -or
        (Test-Path (Join-Path $Path "configs")) -or
        (Test-Path (Join-Path $Path "scripts"))
    )
}

function Resolve-FirstExisting([string[]]$Candidates) {
    foreach ($candidate in $Candidates) {
        if (Test-RepoLike $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    return $null
}

$RepoRoot = (Resolve-Path $RepoRoot).Path

$MainRepo = Resolve-FirstExisting @(
    (Join-Path $RepoRoot "chessquant-ml-labs-v2\chessquant-ml-labs-v2"),
    (Join-Path $RepoRoot "chessquant-ml-labs-v2"),
    $RepoRoot
)

$ResearchRepo = Resolve-FirstExisting @(
    (Join-Path $RepoRoot "chessquant_ml_research_sandbox_v2\chessquant_ml_research_sandbox_v2"),
    (Join-Path $RepoRoot "chessquant_ml_research_sandbox_v2")
)

$CleanupKit = Join-Path $RepoRoot "chessquant_cleanup_kit"
$OldScript1 = Join-Path $RepoRoot "one_time_unify_repo.ps1"
$OldScript2 = Join-Path $RepoRoot "inplace_unify_repo.ps1.txt"
$TopStockfish = Join-Path $RepoRoot "stockfish-windows-x86-64-avx2"

if (-not $MainRepo) {
    throw "Main repo not found under: $RepoRoot"
}

if (-not $ResearchRepo) {
    throw "Research repo not found under: $RepoRoot"
}

Write-Host "[INFO] Main repo resolved to: $MainRepo"
Write-Host "[INFO] Research repo resolved to: $ResearchRepo"

if ($CreateBackup) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $BackupRoot = Join-Path $RepoRoot "_pre_unify_backup_$stamp"
    Write-Step "Creating backup at $BackupRoot"
    Ensure-Dir $BackupRoot

    $backupItems = @(
        "chessquant-ml-labs-v2",
        "chessquant_ml_research_sandbox_v2",
        "chessquant_cleanup_kit",
        "configs",
        "data",
        "notebooks",
        "outputs",
        "scripts",
        "src",
        "tests",
        "stockfish-windows-x86-64-avx2",
        "one_time_unify_repo.ps1",
        "inplace_unify_repo.ps1.txt"
    )

    foreach ($item in $backupItems) {
        $src = Join-Path $RepoRoot $item
        if (Test-Path $src) {
            $dst = Join-Path $BackupRoot $item
            if ((Get-Item $src).PSIsContainer) {
                Copy-Tree $src $dst
            } else {
                Copy-File-IfExists $src $dst
            }
        }
    }

    Write-Ok "Backup created"
}

Write-Step "Preparing final root structure"

$finalDirs = @(
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
    "data\registry",
    "data\samples",
    "artifacts",
    "artifacts\models",
    "artifacts\reports",
    "artifacts\figures",
    "artifacts\exports",
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
    "tools"
)

foreach ($dir in $finalDirs) {
    Ensure-Dir (Join-Path $RepoRoot $dir)
}

Write-Step "Copying root project files from main repo"

$rootFiles = @(
    ".env",
    ".env.example",
    ".env.example.lab",
    "pyproject.toml",
    "poetry.lock",
    "README.md",
    ".gitignore"
)

foreach ($file in $rootFiles) {
    Copy-File-IfExists (Join-Path $MainRepo $file) (Join-Path $RepoRoot $file)
}

Write-Step "Merging configs"
Copy-Tree (Join-Path $MainRepo "configs") (Join-Path $RepoRoot "configs")
Copy-Tree (Join-Path $ResearchRepo "configs") (Join-Path $RepoRoot "configs")
Copy-File-IfExists (Join-Path $MainRepo "config\feature_schema.json") (Join-Path $RepoRoot "configs\features\feature_schema.json")

Write-Step "Merging data"
Copy-Tree (Join-Path $MainRepo "data\raw") (Join-Path $RepoRoot "data\raw")
Copy-Tree (Join-Path $MainRepo "data\enriched") (Join-Path $RepoRoot "data\enriched")
Copy-Tree (Join-Path $MainRepo "data\features") (Join-Path $RepoRoot "data\features")
Copy-Tree (Join-Path $MainRepo "data\cache") (Join-Path $RepoRoot "data\cache")
Copy-Tree (Join-Path $MainRepo "data\registry") (Join-Path $RepoRoot "data\registry")
Copy-Tree (Join-Path $MainRepo "data\snapshots") (Join-Path $RepoRoot "data\registry\snapshots")
Copy-Tree (Join-Path $ResearchRepo "data\features") (Join-Path $RepoRoot "data\features")

Write-Step "Merging scripts"
Copy-Tree (Join-Path $MainRepo "scripts") (Join-Path $RepoRoot "scripts")
Copy-Tree (Join-Path $ResearchRepo "scripts") (Join-Path $RepoRoot "scripts")

Write-Step "Merging source code"
Copy-Tree (Join-Path $MainRepo "src") (Join-Path $RepoRoot "src")
Copy-Tree (Join-Path $ResearchRepo "src") (Join-Path $RepoRoot "src")

Write-Step "Merging tests"
Copy-Tree (Join-Path $MainRepo "tests") (Join-Path $RepoRoot "tests")
Copy-Tree (Join-Path $ResearchRepo "tests") (Join-Path $RepoRoot "tests")

Write-Step "Merging notebooks"
Copy-Tree (Join-Path $MainRepo "notebooks") (Join-Path $RepoRoot "notebooks")
Copy-Tree (Join-Path $ResearchRepo "notebooks") (Join-Path $RepoRoot "notebooks")

Write-Step "Merging docs"
Copy-Tree (Join-Path $MainRepo "docs") (Join-Path $RepoRoot "docs")
Copy-Tree (Join-Path $ResearchRepo "docs") (Join-Path $RepoRoot "docs")

Write-Step "Moving generated outputs into artifacts"
Copy-Tree (Join-Path $MainRepo "outputs\models") (Join-Path $RepoRoot "artifacts\models")
Copy-Tree (Join-Path $MainRepo "outputs\reports") (Join-Path $RepoRoot "artifacts\reports")
Copy-Tree (Join-Path $MainRepo "outputs\figures") (Join-Path $RepoRoot "artifacts\figures")
Copy-Tree (Join-Path $MainRepo "outputs\metrics") (Join-Path $RepoRoot "artifacts\metrics")
Copy-Tree (Join-Path $MainRepo "outputs\web_exports") (Join-Path $RepoRoot "artifacts\exports\web")

Copy-Tree (Join-Path $ResearchRepo "outputs\models") (Join-Path $RepoRoot "artifacts\models")
Copy-Tree (Join-Path $ResearchRepo "outputs\reports") (Join-Path $RepoRoot "artifacts\reports")
Copy-Tree (Join-Path $ResearchRepo "outputs\figures") (Join-Path $RepoRoot "artifacts\figures")
Copy-Tree (Join-Path $ResearchRepo "outputs\metrics") (Join-Path $RepoRoot "artifacts\metrics")
Copy-Tree (Join-Path $ResearchRepo "outputs\web_exports") (Join-Path $RepoRoot "artifacts\exports\web")

Write-Step "Moving production handoff into artifacts\exports"
Copy-Tree (Join-Path $MainRepo "production_handoff") (Join-Path $RepoRoot "artifacts\exports\production_handoff")

Write-Step "Reorganizing MLflow files"
Copy-File-IfExists (Join-Path $MainRepo "mlflow.db") (Join-Path $RepoRoot "mlflow\db\mlflow.db")
Copy-Tree (Join-Path $MainRepo "mlruns") (Join-Path $RepoRoot "mlflow\runs")
Copy-Tree (Join-Path $MainRepo "mlartifacts") (Join-Path $RepoRoot "mlflow\artifacts")

Write-Step "Moving Stockfish under tools"
if (Test-Path $TopStockfish) {
    Copy-Tree $TopStockfish (Join-Path $RepoRoot "tools\stockfish")
}

Write-Step "Writing repo documentation"

$startHere = @"
# Start Here

This repo is the single source of truth for ChessQuant ML work.

## Main folders
- configs/      editable YAML and schema config
- data/         raw, enriched, and feature datasets
- artifacts/    generated models, reports, figures, exports
- mlflow/       experiment database and MLflow tracking store
- scripts/      terminal entrypoints
- src/          reusable Python package code
- tests/        automated checks
- tools/        bundled executables like Stockfish

## Important note
Older nested repos, patch folders, and cleanup helpers were merged into this one structure.
"@

$repoLayout = @"
# Repo Layout

## configs/
Human-edited runtime and experiment configuration.

## data/
Canonical datasets and persistent pipeline inputs.

## artifacts/
Generated outputs from training, evaluation, visualization, and exports.

## mlflow/
Experiment tracking internals:
- db/          sqlite database
- runs/        mlruns metadata store
- artifacts/   files attached to MLflow runs

## scripts/
CLI entrypoints.

## src/
Project code package.

## tests/
Unit, integration, and regression tests.

## tools/
External tools bundled locally.
"@

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "docs\START_HERE.md"), $startHere.Trim(), $utf8NoBom)
[System.IO.File]::WriteAllText((Join-Path $RepoRoot "docs\REPO_LAYOUT.md"), $repoLayout.Trim(), $utf8NoBom)

Write-Step "Updating common config paths"

$yamlFiles = Get-ChildItem -Path (Join-Path $RepoRoot "configs") -Recurse -Include *.yaml,*.yml -File -ErrorAction SilentlyContinue
foreach ($yaml in $yamlFiles) {
    Replace-InFile $yaml.FullName @{
        "outputs/models" = "artifacts/models"
        "outputs/reports" = "artifacts/reports"
        "outputs/figures" = "artifacts/figures"
        "outputs/web_exports" = "artifacts/exports/web"
        "config/feature_schema.json" = "configs/features/feature_schema.json"
    }
}

Write-Step "Removing obsolete top-level duplicates and wrapper folders"

$obsoletePaths = @(
    (Join-Path $RepoRoot "chessquant-ml-labs-v2"),
    (Join-Path $RepoRoot "chessquant_ml_research_sandbox_v2"),
    (Join-Path $RepoRoot "chessquant_cleanup_kit"),
    (Join-Path $RepoRoot "outputs"),
    (Join-Path $RepoRoot "stockfish-windows-x86-64-avx2"),
    (Join-Path $RepoRoot "one_time_unify_repo.ps1"),
    (Join-Path $RepoRoot "inplace_unify_repo.ps1.txt")
)

foreach ($path in $obsoletePaths) {
    Remove-IfExists $path
}

Write-Step "Removing caches and temp files"
Get-ChildItem -Path $RepoRoot -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", ".ruff_cache") } |
    ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
    }

Get-ChildItem -Path $RepoRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "*.bak" -or $_.Name -like "*.pre_patch_backup" } |
    ForEach-Object {
        Remove-Item -Force $_.FullName -ErrorAction SilentlyContinue
    }

Write-Step "Rebuilding artifacts\models\latest pointer"
$ModelRoot = Join-Path $RepoRoot "artifacts\models"
if (Test-Path $ModelRoot) {
    $modelRuns = Get-ChildItem -Path $ModelRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^\d{8}_\d{6}_.+' } |
        Sort-Object Name

    if ($modelRuns.Count -gt 0) {
        [System.IO.File]::WriteAllText((Join-Path $ModelRoot "latest"), $modelRuns[-1].FullName, $utf8NoBom)
    }
}

Write-Ok "Repo restructured successfully"
Write-Host ""
Write-Host "Final root: $RepoRoot"
Write-Host ""
Write-Host "Next commands:"
Write-Host "  cd `"$RepoRoot`""
Write-Host "  poetry check"
Write-Host "  poetry install"