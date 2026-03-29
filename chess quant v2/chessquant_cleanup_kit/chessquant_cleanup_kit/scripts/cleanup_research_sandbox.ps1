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

# Move sandbox docs into docs/
$docMoves = @(
    'MIGRATION_GUIDE.md',
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

# Remove Python caches and backup files
Get-ChildItem -Path $RepoRoot -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem -Path $RepoRoot -Recurse -File -Include '*.pyc', '*.bak', '*.pre_patch_backup' -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Add a local pyproject if missing so the sandbox can be used directly
$pyprojectPath = Join-Path $RepoRoot 'pyproject.toml'
if (-not (Test-Path $pyprojectPath)) {
@'
[tool.poetry]
name = "chessquant-ml-research-sandbox-v2"
version = "0.1.0"
description = "Research sandbox for ChessQuant objective and policy evaluation"
authors = ["John Scott <chatgpt4.mm.acc.2@gmail.com>"]
packages = [{ include = "chessquant_ml", from = "src" }]

[tool.poetry.dependencies]
python = ">=3.13,<3.14"
numpy = "^2.1.0"
pandas = "^2.2.0"
pyarrow = "^18.0.0"
scikit-learn = "^1.5.0"
xgboost = "^2.1.0"
matplotlib = "^3.9.0"
pyyaml = "^6.0.3"
joblib = "^1.4.2"
mlflow = "^2.17.0"

[tool.poetry.group.dev.dependencies]
pytest = "^8.3.0"
ruff = "^0.6.7"

[build-system]
requires = ["poetry-core>=1.8.0"]
build-backend = "poetry.core.masonry.api"
'@ | Set-Content -Path $pyprojectPath -Encoding UTF8
}

# Write documentation
@'
# ChessQuant Research Sandbox - Repository Layout

This repository is the clean experimentation workspace for trying new objectives,
policies, and evaluation methods without touching the production-style training flow.

## Top-level directories

- `configs/` - experiment, dataset, and policy YAMLs
- `data/` - local research parquet inputs copied or prepared for experiments
- `docs/` - migration notes and onboarding docs
- `notebooks/` - ad-hoc exploration
- `outputs/` - model runs, figures, reports, threshold sweeps, policy plots
- `scripts/` - experiment runner, evaluator, visualization entrypoints
- `src/chessquant_ml/` - reusable lab code
- `tests/` - tests for core utilities

## Source package structure

- `src/chessquant_ml/data/` - splitter utilities
- `src/chessquant_ml/lab/` - backtesting, targets, policy logic, plots, viz, tracking

## Expected workflow

1. Place or generate a research parquet under `data/features/`.
2. Run `scripts/train_lab.py` with one experiment config.
3. Run `scripts/evaluate_policy.py` for backtests and threshold sweeps.
4. Run `scripts/visualize_lab.py` for per-session plots and reports.

## Output convention

Every run writes to `outputs/models/<timestamp>_<experiment_name>/`.
The `outputs/models/latest` file points to the newest run directory.
'@ | Set-Content -Path (Join-Path $docsDir 'REPO_LAYOUT.md') -Encoding UTF8

@'
# ChessQuant Research Sandbox - Start Here

## Install

After this cleanup, you can install dependencies directly in the sandbox:

```powershell
poetry install
```

## First run

```powershell
poetry run python scripts/train_lab.py --config configs/experiments/peak_binary_xgb.yaml
poetry run python scripts/evaluate_policy.py backtest-policy --run-dir outputs/models/latest --policy configs/policies/balanced.yaml
poetry run python scripts/visualize_lab.py policy-report --run-dir $(Get-Content outputs/models/latest) --policy configs/policies/balanced.yaml --output-dir outputs/reports/policy_report_peak
```

## Cleanup policy

- Keep only stable code and docs inside the repo.
- Do not keep ad-hoc patch folders after applying them.
- Generated reports belong in `outputs/`.
- Dataset copies belong in `data/`.
'@ | Set-Content -Path (Join-Path $docsDir 'START_HERE.md') -Encoding UTF8

Write-Host "[OK] Research sandbox cleaned: $RepoRoot"
