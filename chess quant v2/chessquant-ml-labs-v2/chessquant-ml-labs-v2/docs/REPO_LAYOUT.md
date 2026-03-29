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
