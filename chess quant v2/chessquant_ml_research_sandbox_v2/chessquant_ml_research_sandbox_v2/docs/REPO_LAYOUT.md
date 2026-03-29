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
