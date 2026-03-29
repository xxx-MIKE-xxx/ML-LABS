# ChessQuant ML Research Sandbox V2

This sandbox is the **research and experimentation layer** for ChessQuant V2.
It is designed to sit on top of your existing pipeline rather than replace it.

Why this layout:
- your current code already has a working end-to-end path: `fetch -> enrich -> features -> train -> export-web`
- the live app depends on a stable browser-oriented export and feature contract
- the ML lab should therefore be a **tight extension** of the current package, not a disconnected repo

This scaffold uses an industry-standard research layout:

```text
project/
├── configs/
│   ├── datasets/
│   ├── experiments/
│   └── policies/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── cache/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── reports/
│   └── web_exports/
├── scripts/
├── src/
│   └── chessquant_ml/
│       ├── cli/
│       ├── pipeline/
│       ├── training/
│       ├── exporting/
│       ├── data/
│       │   └── splitter.py
│       └── lab/
│           ├── targets.py
│           ├── tracking.py
│           ├── backtester.py
│           ├── policy.py
│           └── plots.py
└── tests/
```

## Why this is the recommended shape

It preserves the existing package family (`chessquant_ml`) while introducing:
- `src/` layout for cleaner imports and packaging
- `configs/` separated into datasets, experiments, and policies
- `outputs/` for immediate inspection dumps
- `src/chessquant_ml/data/splitter.py` to make time-aware splits explicit and testable
- `src/chessquant_ml/lab/*` for research-only target engineering, evaluation, and policy logic

## What stays production-aligned

Keep using the main production pipeline under:
- `src/chessquant_ml/pipeline/*`
- `src/chessquant_ml/training/*`
- `src/chessquant_ml/exporting/*`

The handoff says the existing ML LABS flow already covers fetch, enrich, feature building, XGBoost training, and browser export, and that the live/browser path depends on a stable feature order and ONNX/web-compatible export. The current feature family includes session grouping, session P/L, break time, rolling ACPL/time-per-move, and related features. fileciteturn1file0 fileciteturn1file6 fileciteturn1file16

## What this sandbox is for

Use this sandbox for:
- experimenting with new objectives
- swapping model families without destabilizing the production path
- leakage-safe splitting and evaluation
- MLflow-tracked runs
- backtesting decision policies such as stop / continue / short-break

## First migration principle

**Do not rewrite your current working pipeline first.**
Instead:
1. move the package under `src/`
2. create config and output folders
3. add lab modules next to the existing package
4. keep the current CLI and main training/export flow working
5. use the lab scripts to compare objectives before promoting one into production

## Included starter objectives

- `peak_binary`
- `meaningful_upside_binary`
- `future_upside_regression`
- `future_drawdown_regression`

## Included policy configs

- aggressive
- balanced
- conservative

## Included scripts

- `scripts/train_lab.py`
- `scripts/evaluate_policy.py`
- `scripts/run_matrix.ps1`
- `scripts/run_matrix.sh`
- `scripts/refactor_to_src_layout.ps1`
- `scripts/refactor_to_src_layout.sh`

## MLflow

This sandbox assumes your local MLflow tracking server is available at:

```text
http://127.0.0.1:5000/
```

Each run logs:
- config
- fold metrics
- summary metrics
- saved model
- manifest with feature order and target info
- policy backtest summary
- generated charts

## Minimal install additions

```powershell
poetry add mlflow pyyaml pandas pyarrow joblib scikit-learn xgboost numpy matplotlib
```

If missing from your environment:

```powershell
poetry add polars duckdb python-chess zstandard httpx tqdm onnx onnxruntime skl2onnx
```

## Recommended small first run

```powershell
poetry run chessquant-ml all --max-games 200
poetry run python scripts/train_lab.py --config configs/experiments/future_upside_xgb.yaml
poetry run python scripts/evaluate_policy.py --run-dir outputs/models/latest --policy configs/policies/balanced.yaml
```

## Pyproject note

If you move the package to `src/chessquant_ml`, set Poetry package discovery like this:

```toml
[tool.poetry]
packages = [{ include = "chessquant_ml", from = "src" }]
```

A fuller patch note is included in `pyproject_patch.md`.
