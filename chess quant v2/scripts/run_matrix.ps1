$ErrorActionPreference = "Stop"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:MLFLOW_EXPERIMENT_NAME = "chessquant-lab"

poetry run python scripts/train_lab.py --config configs/experiments/peak_binary_xgb.yaml
poetry run python scripts/train_lab.py --config configs/experiments/meaningful_upside_xgb.yaml
poetry run python scripts/train_lab.py --config configs/experiments/future_upside_xgb.yaml
poetry run python scripts/train_lab.py --config configs/experiments/future_drawdown_xgb.yaml
