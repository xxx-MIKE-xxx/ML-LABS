from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from chessquant_ml.data.splitter import iter_chronological_group_splits, leakage_report
from chessquant_ml.lab.targets import add_target
from chessquant_ml.lab.tracking import configure_mlflow, log_config, log_metrics_flat

import mlflow



def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))



def build_model(model_cfg: dict[str, Any]):
    kind = model_cfg["kind"]
    params = model_cfg.get("params", {})
    if kind == "xgb_classifier":
        return XGBClassifier(**params)
    if kind == "xgb_regressor":
        return XGBRegressor(**params)
    if kind == "logistic_regression":
        return LogisticRegression(**params)
    if kind == "linear_regression":
        return LinearRegression(**params)
    raise ValueError(f"Unsupported model kind: {kind}")



def choose_features(df: pd.DataFrame, *, drop_cols: list[str], target_col: str, keep_numeric_only: bool) -> list[str]:
    drops = set(drop_cols + [target_col])
    cols = []
    for col in df.columns:
        if col in drops:
            continue
        if keep_numeric_only and not (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])):
            continue
        cols.append(col)
    if not cols:
        raise ValueError("No feature columns found after exclusions")
    return cols



def summarize_classification(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {"f1": float(f1_score(y_true, y_pred, zero_division=0))}
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics



def summarize_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mlflow_cfg = cfg["mlflow"]
    configure_mlflow(mlflow_cfg["tracking_uri"], mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=cfg["experiment_name"]):
        df = pd.read_parquet(cfg["paths"]["features_parquet"])
        schema = cfg["schema"]
        objective = cfg["objective"]
        df, target_col, task_type = add_target(
            df,
            group_col=schema["group_col"],
            order_col=schema["order_col"],
            cum_pl_col=schema["cum_pl_col"],
            objective_kind=objective["kind"],
            min_gain_elo=float(objective.get("min_gain_elo", 8.0)),
        )
        feature_cols = choose_features(
            df,
            drop_cols=schema.get("drop_cols", []),
            target_col=target_col,
            keep_numeric_only=bool(schema.get("keep_numeric_only", True)),
        )
        X = df[feature_cols].fillna(0.0)
        y = df[target_col]

        split_cfg = cfg["split"]
        fold_metrics = []
        first_leakage = None
        for split in iter_chronological_group_splits(
            df,
            group_col=schema["group_col"],
            time_col=schema["time_col"],
            n_splits=int(split_cfg.get("n_splits", 5)),
            group_gap=int(split_cfg.get("group_gap", 0)),
        ):
            model = build_model(cfg["model"])
            X_train = X.iloc[split.train_idx]
            y_train = y.iloc[split.train_idx]
            X_valid = X.iloc[split.valid_idx]
            y_valid = y.iloc[split.valid_idx]
            model.fit(X_train, y_train)
            if task_type == "classification":
                y_prob = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else np.asarray(model.predict(X_valid), dtype=float)
                metrics = summarize_classification(y_valid.to_numpy(dtype=int), y_prob)
            else:
                y_pred = np.asarray(model.predict(X_valid), dtype=float)
                metrics = summarize_regression(y_valid.to_numpy(dtype=float), y_pred)
            metrics["fold"] = float(split.fold)
            fold_metrics.append(metrics)
            if first_leakage is None:
                first_leakage = leakage_report(df, train_idx=split.train_idx, valid_idx=split.valid_idx, group_col=schema["group_col"], time_col=schema["time_col"])

        model = build_model(cfg["model"])
        model.fit(X, y)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg["paths"]["output_root"]) / f"{stamp}_{cfg['experiment_name']}"
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model.joblib"
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        manifest_path = run_dir / "manifest.json"

        joblib.dump(model, model_path)
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        summary = {
            "experiment_name": cfg["experiment_name"],
            "task_type": task_type,
            "target_col": target_col,
            "n_rows": int(len(df)),
            "n_features": int(len(feature_cols)),
            "feature_cols": feature_cols,
            "fold_metrics": fold_metrics,
            "first_leakage_report": first_leakage,
        }
        if fold_metrics:
            for key in [k for k in fold_metrics[0].keys() if k != "fold"]:
                vals = [m[key] for m in fold_metrics]
                summary[f"mean_{key}"] = float(np.mean(vals))
                summary[f"std_{key}"] = float(np.std(vals))

        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        manifest = {
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "features_path": cfg["paths"]["features_parquet"],
            "feature_order": feature_cols,
            "task_type": task_type,
            "target_col": target_col,
            "objective": objective,
            "schema": schema,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        log_config(cfg, run_dir / "mlflow_config.json")
        log_metrics_flat({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(manifest_path))

        latest = Path(cfg["paths"]["output_root"]) / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(run_dir.resolve(), target_is_directory=True)
        except OSError:
            latest.write_text(str(run_dir.resolve()), encoding="utf-8")

        print(json.dumps({"run_dir": str(run_dir), "task_type": task_type, "target_col": target_col}, indent=2))


if __name__ == "__main__":
    main()
