from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from chessquant_ml.settings import settings
from chessquant_ml.utils.io import read_json, write_json


def _feature_columns() -> list[str]:
    schema = read_json(settings.feature_schema_path)
    return list(schema["core_features"])


def train_tilt_model(feature_path: Path) -> tuple[Path, Path]:
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    print(f"Loading features from {feature_path} ...")
    df = pl.read_parquet(feature_path)
    feature_cols = _feature_columns()
    pdf = df.select(feature_cols + ["target", "session_id", "rating_diff"]).to_pandas()

    print(f"Rows: {len(pdf)} | Features: {len(feature_cols)}")

    X = pdf[feature_cols]
    y = pdf["target"]
    groups = pdf["session_id"]

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 150,
        "gamma": 1.0,
        "min_child_weight": 5,
        "scale_pos_weight": 5,
        "n_jobs": -1,
        "random_state": 42,
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    aucs: list[float] = []

    print("Running grouped CV ...")
    for fold_idx, (train_idx, valid_idx) in enumerate(
        tqdm(cv.split(X, y, groups), total=5, desc="CV folds", unit="fold"),
        start=1,
    ):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[valid_idx])[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], probs))
        aucs.append(auc)
        print(f"  fold {fold_idx}: auc={auc:.4f}")

    print("Fitting final model on full dataset ...")
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    probs_all = final_model.predict_proba(X)[:, 1]

    thresholds = np.arange(0.30, 0.90, 0.02)
    baseline = float(pdf["rating_diff"].sum())
    best_threshold = 0.5
    best_pl = float("-inf")

    print("Optimizing decision threshold ...")
    pdf = pdf.copy()
    pdf["tilt_prob"] = probs_all
    for t in tqdm(thresholds, desc="Threshold sweep", unit="thr"):
        sim_pl = 0.0
        for _, grp in pdf.groupby("session_id"):
            stops = grp.index[grp["tilt_prob"] > t]
            if len(stops) > 0:
                stop_idx = stops[0]
                sim_pl += float(grp.loc[:stop_idx, "rating_diff"].sum())
            else:
                sim_pl += float(grp["rating_diff"].sum())
        if sim_pl > best_pl:
            best_pl = sim_pl
            best_threshold = float(t)

    model_path = settings.models_dir / "tilt_xgb.json"
    config_path = settings.models_dir / "tilt_config.joblib"
    summary_path = settings.artifacts_dir / "training_summary.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model.save_model(model_path)
    joblib.dump(
        {
            "features": feature_cols,
            "params": params,
            "threshold": best_threshold,
            "cv_auc_mean": float(np.mean(aucs)),
            "cv_auc_std": float(np.std(aucs)),
            "pl_improvement_est": float(best_pl - baseline),
        },
        config_path,
    )
    write_json(
        summary_path,
        {
            "rows": int(len(pdf)),
            "features": feature_cols,
            "cv_auc_scores": aucs,
            "cv_auc_mean": float(np.mean(aucs)),
            "cv_auc_std": float(np.std(aucs)),
            "threshold": best_threshold,
            "baseline_rating_diff_sum": baseline,
            "optimized_rating_diff_sum": float(best_pl),
            "estimated_gain": float(best_pl - baseline),
        },
    )
    print(
        f"Training summary: auc_mean={float(np.mean(aucs)):.4f}, "
        f"threshold={best_threshold:.2f}, est_gain={float(best_pl - baseline):.2f}"
    )
    return model_path, config_path