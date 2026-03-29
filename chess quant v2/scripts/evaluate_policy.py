from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from chessquant_ml.lab.backtester import backtest_stop_policy_detailed
from chessquant_ml.lab.plots import save_session_plot
from chessquant_ml.lab.policy import (
    action_from_objective,
    objective_threshold_name,
    threshold_from_policy,
)
from chessquant_ml.lab.viz import plot_threshold_sweep


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def resolve_run_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        return p
    if p.is_file():
        return Path(p.read_text(encoding="utf-8").strip())
    raise FileNotFoundError(path_str)


def load_scoring_frame(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any], str]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    oof_path = Path(manifest.get("oof_predictions_path", ""))
    if oof_path.exists():
        df = pd.read_parquet(oof_path)
        df = df[df["oof_is_validation"] == 1].copy()
        if df.empty:
            raise ValueError(f"OOF predictions file exists but contains no validation rows: {oof_path}")
        df["policy_score"] = pd.to_numeric(df["oof_score"], errors="coerce")
        return df, manifest, "oof_validation"

    model = joblib.load(run_dir / "model.joblib")
    df = pd.read_parquet(manifest["features_path"]).copy()
    feature_order = manifest["feature_order"]
    X = df[feature_order].fillna(0.0)
    if manifest["task_type"] == "classification":
        score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.asarray(model.predict(X), dtype=float)
    else:
        score = np.asarray(model.predict(X), dtype=float)
    df["policy_score"] = score
    return df, manifest, "in_sample_fallback"


def evaluate_single_threshold(
    df: pd.DataFrame,
    *,
    manifest: dict[str, Any],
    policy_cfg: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    objective_kind = manifest["objective"]["kind"]
    schema = manifest["schema"]

    work = df.copy()
    work["policy_action"] = action_from_objective(
        work["policy_score"].to_numpy(dtype=float),
        objective_kind=objective_kind,
        policy_cfg=policy_cfg,
        threshold_override=threshold,
    )

    session_df, summary = backtest_stop_policy_detailed(
        work,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        action_col="policy_action",
    )

    out = {
        "policy": policy_cfg["name"],
        "objective_kind": objective_kind,
        "threshold_name": objective_threshold_name(objective_kind),
        "threshold_value": float(threshold),
        "n_sessions": summary.n_sessions,
        "n_stop_sessions": summary.n_stop_sessions,
        "total_actual_final_pl": summary.total_actual_final_pl,
        "total_stop_pl": summary.total_stop_pl,
        "total_peak_pl": summary.total_peak_pl,
        "total_elo_delta_vs_actual": summary.total_elo_delta_vs_actual,
        "total_elo_left_vs_peak": summary.total_elo_left_vs_peak,
        "mean_realized_pl": summary.mean_realized_pl,
        "mean_actual_final_pl": summary.mean_actual_final_pl,
        "mean_peak_pl": summary.mean_peak_pl,
        "mean_capture_ratio": summary.mean_capture_ratio,
        "mean_stop_game_index": summary.mean_stop_game_index,
    }
    return session_df, out


def cmd_backtest_policy(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    policy_cfg = load_yaml(args.policy)
    df, manifest, score_source = load_scoring_frame(run_dir)

    threshold = threshold_from_policy(policy_cfg, manifest["objective"]["kind"])
    session_df, out = evaluate_single_threshold(
        df,
        manifest=manifest,
        policy_cfg=policy_cfg,
        threshold=threshold,
    )
    out["score_source"] = score_source

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / f"policy_eval_{policy_cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    session_csv = out_dir / "session_policy_summary.csv"
    summary_json = out_dir / "policy_summary.json"
    preview_png = out_dir / "policy_preview.png"

    session_df.to_csv(session_csv, index=False)
    summary_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    schema = manifest["schema"]
    plot_df = df.copy()
    plot_df["policy_action"] = action_from_objective(
        plot_df["policy_score"].to_numpy(dtype=float),
        objective_kind=manifest["objective"]["kind"],
        policy_cfg=policy_cfg,
        threshold_override=threshold,
    )
    save_session_plot(
        plot_df,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        action_col="policy_action",
        out_path=preview_png,
    )

    print(
        json.dumps(
            {
                **out,
                "run_dir": str(run_dir),
                "output_dir": str(out_dir),
                "session_csv": str(session_csv),
                "summary_json": str(summary_json),
                "preview_png": str(preview_png),
            },
            indent=2,
        )
    )


def cmd_threshold_sweep(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir)
    policy_cfg = load_yaml(args.policy)
    df, manifest, score_source = load_scoring_frame(run_dir)
    objective_kind = manifest["objective"]["kind"]

    default_threshold = threshold_from_policy(policy_cfg, objective_kind)

    start = args.start if args.start is not None else max(0.0, default_threshold - 0.25)
    stop = args.stop if args.stop is not None else (default_threshold + 0.25)
    thresholds = np.linspace(start, stop, num=int(args.num))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        _, out = evaluate_single_threshold(
            df,
            manifest=manifest,
            policy_cfg=policy_cfg,
            threshold=float(threshold),
        )
        rows.append(out)

    sweep_df = pd.DataFrame(rows)
    metric_col = args.metric
    if metric_col not in sweep_df.columns:
        raise ValueError(f"Metric '{metric_col}' is not available. Available: {sorted(sweep_df.columns)}")

    best_idx = int(sweep_df[metric_col].astype(float).idxmax())
    best_row = sweep_df.iloc[best_idx].to_dict()

    out_dir = Path(args.output_dir) if args.output_dir else run_dir / f"threshold_sweep_{policy_cfg['name']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "threshold_sweep.csv"
    png_path = out_dir / "threshold_sweep.png"
    summary_path = out_dir / "threshold_sweep_summary.json"

    sweep_df.to_csv(csv_path, index=False)
    plot_threshold_sweep(
        sweep_df,
        threshold_col="threshold_value",
        metric_col=metric_col,
        output_path=png_path,
    )

    payload = {
        "policy": policy_cfg["name"],
        "objective_kind": objective_kind,
        "threshold_name": objective_threshold_name(objective_kind),
        "score_source": score_source,
        "optimized_metric": metric_col,
        "default_threshold": float(default_threshold),
        "best_threshold": float(best_row["threshold_value"]),
        "best_metric_value": float(best_row[metric_col]),
        "best_row": best_row,
        "csv_path": str(csv_path),
        "plot_path": str(png_path),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate policy quality on ChessQuant research runs")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest-policy")
    p_backtest.add_argument("--run-dir", required=True)
    p_backtest.add_argument("--policy", required=True)
    p_backtest.add_argument("--output-dir")
    p_backtest.set_defaults(func=cmd_backtest_policy)

    p_sweep = sub.add_parser("threshold-sweep")
    p_sweep.add_argument("--run-dir", required=True)
    p_sweep.add_argument("--policy", required=True)
    p_sweep.add_argument("--output-dir")
    p_sweep.add_argument("--metric", default="total_elo_delta_vs_actual")
    p_sweep.add_argument("--start", type=float)
    p_sweep.add_argument("--stop", type=float)
    p_sweep.add_argument("--num", type=int, default=21)
    p_sweep.set_defaults(func=cmd_threshold_sweep)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
