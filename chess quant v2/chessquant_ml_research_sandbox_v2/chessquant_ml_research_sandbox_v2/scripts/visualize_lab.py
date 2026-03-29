from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from chessquant_ml.lab.backtester import backtest_stop_policy_detailed
from chessquant_ml.lab.policy import action_from_objective, threshold_from_policy
from chessquant_ml.lab.viz import (
    choose_numeric_features,
    load_manifest,
    plot_correlation_matrix,
    plot_feature_histograms,
    plot_policy_session_traces,
    plot_run_comparison,
    plot_session_traces,
    summarize_numeric_frame,
)


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def dataset_report(args: argparse.Namespace) -> None:
    df = pd.read_parquet(args.features_parquet)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_numeric_features(df, max_features=args.max_features)
    stats = summarize_numeric_frame(df, exclude=["target", "session_max", "is_max", "first_max_game"])
    stats_path = out_dir / "numeric_summary.csv"
    stats.to_csv(stats_path)

    hist_dir = out_dir / "histograms"
    plot_feature_histograms(
        df,
        feature_cols=selected,
        target_col=args.target_col if args.target_col in df.columns else None,
        output_dir=hist_dir,
    )
    plot_correlation_matrix(df, feature_cols=selected, output_path=out_dir / "correlation_matrix.png")
    if args.session_col in df.columns and args.time_col in df.columns and args.cum_pl_col in df.columns:
        plot_session_traces(
            df,
            session_col=args.session_col,
            time_col=args.time_col,
            cum_pl_col=args.cum_pl_col,
            output_dir=out_dir / "session_traces",
            max_sessions=args.max_sessions,
        )

    print(f"Wrote dataset report to {out_dir}")
    print(f"Numeric summary: {stats_path}")


def compare_runs(args: argparse.Namespace) -> None:
    csv_path, chart_path = plot_run_comparison(args.run_dirs, output_dir=args.output_dir)
    print(f"Comparison CSV: {csv_path}")
    print(f"Comparison chart: {chart_path}")


def policy_report(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)
    policy_cfg = load_yaml(args.policy)
    objective_kind = manifest["objective"]["kind"]
    threshold = threshold_from_policy(policy_cfg, objective_kind)

    oof_path = Path(manifest["oof_predictions_path"])
    df = pd.read_parquet(oof_path)
    df = df[df["oof_is_validation"] == 1].copy()
    df["policy_score"] = pd.to_numeric(df["oof_score"], errors="coerce")
    df["policy_action"] = action_from_objective(
        df["policy_score"].to_numpy(dtype=float),
        objective_kind=objective_kind,
        policy_cfg=policy_cfg,
        threshold_override=threshold,
    )

    schema = manifest["schema"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_col = args.y_col
    if y_col not in df.columns:
        raise ValueError(f"Column '{y_col}' not found in OOF dataframe")

    session_plot_dir = out_dir / "session_policy_plots"
    written = plot_policy_session_traces(
        df,
        session_col=schema["group_col"],
        order_col=schema["order_col"],
        y_col=y_col,
        action_col="policy_action",
        output_dir=session_plot_dir,
        max_sessions=args.max_sessions,
        actual_peak_col=schema["cum_pl_col"],
        score_col="policy_score",
    )

    session_df, summary = backtest_stop_policy_detailed(
        df,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        action_col="policy_action",
    )
    summary_payload = {
        "policy": policy_cfg["name"],
        "objective_kind": objective_kind,
        "threshold": float(threshold),
        "y_col": y_col,
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

    session_csv = out_dir / "session_policy_summary.csv"
    summary_json = out_dir / "policy_report_summary.json"
    session_df.to_csv(session_csv, index=False)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                **summary_payload,
                "session_csv": str(session_csv),
                "summary_json": str(summary_json),
                "session_plot_dir": str(session_plot_dir),
                "n_session_plots": len(written),
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualization utilities for ChessQuant ML lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dataset = sub.add_parser("dataset-report")
    p_dataset.add_argument("--features-parquet", required=True)
    p_dataset.add_argument("--output-dir", required=True)
    p_dataset.add_argument("--target-col", default="target")
    p_dataset.add_argument("--session-col", default="session_id")
    p_dataset.add_argument("--time-col", default="created_at_ms")
    p_dataset.add_argument("--cum-pl-col", default="session_pl")
    p_dataset.add_argument("--max-features", type=int, default=12)
    p_dataset.add_argument("--max-sessions", type=int, default=16)
    p_dataset.set_defaults(func=dataset_report)

    p_compare = sub.add_parser("compare-runs")
    p_compare.add_argument("--run-dirs", nargs="+", required=True)
    p_compare.add_argument("--output-dir", required=True)
    p_compare.set_defaults(func=compare_runs)

    p_policy = sub.add_parser("policy-report")
    p_policy.add_argument("--run-dir", required=True)
    p_policy.add_argument("--policy", required=True)
    p_policy.add_argument("--output-dir", required=True)
    p_policy.add_argument("--y-col", default="session_pl")
    p_policy.add_argument("--max-sessions", type=int)
    p_policy.set_defaults(func=policy_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
