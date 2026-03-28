from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chessquant_ml.lab.viz import (
    choose_numeric_features,
    plot_correlation_matrix,
    plot_feature_histograms,
    plot_run_comparison,
    plot_session_traces,
    summarize_numeric_frame,
)


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

    return parser



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
