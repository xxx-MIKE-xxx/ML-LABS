from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LEAKY_COLUMNS = {
    "target",
    "session_max",
    "is_max",
    "first_max_game",
}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_metrics(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    return json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))


def load_manifest(run_dir: str | Path) -> dict:
    run_path = Path(run_dir)
    return json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))


def summarize_numeric_frame(df: pd.DataFrame, *, exclude: Iterable[str] | None = None) -> pd.DataFrame:
    exclude_set = set(exclude or [])
    numeric_cols = [
        c for c in df.columns
        if c not in exclude_set and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    stats = df[numeric_cols].describe().T
    stats["missing_frac"] = df[numeric_cols].isna().mean()
    return stats.sort_index()


def choose_numeric_features(
    df: pd.DataFrame,
    *,
    max_features: int = 12,
    extra_exclude: Iterable[str] | None = None,
) -> list[str]:
    exclude = DEFAULT_LEAKY_COLUMNS | set(extra_exclude or [])
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]
    preferred = [
        "my_acpl",
        "my_blunder_count",
        "largest_eval_drop_cp",
        "avg_eval_drop_cp",
        "eval_volatility",
        "rating_diff",
        "result",
        "my_avg_secs_per_move",
        "games_played",
        "session_pl",
        "speed_vs_start",
        "break_time_sec",
        "loss_streak",
        "roll_5_acpl_mean",
        "roll_5_time_per_move",
        "log_break_time",
    ]
    chosen = [c for c in preferred if c in numeric_cols]
    if len(chosen) < max_features:
        for c in numeric_cols:
            if c not in chosen:
                chosen.append(c)
            if len(chosen) >= max_features:
                break
    return chosen[:max_features]


def plot_feature_histograms(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str | None,
    output_dir: str | Path,
) -> list[Path]:
    out_dir = ensure_dir(output_dir)
    written: list[Path] = []
    for col in feature_cols:
        plt.figure(figsize=(7, 4))
        series = pd.to_numeric(df[col], errors="coerce")
        if target_col and target_col in df.columns and df[target_col].nunique(dropna=True) <= 8:
            for value in sorted(df[target_col].dropna().unique().tolist()):
                mask = df[target_col] == value
                plt.hist(series[mask].dropna(), bins=30, alpha=0.6, label=f"{target_col}={value}")
            plt.legend()
        else:
            plt.hist(series.dropna(), bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        out_path = out_dir / f"hist_{col}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        written.append(out_path)
    return written


def plot_correlation_matrix(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    output_path: str | Path,
) -> Path:
    corr = df[feature_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)
    fig, ax = plt.subplots(figsize=(max(7, len(feature_cols) * 0.6), max(6, len(feature_cols) * 0.6)))
    im = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=90)
    ax.set_yticklabels(feature_cols)
    ax.set_title("Feature correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_session_traces(
    df: pd.DataFrame,
    *,
    session_col: str,
    time_col: str,
    cum_pl_col: str,
    output_dir: str | Path,
    max_sessions: int = 16,
) -> list[Path]:
    out_dir = ensure_dir(output_dir)
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    sizes = work.groupby(session_col).size().sort_values(ascending=False)
    selected = sizes.head(max_sessions).index.tolist()
    written: list[Path] = []
    for session_id in selected:
        sdf = work[work[session_col] == session_id].sort_values(time_col)
        plt.figure(figsize=(7, 4))
        x = np.arange(1, len(sdf) + 1)
        plt.plot(x, pd.to_numeric(sdf[cum_pl_col], errors="coerce").to_numpy())
        plt.axhline(0.0, linewidth=1)
        plt.title(f"Session trace: {session_id}")
        plt.xlabel("game_in_session")
        plt.ylabel(cum_pl_col)
        out_path = out_dir / f"session_{session_id}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=140)
        plt.close()
        written.append(out_path)
    return written


def plot_run_comparison(
    run_dirs: list[str | Path],
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = ensure_dir(output_dir)
    rows: list[dict] = []
    for run_dir in run_dirs:
        metrics = load_metrics(run_dir)
        row = {"run": Path(run_dir).name}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value
        rows.append(row)
    comp = pd.DataFrame(rows)
    csv_path = out_dir / "run_comparison.csv"
    comp.to_csv(csv_path, index=False)

    numeric_cols = [c for c in comp.columns if c != "run"]
    score_candidates = [
        "mean_roc_auc",
        "mean_pr_auc",
        "mean_f1",
        "mean_r2",
        "mean_rmse",
        "mean_mae",
    ]
    chart_cols = [c for c in score_candidates if c in numeric_cols]
    if not chart_cols:
        chart_cols = numeric_cols[:4]

    fig, ax = plt.subplots(figsize=(max(7, len(comp) * 1.6), 5))
    x = np.arange(len(comp))
    width = 0.8 / max(1, len(chart_cols))
    for idx, col in enumerate(chart_cols):
        ax.bar(x + idx * width, comp[col].to_numpy(), width=width, label=col)
    ax.set_xticks(x + width * max(0, len(chart_cols) - 1) / 2)
    ax.set_xticklabels(comp["run"].tolist(), rotation=20, ha="right")
    ax.set_title("Run comparison")
    ax.set_ylabel("metric value")
    ax.legend()
    fig.tight_layout()
    chart_path = out_dir / "run_comparison.png"
    fig.savefig(chart_path, dpi=140)
    plt.close(fig)
    return csv_path, chart_path
