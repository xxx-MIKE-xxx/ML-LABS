from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    n_sessions: int
    n_stop_sessions: int
    total_actual_final_pl: float
    total_stop_pl: float
    total_peak_pl: float
    total_elo_delta_vs_actual: float
    total_elo_left_vs_peak: float
    mean_realized_pl: float
    mean_actual_final_pl: float
    mean_peak_pl: float
    mean_capture_ratio: float
    mean_stop_game_index: float


def backtest_stop_policy(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    action_col: str,
) -> BacktestSummary:
    _, summary = backtest_stop_policy_detailed(
        df,
        group_col=group_col,
        order_col=order_col,
        cum_pl_col=cum_pl_col,
        action_col=action_col,
    )
    return summary


def backtest_stop_policy_detailed(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    action_col: str,
) -> tuple[pd.DataFrame, BacktestSummary]:
    work = df.sort_values([group_col, order_col]).copy()
    records: list[dict] = []

    for session_id, g in work.groupby(group_col, sort=False):
        g = g.reset_index(drop=True)
        stop_positions = g.index[g[action_col] == "stop"].tolist()
        stop_triggered = bool(stop_positions)
        stop_idx = int(stop_positions[0]) if stop_positions else int(len(g) - 1)

        peak_pl = float(pd.to_numeric(g[cum_pl_col], errors="coerce").max())
        actual_final_pl = float(pd.to_numeric(g[cum_pl_col], errors="coerce").iloc[-1])
        stop_pl = float(pd.to_numeric(g[cum_pl_col], errors="coerce").iloc[stop_idx])

        capture_ratio = np.nan
        if peak_pl != 0:
            capture_ratio = stop_pl / peak_pl

        records.append(
            {
                "session_id": str(session_id),
                "n_games": int(len(g)),
                "stop_triggered": bool(stop_triggered),
                "stop_game_index": int(stop_idx + 1),
                "actual_final_game_index": int(len(g)),
                "peak_game_index": int(pd.to_numeric(g[cum_pl_col], errors="coerce").idxmax() + 1),
                "stop_pl": float(stop_pl),
                "actual_final_pl": float(actual_final_pl),
                "peak_pl": float(peak_pl),
                "elo_delta_vs_actual": float(stop_pl - actual_final_pl),
                "elo_left_vs_peak": float(peak_pl - stop_pl),
                "capture_ratio": float(capture_ratio) if not np.isnan(capture_ratio) else None,
            }
        )

    session_df = pd.DataFrame(records)
    if session_df.empty:
        summary = BacktestSummary(
            n_sessions=0,
            n_stop_sessions=0,
            total_actual_final_pl=0.0,
            total_stop_pl=0.0,
            total_peak_pl=0.0,
            total_elo_delta_vs_actual=0.0,
            total_elo_left_vs_peak=0.0,
            mean_realized_pl=0.0,
            mean_actual_final_pl=0.0,
            mean_peak_pl=0.0,
            mean_capture_ratio=0.0,
            mean_stop_game_index=0.0,
        )
        return session_df, summary

    capture = pd.to_numeric(session_df["capture_ratio"], errors="coerce")
    summary = BacktestSummary(
        n_sessions=int(len(session_df)),
        n_stop_sessions=int(session_df["stop_triggered"].sum()),
        total_actual_final_pl=float(session_df["actual_final_pl"].sum()),
        total_stop_pl=float(session_df["stop_pl"].sum()),
        total_peak_pl=float(session_df["peak_pl"].sum()),
        total_elo_delta_vs_actual=float(session_df["elo_delta_vs_actual"].sum()),
        total_elo_left_vs_peak=float(session_df["elo_left_vs_peak"].sum()),
        mean_realized_pl=float(session_df["stop_pl"].mean()),
        mean_actual_final_pl=float(session_df["actual_final_pl"].mean()),
        mean_peak_pl=float(session_df["peak_pl"].mean()),
        mean_capture_ratio=float(capture.dropna().mean()) if capture.notna().any() else 0.0,
        mean_stop_game_index=float(session_df["stop_game_index"].mean()),
    )
    return session_df, summary
