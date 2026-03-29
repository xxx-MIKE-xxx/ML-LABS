from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    n_sessions: int
    n_stop_sessions: int
    n_margin_calls: int              # NEW: Sessions halted by risk limits, not the model
    total_actual_final_pl: float
    total_stop_pl: float
    total_peak_pl: float
    total_tapl: float                # NEW: Tilt-Adjusted P/L (Business North Star)
    total_elo_delta_vs_actual: float
    total_elo_left_vs_peak: float
    mean_realized_pl: float
    mean_actual_final_pl: float
    mean_peak_pl: float
    mean_tapl: float                 # NEW: Mean Tilt-Adjusted P/L
    mean_capture_ratio: float
    mean_stop_game_index: float


def backtest_stop_policy(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    action_col: str,
    max_drawdown: float = -50.0,     # NEW: Threshold for Simulated Margin Call
    pain_multiplier: float = 2.5,    # NEW: Beta for Prospect Theory penalty
) -> BacktestSummary:
    _, summary = backtest_stop_policy_detailed(
        df,
        group_col=group_col,
        order_col=order_col,
        cum_pl_col=cum_pl_col,
        action_col=action_col,
        max_drawdown=max_drawdown,
        pain_multiplier=pain_multiplier
    )
    return summary


def backtest_stop_policy_detailed(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    action_col: str,
    max_drawdown: float = -50.0,
    pain_multiplier: float = 2.5,
) -> tuple[pd.DataFrame, BacktestSummary]:
    work = df.sort_values([group_col, order_col]).copy()
    records: list[dict] = []

    for session_id, g in work.groupby(group_col, sort=False):
        g = g.reset_index(drop=True)
        
        # 1. Identify Model-recommended stops
        action_stops = g.index[g[action_col] == "stop"].tolist()
        
        # 2. Identify Simulated Margin Calls (Ghost Tilt overrides)
        cum_pl_series = pd.to_numeric(g[cum_pl_col], errors="coerce")
        margin_calls = g.index[cum_pl_series <= max_drawdown].tolist()

        # 3. Determine the *effective* stop point (whichever happens first)
        action_idx = int(action_stops[0]) if action_stops else int(len(g) - 1)
        mc_idx = int(margin_calls[0]) if margin_calls else int(len(g) - 1)
        
        stop_idx = min(action_idx, mc_idx)
        
        # Flags for reporting
        margin_triggered = bool(margin_calls and stop_idx == mc_idx)
        stop_triggered = bool(action_stops and stop_idx == action_idx and not margin_triggered)

        # Calculate standard metrics
        peak_pl = float(cum_pl_series.max())
        actual_final_pl = float(cum_pl_series.iloc[-1])
        stop_pl = float(cum_pl_series.iloc[stop_idx])

        # NEW: Calculate Tilt-Adjusted P/L (TAPL)
        tapl = float(stop_pl) if stop_pl >= 0 else float(stop_pl * pain_multiplier)

        capture_ratio = np.nan
        if peak_pl != 0:
            capture_ratio = stop_pl / peak_pl

        records.append(
            {
                "session_id": str(session_id),
                "n_games": int(len(g)),
                "stop_triggered": bool(stop_triggered),
                "margin_triggered": bool(margin_triggered), # Logs if the user was saved by a margin call
                "stop_game_index": int(stop_idx + 1),
                "actual_final_game_index": int(len(g)),
                "peak_game_index": int(cum_pl_series.idxmax() + 1),
                "stop_pl": float(stop_pl),
                "tapl": float(tapl),                        # The new North Star score for this session
                "actual_final_pl": float(actual_final_pl),
                "peak_pl": float(peak_pl),
                "elo_delta_vs_actual": float(stop_pl - actual_final_pl), # Native Opportunity Cost
                "elo_left_vs_peak": float(peak_pl - stop_pl),
                "capture_ratio": float(capture_ratio) if not np.isnan(capture_ratio) else None,
            }
        )

    session_df = pd.DataFrame(records)
    
    if session_df.empty:
        summary = BacktestSummary(
            n_sessions=0,
            n_stop_sessions=0,
            n_margin_calls=0,
            total_actual_final_pl=0.0,
            total_stop_pl=0.0,
            total_peak_pl=0.0,
            total_tapl=0.0,
            total_elo_delta_vs_actual=0.0,
            total_elo_left_vs_peak=0.0,
            mean_realized_pl=0.0,
            mean_actual_final_pl=0.0,
            mean_peak_pl=0.0,
            mean_tapl=0.0,
            mean_capture_ratio=0.0,
            mean_stop_game_index=0.0,
        )
        return session_df, summary

    capture = pd.to_numeric(session_df["capture_ratio"], errors="coerce")
    summary = BacktestSummary(
        n_sessions=int(len(session_df)),
        n_stop_sessions=int(session_df["stop_triggered"].sum()),
        n_margin_calls=int(session_df["margin_triggered"].sum()),
        total_actual_final_pl=float(session_df["actual_final_pl"].sum()),
        total_stop_pl=float(session_df["stop_pl"].sum()),
        total_peak_pl=float(session_df["peak_pl"].sum()),
        total_tapl=float(session_df["tapl"].sum()),
        total_elo_delta_vs_actual=float(session_df["elo_delta_vs_actual"].sum()),
        total_elo_left_vs_peak=float(session_df["elo_left_vs_peak"].sum()),
        mean_realized_pl=float(session_df["stop_pl"].mean()),
        mean_actual_final_pl=float(session_df["actual_final_pl"].mean()),
        mean_peak_pl=float(session_df["peak_pl"].mean()),
        mean_tapl=float(session_df["tapl"].mean()),
        mean_capture_ratio=float(capture.dropna().mean()) if capture.notna().any() else 0.0,
        mean_stop_game_index=float(session_df["stop_game_index"].mean()),
    )
    return session_df, summary