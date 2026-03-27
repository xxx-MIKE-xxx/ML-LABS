from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class BacktestSummary:
    n_sessions: int
    mean_realized_pl: float
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
    work = df.sort_values([group_col, order_col]).copy()
    realized = []
    peaks = []
    stops = []

    for _, g in work.groupby(group_col, sort=False):
        g = g.reset_index(drop=True)
        peak_pl = float(g[cum_pl_col].max())
        stop_positions = g.index[g[action_col] == "stop"].tolist()
        stop_idx = int(stop_positions[0]) if stop_positions else int(len(g) - 1)
        realized_pl = float(g.loc[stop_idx, cum_pl_col])
        peaks.append(peak_pl)
        realized.append(realized_pl)
        stops.append(stop_idx + 1)

    realized_arr = np.asarray(realized, dtype=float)
    peaks_arr = np.asarray(peaks, dtype=float)
    capture = np.divide(
        realized_arr,
        peaks_arr,
        out=np.zeros_like(realized_arr),
        where=peaks_arr != 0,
    )
    return BacktestSummary(
        n_sessions=int(len(realized)),
        mean_realized_pl=float(realized_arr.mean()) if len(realized_arr) else 0.0,
        mean_peak_pl=float(peaks_arr.mean()) if len(peaks_arr) else 0.0,
        mean_capture_ratio=float(capture.mean()) if len(capture) else 0.0,
        mean_stop_game_index=float(np.mean(stops)) if stops else 0.0,
    )
