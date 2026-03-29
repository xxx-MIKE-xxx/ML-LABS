from __future__ import annotations

import numpy as np
import pandas as pd



def _reverse_cummax(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    running = -np.inf
    for i in range(len(values) - 1, -1, -1):
        running = max(running, float(values[i]))
        out[i] = running
    return out



def _reverse_cummin(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    running = np.inf
    for i in range(len(values) - 1, -1, -1):
        running = min(running, float(values[i]))
        out[i] = running
    return out



def _distance_to_earliest_future_max(values: np.ndarray) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=float)
    running_max = -np.inf
    running_idx = n - 1
    for i in range(n - 1, -1, -1):
        v = float(values[i])
        if v >= running_max:
            running_max = v
            running_idx = i
        out[i] = float(running_idx - i)
    return out



def add_target(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    objective_kind: str,
    min_gain_elo: float = 8.0,
) -> tuple[pd.DataFrame, str, str]:
    work = df.sort_values([group_col, order_col]).copy()
    target_col = f"target__{objective_kind}"

    if objective_kind == "peak_binary":
        work[target_col] = 0
        for _, g in work.groupby(group_col, sort=False):
            idx = g.index.to_list()
            vals = g[cum_pl_col].to_numpy(dtype=float)
            peak = np.max(vals)
            first_peak_pos = int(np.where(vals == peak)[0][0])
            work.loc[idx[first_peak_pos], target_col] = 1
        return work, target_col, "classification"

    targets = []
    for _, g in work.groupby(group_col, sort=False):
        vals = g[cum_pl_col].to_numpy(dtype=float)
        if objective_kind == "meaningful_upside_binary":
            future_upside = _reverse_cummax(vals) - vals
            target_vals = (future_upside >= float(min_gain_elo)).astype(int)
            task_type = "classification"
        elif objective_kind == "future_upside_regression":
            target_vals = _reverse_cummax(vals) - vals
            task_type = "regression"
        elif objective_kind == "future_drawdown_regression":
            target_vals = vals - _reverse_cummin(vals)
            task_type = "regression"
        elif objective_kind == "games_to_peak_regression":
            target_vals = _distance_to_earliest_future_max(vals)
            task_type = "regression"
        else:
            raise ValueError(f"Unsupported objective_kind: {objective_kind}")
        targets.append(pd.Series(target_vals, index=g.index))

    work[target_col] = pd.concat(targets).sort_index()
    return work, target_col, task_type
