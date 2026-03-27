from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class FoldIndices:
    fold: int
    train_idx: list[int]
    valid_idx: list[int]
    train_groups: list[str]
    valid_groups: list[str]



def build_sessions_from_gap(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    gap_minutes: int = 60,
    session_col: str = "session_id",
) -> pd.DataFrame:
    """Infer session ids from a timestamp gap.

    Rows are sorted by timestamp and a new session starts whenever the gap to the
    previous game is strictly greater than `gap_minutes`.
    """
    work = df.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors="coerce")
    work = work.sort_values(timestamp_col).reset_index(drop=True)
    gap = work[timestamp_col].diff().dt.total_seconds().div(60)
    new_session = gap.isna() | (gap > gap_minutes)
    work[session_col] = new_session.cumsum().map(lambda x: f"session_{int(x):05d}")
    return work



def assert_monotonic_group_time(df: pd.DataFrame, *, group_col: str, time_col: str) -> None:
    work = df[[group_col, time_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    per_group = work.groupby(group_col, sort=False)[time_col].min().sort_values()
    if per_group.isna().any():
        bad = per_group[per_group.isna()].index.tolist()
        raise ValueError(f"Missing timestamps in groups: {bad[:5]}")



def iter_chronological_group_splits(
    df: pd.DataFrame,
    *,
    group_col: str,
    time_col: str,
    n_splits: int = 5,
    group_gap: int = 0,
) -> Iterator[FoldIndices]:
    """Chronological group split to reduce leakage in personal time-series data.

    Groups are ordered by their earliest timestamp. Validation groups always come
    strictly after training groups. Optional `group_gap` excludes recent train groups
    immediately before validation.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    work = df.reset_index(drop=True).copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    assert_monotonic_group_time(work, group_col=group_col, time_col=time_col)

    group_times = (
        work.groupby(group_col, sort=False)[time_col]
        .min()
        .sort_values()
    )
    ordered_groups = group_times.index.tolist()
    n_groups = len(ordered_groups)
    if n_groups < n_splits + 1:
        raise ValueError(
            f"Need at least {n_splits + 1} groups for {n_splits} chronological splits; got {n_groups}."
        )

    # Use an expanding-window schedule.
    for fold in range(1, n_splits + 1):
        valid_start = max(1, (n_groups * fold) // (n_splits + 1))
        valid_end = max(valid_start + 1, (n_groups * (fold + 1)) // (n_splits + 1))
        valid_groups = ordered_groups[valid_start:valid_end]
        train_end = max(0, valid_start - group_gap)
        train_groups = ordered_groups[:train_end]
        if not train_groups or not valid_groups:
            continue

        train_mask = work[group_col].isin(train_groups)
        valid_mask = work[group_col].isin(valid_groups)
        yield FoldIndices(
            fold=fold,
            train_idx=work.index[train_mask].tolist(),
            valid_idx=work.index[valid_mask].tolist(),
            train_groups=[str(x) for x in train_groups],
            valid_groups=[str(x) for x in valid_groups],
        )



def leakage_report(
    df: pd.DataFrame,
    *,
    train_idx: list[int],
    valid_idx: list[int],
    group_col: str,
    time_col: str,
) -> dict:
    work = df.reset_index(drop=True).copy()
    work[time_col] = pd.to_datetime(work[time_col], utc=True, errors="coerce")
    train = work.loc[train_idx]
    valid = work.loc[valid_idx]
    train_groups = set(train[group_col].astype(str).unique().tolist())
    valid_groups = set(valid[group_col].astype(str).unique().tolist())
    overlap = sorted(train_groups & valid_groups)
    return {
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_groups": int(len(train_groups)),
        "valid_groups": int(len(valid_groups)),
        "group_overlap_count": int(len(overlap)),
        "group_overlap_examples": overlap[:10],
        "max_train_time": None if train.empty else str(train[time_col].max()),
        "min_valid_time": None if valid.empty else str(valid[time_col].min()),
    }
