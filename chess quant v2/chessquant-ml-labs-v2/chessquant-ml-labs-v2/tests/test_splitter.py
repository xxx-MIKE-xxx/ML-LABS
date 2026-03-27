from __future__ import annotations

import pandas as pd

from chessquant_ml.data.splitter import build_sessions_from_gap, iter_chronological_group_splits



def test_build_sessions_from_gap() -> None:
    df = pd.DataFrame(
        {
            "created_at": [
                "2026-01-01T10:00:00Z",
                "2026-01-01T10:30:00Z",
                "2026-01-01T12:15:00Z",
            ]
        }
    )
    out = build_sessions_from_gap(df, timestamp_col="created_at", gap_minutes=60)
    assert out["session_id"].nunique() == 2



def test_iter_chronological_group_splits_has_no_overlap() -> None:
    df = pd.DataFrame(
        {
            "session_id": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4", "s5", "s5", "s6", "s6"],
            "created_at": pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC"),
            "x": range(12),
        }
    )
    splits = list(iter_chronological_group_splits(df, group_col="session_id", time_col="created_at", n_splits=3))
    assert splits
    for split in splits:
        train_groups = set(df.iloc[split.train_idx]["session_id"].tolist())
        valid_groups = set(df.iloc[split.valid_idx]["session_id"].tolist())
        assert train_groups.isdisjoint(valid_groups)
