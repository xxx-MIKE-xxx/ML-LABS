from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


REGISTRY_COLUMNS = {
    "game_id": pl.Utf8,
    "created_at_ms": pl.Int64,
    "last_move_at_ms": pl.Int64,
    "raw_present": pl.Boolean,
    "enriched_present": pl.Boolean,
    "features_present": pl.Boolean,
    "has_lichess_eval": pl.Boolean,
    "engine_version": pl.Utf8,
    "enriched_version": pl.Utf8,
    "feature_version": pl.Utf8,
}


def empty_registry() -> pl.DataFrame:
    return pl.DataFrame(schema=REGISTRY_COLUMNS)


def load_registry(path: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    return empty_registry()


def save_registry(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def upsert_registry(existing: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        out = updates
    else:
        out = pl.concat([existing, updates], how="diagonal_relaxed")
        out = out.sort(["game_id", "created_at_ms"]).group_by("game_id").tail(1)
    # ensure stable column order
    cols = list(REGISTRY_COLUMNS.keys())
    for col, dtype in REGISTRY_COLUMNS.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return out.select(cols)


def latest_seen_created_at_ms(registry: pl.DataFrame) -> int | None:
    if registry.is_empty() or "created_at_ms" not in registry.columns:
        return None
    value = registry.select(pl.col("created_at_ms").max()).item()
    return int(value) if value is not None else None
