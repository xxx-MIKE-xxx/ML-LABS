from __future__ import annotations

from pathlib import Path

import polars as pl

from chessquant_ml.data.registry import load_registry, save_registry, upsert_registry
from chessquant_ml.settings import settings


REGISTRY_PATH = settings.registry_dir / "game_index.parquet"


def registry_path() -> Path:
    return REGISTRY_PATH


def load_game_registry() -> pl.DataFrame:
    return load_registry(REGISTRY_PATH)


def save_game_registry(df: pl.DataFrame) -> None:
    save_registry(df, REGISTRY_PATH)


def mark_raw_games(raw_df: pl.DataFrame) -> pl.DataFrame:
    cols = []
    if "game_id" not in raw_df.columns:
        raise ValueError("raw_df must contain game_id")
    cols.append(pl.col("game_id"))
    cols.append(pl.col("created_at_ms") if "created_at_ms" in raw_df.columns else pl.lit(None).cast(pl.Int64).alias("created_at_ms"))
    cols.append(pl.col("last_move_at_ms") if "last_move_at_ms" in raw_df.columns else pl.lit(None).cast(pl.Int64).alias("last_move_at_ms"))
    cols.extend([
        pl.lit(True).alias("raw_present"),
        pl.lit(False).alias("enriched_present"),
        pl.lit(False).alias("features_present"),
        (pl.col("has_lichess_eval") if "has_lichess_eval" in raw_df.columns else pl.lit(False)).cast(pl.Boolean).alias("has_lichess_eval"),
        pl.lit(None).cast(pl.Utf8).alias("engine_version"),
        pl.lit(None).cast(pl.Utf8).alias("enriched_version"),
        pl.lit(None).cast(pl.Utf8).alias("feature_version"),
    ])
    updates = raw_df.select(cols)
    current = load_game_registry()
    merged = upsert_registry(current, updates)
    save_game_registry(merged)
    return merged


def select_games_needing_enrichment(raw_df: pl.DataFrame) -> pl.DataFrame:
    registry = load_game_registry()
    if registry.is_empty():
        return raw_df
    need_ids = registry.filter(~pl.col("enriched_present")).select("game_id")
    if need_ids.is_empty():
        return raw_df.head(0)
    return raw_df.join(need_ids, on="game_id", how="inner")
