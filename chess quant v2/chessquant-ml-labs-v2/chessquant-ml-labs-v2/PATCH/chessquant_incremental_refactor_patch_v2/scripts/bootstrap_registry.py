from __future__ import annotations

from pathlib import Path
import polars as pl

from chessquant_ml.settings import settings
from chessquant_ml.data.registry import empty_registry, save_registry, upsert_registry


def _scan_json(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    if path.suffix.lower() == ".json":
        return pl.read_json(path)
    raise ValueError(f"Unsupported input file: {path}")


def main() -> None:
    settings.ensure_dirs()
    registry = empty_registry()

    raw_files = list(settings.raw_dir.glob("*.json"))
    for raw_file in raw_files:
        raw_df = _scan_json(raw_file)
        if raw_df.is_empty() or "game_id" not in raw_df.columns:
            continue
        created = pl.col("created_at_ms") if "created_at_ms" in raw_df.columns else pl.lit(None).cast(pl.Int64).alias("created_at_ms")
        last_move = pl.col("last_move_at_ms") if "last_move_at_ms" in raw_df.columns else pl.lit(None).cast(pl.Int64).alias("last_move_at_ms")
        has_lichess = pl.col("has_lichess_eval") if "has_lichess_eval" in raw_df.columns else pl.lit(False)
        updates = raw_df.select([
            pl.col("game_id"),
            created,
            last_move,
            pl.lit(True).alias("raw_present"),
            pl.lit(False).alias("enriched_present"),
            pl.lit(False).alias("features_present"),
            has_lichess.cast(pl.Boolean).alias("has_lichess_eval"),
            pl.lit(None).cast(pl.Utf8).alias("engine_version"),
            pl.lit(None).cast(pl.Utf8).alias("enriched_version"),
            pl.lit(None).cast(pl.Utf8).alias("feature_version"),
        ])
        registry = upsert_registry(registry, updates)

    enriched_files = list(settings.enriched_dir.glob("*.json"))
    for enriched_file in enriched_files:
        enriched_df = _scan_json(enriched_file)
        if enriched_df.is_empty() or "game_id" not in enriched_df.columns:
            continue
        updates = enriched_df.select([
            pl.col("game_id"),
            (pl.col("created_at_ms") if "created_at_ms" in enriched_df.columns else pl.lit(None).cast(pl.Int64).alias("created_at_ms")),
            (pl.col("last_move_at_ms") if "last_move_at_ms" in enriched_df.columns else pl.lit(None).cast(pl.Int64).alias("last_move_at_ms")),
            pl.lit(True).alias("raw_present"),
            pl.lit(True).alias("enriched_present"),
            pl.lit(False).alias("features_present"),
            (pl.col("has_lichess_eval") if "has_lichess_eval" in enriched_df.columns else pl.lit(False)).cast(pl.Boolean).alias("has_lichess_eval"),
            pl.lit("unknown").alias("engine_version"),
            pl.lit("bootstrap_v1").alias("enriched_version"),
            pl.lit(None).cast(pl.Utf8).alias("feature_version"),
        ])
        registry = upsert_registry(registry, updates)

    feature_files = list(settings.features_dir.glob("*.parquet"))
    for feature_file in feature_files:
        feat_df = pl.read_parquet(feature_file)
        if feat_df.is_empty() or "game_id" not in feat_df.columns:
            continue
        updates = feat_df.select([
            pl.col("game_id"),
            (pl.col("created_at_ms") if "created_at_ms" in feat_df.columns else pl.lit(None).cast(pl.Int64).alias("created_at_ms")),
            (pl.col("last_move_at_ms") if "last_move_at_ms" in feat_df.columns else pl.lit(None).cast(pl.Int64).alias("last_move_at_ms")),
            pl.lit(True).alias("raw_present"),
            pl.lit(True).alias("enriched_present"),
            pl.lit(True).alias("features_present"),
            (pl.col("has_lichess_eval") if "has_lichess_eval" in feat_df.columns else pl.lit(False)).cast(pl.Boolean).alias("has_lichess_eval"),
            pl.lit("unknown").alias("engine_version"),
            pl.lit("bootstrap_v1").alias("enriched_version"),
            pl.lit("bootstrap_v1").alias("feature_version"),
        ])
        registry = upsert_registry(registry, updates)

    out_path = settings.registry_dir / "game_index.parquet"
    save_registry(registry, out_path)
    print(f"Registry written: {out_path}")
    print(f"Rows: {registry.height}")


if __name__ == "__main__":
    main()
