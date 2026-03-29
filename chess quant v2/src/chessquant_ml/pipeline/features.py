from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from tqdm.auto import tqdm

from chessquant_ml.settings import settings
from chessquant_ml.utils.io import read_json, write_parquet
from chessquant_ml.utils.time import assign_time_of_day, to_local_hour


def _result_for_user(game: dict[str, Any], user_color: str) -> float:
    winner = game.get("winner")
    if winner == user_color:
        return 1.0
    if winner is None:
        return 0.5
    return 0.0


def _rating_diff_for_user(game: dict[str, Any], user_color: str) -> float:
    try:
        return float(game.get("players", {}).get(user_color, {}).get("ratingDiff", 0) or 0)
    except Exception:
        return 0.0


def _move_count(game: dict[str, Any]) -> int:
    moves = str(game.get("moves", "") or "")
    return max(1, len([m for m in moves.split() if m]))


def _row_from_game(game: dict[str, Any]) -> dict[str, Any]:
    hero = settings.lichess_username.lower()
    white_name = str(game.get("players", {}).get("white", {}).get("user", {}).get("name", "")).lower()
    user_color = "white" if white_name == hero else "black"
    created_at = int(game.get("createdAt", 0) or 0)
    last_move_at = int(game.get("lastMoveAt", created_at) or created_at)
    move_count = _move_count(game)
    duration_sec = max(0.0, (last_move_at - created_at) / 1000.0)
    avg_secs = duration_sec / max(1, move_count // 2)
    engine = game.get("cq_engine", {}) or {}
    local_hour = to_local_hour(created_at, settings.local_timezone)
    tod = assign_time_of_day(local_hour)

    return {
        "game_id": str(game.get("id", "")),
        "created_at_ms": created_at,
        "last_move_at_ms": last_move_at,
        "user_color": user_color,
        "my_acpl": float(engine.get("my_acpl", 0) or 0),
        "my_blunder_count": float(engine.get("my_blunder_count", 0) or 0),
        "my_mistake_count": float(engine.get("my_mistake_count", 0) or 0),
        "my_inaccuracy_count": float(engine.get("my_inaccuracy_count", 0) or 0),
        "largest_eval_drop_cp": float(engine.get("largest_eval_drop_cp", 0) or 0),
        "avg_eval_drop_cp": float(engine.get("avg_eval_drop_cp", 0) or 0),
        "eval_volatility": float(engine.get("eval_volatility", 0) or 0),
        "threw_winning_position": int(engine.get("threw_winning_position", 0) or 0),
        "engine_source": str(engine.get("engine_source", "unknown")),
        "rating_diff": _rating_diff_for_user(game, user_color),
        "result": _result_for_user(game, user_color),
        "my_avg_secs_per_move": float(avg_secs),
        "tod_label": tod,
    }


def _with_group_features(df: pl.DataFrame) -> pl.DataFrame:
    session_gap_ms = settings.session_gap_minutes * 60 * 1000
    df = df.sort("created_at_ms")
    df = df.with_columns(
        [
            (pl.col("created_at_ms") - pl.col("created_at_ms").shift(1)).alias("time_gap_ms"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("time_gap_ms") > session_gap_ms) | pl.col("time_gap_ms").is_null())
            .cast(pl.Int64)
            .cum_sum()
            .alias("session_id")
        ]
    )
    df = df.with_columns(
        [
            pl.int_range(0, pl.len()).over("session_id").add(1).alias("games_played"),
            pl.col("rating_diff").cum_sum().over("session_id").alias("session_pl"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("my_avg_secs_per_move").first().over("session_id").alias("first_speed"),
            (pl.col("result") == 0.0).cast(pl.Int64).alias("is_loss"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("my_avg_secs_per_move") / (pl.col("first_speed") + 0.001)).alias("speed_vs_start"),
            pl.col("last_move_at_ms").shift(1).over("session_id").alias("prev_game_end_ms"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("created_at_ms") - pl.col("prev_game_end_ms")).fill_null(0) / 1000.0)
            .clip(lower_bound=0)
            .alias("break_time_sec"),
            pl.when(pl.col("result") == 0.0)
            .then(
                pl.int_range(0, pl.len()).over((pl.col("result") != 0.0).cast(pl.Int64).cum_sum())
            )
            .otherwise(0)
            .alias("loss_streak"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("my_acpl").rolling_mean(window_size=5, min_samples=1).over("session_id").alias("roll_5_acpl_mean"),
            pl.col("my_avg_secs_per_move").rolling_mean(window_size=5, min_samples=1).over("session_id").alias("roll_5_time_per_move"),
            (pl.col("break_time_sec") + 1).log().alias("log_break_time"),
        ]
    )
    for label in ["morning", "midday", "evening", "night"]:
        df = df.with_columns((pl.col("tod_label") == label).cast(pl.Int8).alias(f"tod_{label}"))
    return df


def _attach_target(df: pl.DataFrame) -> pl.DataFrame:
    session_max = df.group_by("session_id").agg(pl.max("session_pl").alias("session_max"))
    df = df.join(session_max, on="session_id", how="left")
    df = df.with_columns((pl.col("session_pl") == pl.col("session_max")).alias("is_max"))
    first_max = (
        df.filter(pl.col("is_max"))
        .group_by("session_id")
        .agg(pl.min("games_played").alias("first_max_game"))
    )
    df = df.join(first_max, on="session_id", how="left")
    df = df.with_columns(
        (pl.col("games_played") == pl.col("first_max_game")).cast(pl.Int8).alias("target")
    )
    return df


def build_feature_table(enriched_path: Path, out_path: Path) -> Path:
    games: list[dict[str, Any]] = read_json(enriched_path)

    rows: list[dict[str, Any]] = []
    for game in tqdm(games, desc="Building feature rows", unit="game"):
        rows.append(_row_from_game(game))

    df = pl.DataFrame(rows)
    df = _with_group_features(df)
    df = _attach_target(df)
    write_parquet(df, out_path)
    return out_path