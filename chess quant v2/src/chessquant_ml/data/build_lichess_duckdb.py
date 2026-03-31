from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import duckdb


def log(msg: str) -> None:
    print(msg, flush=True)


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DuckDB database over one or more folders of Lichess parquet shards. "
            "Creates a raw view plus lightweight helper tables/views for downstream analysis."
        )
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        default=[],
        help=(
            "Directory containing parquet shards. Repeat this flag to include multiple "
            "download-batch folders."
        ),
    )
    parser.add_argument(
        "--input-glob",
        action="append",
        default=[],
        help=(
            "Glob pattern for parquet shards, e.g. "
            "'data/raw/lichess_db/**/*.parquet'. Repeatable."
        ),
    )
    parser.add_argument(
        "--db-path",
        default="data/cache/lichess.duckdb",
        help="Output DuckDB file.",
    )
    parser.add_argument(
        "--raw-view-name",
        default="games_raw",
        help="Name of the raw parquet-backed DuckDB view.",
    )
    parser.add_argument(
        "--catalog-table-name",
        default="games_catalog",
        help="Name of the lightweight catalog table.",
    )
    parser.add_argument(
        "--player-view-name",
        default="player_games_long",
        help="Name of the long-format player-game view.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing objects if they already exist.",
    )
    return parser.parse_args()


def collect_parquet_files(input_dirs: list[str], input_globs: list[str]) -> list[Path]:
    files: list[Path] = []

    for dir_str in input_dirs:
        root = Path(dir_str)
        if not root.exists():
            raise FileNotFoundError(f"Input dir not found: {root}")
        files.extend(root.rglob("*.parquet"))

    for pattern in input_globs:
        files.extend(Path().glob(pattern))

    deduped = sorted({p.resolve() for p in files if p.is_file()})
    if not deduped:
        raise RuntimeError("No parquet files found from the provided input dirs/globs.")
    return deduped


def parquet_sql_list(paths: Iterable[Path]) -> str:
    return "[" + ", ".join(sql_str(p.as_posix()) for p in paths) + "]"


def existing_columns(
    con: duckdb.DuckDBPyConnection,
    parquet_paths: list[Path],
) -> list[str]:
    list_sql = parquet_sql_list(parquet_paths)
    rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet({list_sql}, union_by_name = true, filename = true)
        """
    ).fetchall()
    return [row[0] for row in rows]


def pick_first(existing: set[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in existing:
            return c
    return None


def maybe_expr(alias: str, existing: set[str], candidates: list[str]) -> str | None:
    col = pick_first(existing, candidates)
    if col is None:
        return None
    return f"{quote_ident(col)} AS {quote_ident(alias)}"


def create_raw_view(
    con: duckdb.DuckDBPyConnection,
    parquet_paths: list[Path],
    raw_view_name: str,
) -> None:
    list_sql = parquet_sql_list(parquet_paths)
    con.execute(f"DROP VIEW IF EXISTS {quote_ident(raw_view_name)}")
    con.execute(
        f"""
        CREATE VIEW {quote_ident(raw_view_name)} AS
        SELECT *
        FROM read_parquet({list_sql}, union_by_name = true, filename = true)
        """
    )


def build_catalog_select(existing: set[str], raw_view_name: str) -> str:
    exprs: list[str] = []

    exprs.append("filename AS source_file")

    mappings = {
        "game_id": ["id", "game_id", "gameId"],
        "white_player": ["white", "white_player", "whiteName", "white_name"],
        "black_player": ["black", "black_player", "blackName", "black_name"],
        "white_elo": ["whiteElo", "white_elo", "white_rating"],
        "black_elo": ["blackElo", "black_elo", "black_rating"],
        "winner": ["winner"],
        "victory_status": ["victory_status", "status", "termination"],
        "rated": ["rated"],
        "speed": ["speed"],
        "opening": ["opening", "opening_name"],
        "moves": ["moves", "pgn_moves"],
        "utc_date": ["UTCDate", "utc_date", "utcDate", "date"],
        "created_at": ["createdAt", "created_at", "created_at_ms"],
        "last_move_at": ["lastMoveAt", "last_move_at", "last_move_at_ms"],
    }

    for alias, candidates in mappings.items():
        expr = maybe_expr(alias, existing, candidates)
        if expr is not None:
            exprs.append(expr)

    if "moves" in existing:
        exprs.append("length(moves) AS moves_text_len")
    elif "pgn_moves" in existing:
        exprs.append("length(pgn_moves) AS moves_text_len")

    if not exprs:
        exprs = ["*"]

    return (
        "SELECT\n    "
        + ",\n    ".join(exprs)
        + f"\nFROM {quote_ident(raw_view_name)}"
    )


def create_catalog_table(
    con: duckdb.DuckDBPyConnection,
    existing: set[str],
    raw_view_name: str,
    catalog_table_name: str,
) -> None:
    con.execute(f"DROP TABLE IF EXISTS {quote_ident(catalog_table_name)}")

    con.execute(
        f"""
        CREATE TABLE {quote_ident(catalog_table_name)} AS
        SELECT
            md5(
                coalesce(filename, '') || '|' ||
                coalesce(Event, '') || '|' ||
                coalesce(Site, '') || '|' ||
                coalesce(White, '') || '|' ||
                coalesce(Black, '') || '|' ||
                coalesce(CAST(UTCDate AS VARCHAR), '') || '|' ||
                coalesce(CAST(UTCTime AS VARCHAR), '') || '|' ||
                coalesce(movetext, '')
            ) AS game_id,

            White AS white_player,
            Black AS black_player,
            CAST(WhiteElo AS DOUBLE) AS white_elo,
            CAST(BlackElo AS DOUBLE) AS black_elo,

            Result AS result,

            CASE
                WHEN Result = '1-0' THEN 'white'
                WHEN Result = '0-1' THEN 'black'
                ELSE 'draw_or_other'
            END AS winner,

            UTCDate AS utc_date,
            CAST(UTCTime AS VARCHAR) AS utc_time_text,

            TRY_CAST(
                CAST(UTCDate AS VARCHAR) || ' ' || CAST(UTCTime AS VARCHAR)
                AS TIMESTAMP WITH TIME ZONE
            ) AS game_ts_tz,

            TRY_CAST(
                CAST(UTCDate AS VARCHAR) || ' ' || split_part(CAST(UTCTime AS VARCHAR), '+', 1)
                AS TIMESTAMP
            ) AS game_ts,

            ECO AS eco,
            Opening AS opening,
            Termination AS victory_status,
            TimeControl AS time_control,
            movetext AS moves,
            filename AS source_file
        FROM {quote_ident(raw_view_name)}
        """
    )


def create_player_view(
    con: duckdb.DuckDBPyConnection,
    catalog_table_name: str,
    player_view_name: str,
    catalog_columns: set[str],
) -> None:
    required = {"game_id", "white_player", "black_player"}
    if not required.issubset(catalog_columns):
        log(
            "Skipping player long view because games_catalog does not contain all of: "
            "game_id, white_player, black_player"
        )
        return

    con.execute(f"DROP VIEW IF EXISTS {quote_ident(player_view_name)}")
    con.execute(
        f"""
        CREATE VIEW {quote_ident(player_view_name)} AS
        SELECT
            game_id,
            white_player AS player_name,
            'white' AS side,
            white_elo AS player_elo,
            black_elo AS opponent_elo,
            result,
            winner,
            utc_date,
            utc_time_text,
            game_ts,
            opening,
            eco,
            victory_status,
            time_control,
            source_file
        FROM {quote_ident(catalog_table_name)}
        WHERE white_player IS NOT NULL

        UNION ALL

        SELECT
            game_id,
            black_player AS player_name,
            'black' AS side,
            black_elo AS player_elo,
            white_elo AS opponent_elo,
            result,
            winner,
            utc_date,
            utc_time_text,
            game_ts,
            opening,
            eco,
            victory_status,
            time_control,
            source_file
        FROM {quote_ident(catalog_table_name)}
        WHERE black_player IS NOT NULL
        """
    )


def write_build_metadata(
    con: duckdb.DuckDBPyConnection,
    parquet_paths: list[Path],
) -> None:
    con.execute("DROP TABLE IF EXISTS build_metadata")
    con.execute(
        """
        CREATE TABLE build_metadata (
            built_at_utc VARCHAR,
            parquet_file_count BIGINT,
            parquet_files_json VARCHAR
        )
        """
    )
    payload = json.dumps([p.as_posix() for p in parquet_paths], indent=2)
    con.execute(
        """
        INSERT INTO build_metadata VALUES (?, ?, ?)
        """,
        [
            datetime.now(timezone.utc).isoformat(),
            len(parquet_paths),
            payload,
        ],
    )


def main() -> int:
    args = parse_args()

    parquet_paths = collect_parquet_files(args.input_dir, args.input_glob)
    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Found {len(parquet_paths)} parquet files.")
    log(f"Building DuckDB at: {db_path}")

    con = duckdb.connect(str(db_path))

    cols = existing_columns(con, parquet_paths)
    cols_set = set(cols)

    create_raw_view(con, parquet_paths, args.raw_view_name)
    log(f"Created raw view: {args.raw_view_name}")

    create_catalog_table(
        con,
        cols_set,
        args.raw_view_name,
        args.catalog_table_name,
    )
    log(f"Created catalog table: {args.catalog_table_name}")

    catalog_cols = {
        row[1]
        for row in con.execute(
            f"PRAGMA table_info({quote_ident(args.catalog_table_name)})"
        ).fetchall()
    }

    create_player_view(
        con,
        args.catalog_table_name,
        args.player_view_name,
        catalog_cols,
    )

    write_build_metadata(con, parquet_paths)

    summary = con.execute(
        f"""
        SELECT COUNT(*) AS n_rows
        FROM {quote_ident(args.catalog_table_name)}
        """
    ).fetchone()

    log(f"Catalog row count: {summary[0]}")
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())