from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a diverse player subset from the Lichess DuckDB catalog "
            "for downstream analysis and modeling."
        )
    )
    parser.add_argument(
        "--db-path",
        default="data/cache/lichess.duckdb",
        help="Path to DuckDB database built from Lichess parquet shards.",
    )
    parser.add_argument(
        "--source-table",
        default="games_catalog",
        help="Catalog table/view to read from.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/enriched/player_selection",
        help="Directory for selected players, stats, and selected games subset.",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=50,
        help="Minimum number of games per player.",
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=5,
        help="Minimum number of separate sessions per player.",
    )
    parser.add_argument(
        "--session-gap-hours",
        type=int,
        default=24,
        help="Gap that starts a new session.",
    )
    parser.add_argument(
        "--min-elo",
        type=int,
        default=1000,
        help="Minimum mean elo for selection.",
    )
    parser.add_argument(
        "--max-elo",
        type=int,
        default=2700,
        help="Maximum mean elo for selection.",
    )
    parser.add_argument(
        "--elo-bin-size",
        type=int,
        default=200,
        help="Elo bin width for stratified selection.",
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=20,
        help="Maximum selected players per elo bin.",
    )
    parser.add_argument(
        "--min-active-months",
        type=int,
        default=2,
        help="Require players to span at least this many distinct months.",
    )
    parser.add_argument(
        "--min-coverage-days",
        type=int,
        default=14,
        help="Require players to span at least this many days between first and last game.",
    )
    parser.add_argument(
        "--rated-only",
        action="store_true",
        help="Use only rated games.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_timestamp_expr() -> str:
    # Robustly handles:
    # - ms epoch integers
    # - regular timestamps
    # - UTCDate strings/dates
    return """
    COALESCE(
        CASE
            WHEN TRY_CAST(last_move_at AS BIGINT) IS NOT NULL
                 AND TRY_CAST(last_move_at AS BIGINT) > 100000000000
            THEN to_timestamp(CAST(last_move_at AS DOUBLE) / 1000.0)
            ELSE TRY_CAST(last_move_at AS TIMESTAMP)
        END,
        CASE
            WHEN TRY_CAST(created_at AS BIGINT) IS NOT NULL
                 AND TRY_CAST(created_at AS BIGINT) > 100000000000
            THEN to_timestamp(CAST(created_at AS DOUBLE) / 1000.0)
            ELSE TRY_CAST(created_at AS TIMESTAMP)
        END,
        TRY_CAST(utc_date AS TIMESTAMP),
        CAST(TRY_CAST(utc_date AS DATE) AS TIMESTAMP)
    )
    """


def main() -> int:
    args = parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_path}")

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    con = duckdb.connect(str(db_path))
    source = quote_ident(args.source_table)

    # Basic existence check
    con.execute(f"SELECT COUNT(*) FROM {source}")

    rated_filter = "AND COALESCE(rated, FALSE) = TRUE" if args.rated_only else ""
    ts_expr = build_timestamp_expr()

    # 1) Normalize into one row per player per game
    con.execute("DROP VIEW IF EXISTS player_games_norm")
    con.execute(
        f"""
        CREATE TEMP VIEW player_games_norm AS
        WITH base AS (
            SELECT
                game_id,
                white_player,
                black_player,
                TRY_CAST(white_elo AS DOUBLE) AS white_elo,
                TRY_CAST(black_elo AS DOUBLE) AS black_elo,
                winner,
                victory_status,
                rated,
                speed,
                opening,
                source_file,
                {ts_expr} AS game_ts
            FROM {source}
        )
        SELECT
            game_id,
            'white' AS side,
            white_player AS player_name,
            white_elo AS player_elo,
            black_elo AS opponent_elo,
            CASE
                WHEN winner = 'white' THEN 'win'
                WHEN winner = 'black' THEN 'loss'
                ELSE 'draw_or_other'
            END AS outcome,
            victory_status,
            rated,
            speed,
            opening,
            source_file,
            game_ts
        FROM base
        WHERE white_player IS NOT NULL
          AND game_ts IS NOT NULL
          AND white_elo IS NOT NULL
          {rated_filter}

        UNION ALL

        SELECT
            game_id,
            'black' AS side,
            black_player AS player_name,
            black_elo AS player_elo,
            white_elo AS opponent_elo,
            CASE
                WHEN winner = 'black' THEN 'win'
                WHEN winner = 'white' THEN 'loss'
                ELSE 'draw_or_other'
            END AS outcome,
            victory_status,
            rated,
            speed,
            opening,
            source_file,
            game_ts
        FROM base
        WHERE black_player IS NOT NULL
          AND game_ts IS NOT NULL
          AND black_elo IS NOT NULL
          {rated_filter}
        """
    )

    # 2) Order per player and detect sessions + rating changes
    con.execute("DROP VIEW IF EXISTS player_games_ordered")
    con.execute(
        f"""
        CREATE TEMP VIEW player_games_ordered AS
        WITH ordered AS (
            SELECT
                *,
                LAG(game_ts) OVER (
                    PARTITION BY player_name
                    ORDER BY game_ts, game_id
                ) AS prev_game_ts,
                LAG(player_elo) OVER (
                    PARTITION BY player_name
                    ORDER BY game_ts, game_id
                ) AS prev_player_elo
            FROM player_games_norm
        )
        SELECT
            *,
            CASE
                WHEN prev_game_ts IS NULL THEN 1
                WHEN game_ts > prev_game_ts + INTERVAL '{args.session_gap_hours} hours' THEN 1
                ELSE 0
            END AS new_session_flag,
            CASE
                WHEN prev_player_elo IS NULL THEN NULL
                ELSE player_elo - prev_player_elo
            END AS elo_delta
        FROM ordered
        """
    )

    con.execute("DROP VIEW IF EXISTS player_games_with_sessions")
    con.execute(
        """
        CREATE TEMP VIEW player_games_with_sessions AS
        SELECT
            *,
            SUM(new_session_flag) OVER (
                PARTITION BY player_name
                ORDER BY game_ts, game_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS session_id
        FROM player_games_ordered
        """
    )

    # 3) Build per-player profile table
    con.execute("DROP TABLE IF EXISTS player_profiles")
    con.execute(
        f"""
        CREATE TEMP TABLE player_profiles AS
        SELECT
            player_name,
            COUNT(*) AS n_games,
            COUNT(DISTINCT session_id) AS n_sessions,
            MIN(game_ts) AS first_game_ts,
            MAX(game_ts) AS last_game_ts,
            date_diff('day', MIN(game_ts), MAX(game_ts)) AS coverage_days,
            COUNT(DISTINCT strftime(game_ts, '%Y-%m')) AS active_months,
            AVG(player_elo) AS mean_elo,
            MIN(player_elo) AS min_elo_obs,
            MAX(player_elo) AS max_elo_obs,
            STDDEV_SAMP(player_elo) AS rating_stddev,
            STDDEV_SAMP(opponent_elo) AS opponent_elo_stddev,
            COUNT(DISTINCT speed) AS speed_diversity,
            COUNT(DISTINCT opening) AS opening_diversity,
            COUNT(DISTINCT outcome) AS outcome_diversity,
            SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
            SUM(CASE WHEN outcome = 'draw_or_other' THEN 1 ELSE 0 END) AS draws_or_other,
            SUM(CASE WHEN elo_delta < 0 THEN 1 ELSE 0 END) AS elo_drop_steps,
            SUM(CASE WHEN elo_delta > 0 THEN 1 ELSE 0 END) AS elo_up_steps
        FROM player_games_with_sessions
        GROUP BY player_name
        """
    )

    # 4) Filter eligible players and stratify by elo bins
    con.execute("DROP TABLE IF EXISTS eligible_players")
    con.execute(
        f"""
        CREATE TEMP TABLE eligible_players AS
        WITH base AS (
            SELECT
                *,
                CAST(FLOOR((mean_elo - {args.min_elo}) / {args.elo_bin_size}) AS INTEGER) AS elo_bin_idx,
                CAST({args.min_elo} + CAST(FLOOR((mean_elo - {args.min_elo}) / {args.elo_bin_size}) AS INTEGER) * {args.elo_bin_size} AS INTEGER) AS elo_bin_lo,
                CAST({args.min_elo} + (CAST(FLOOR((mean_elo - {args.min_elo}) / {args.elo_bin_size}) AS INTEGER) + 1) * {args.elo_bin_size} AS INTEGER) AS elo_bin_hi,
                (
                    LN(1 + n_games) * 1.5
                    + LN(1 + n_sessions) * 2.0
                    + LN(1 + GREATEST(coverage_days, 0)) * 1.2
                    + active_months * 1.0
                    + speed_diversity * 0.8
                    + LN(1 + opening_diversity) * 0.7
                    + COALESCE(opponent_elo_stddev, 0) / 150.0
                    + COALESCE(rating_stddev, 0) / 75.0
                    + elo_drop_steps * 0.35
                    + CASE WHEN outcome_diversity >= 3 THEN 0.5 ELSE 0 END
                ) AS selection_score
            FROM player_profiles
            WHERE n_games > {args.min_games}
              AND n_sessions > {args.min_sessions - 1}
              AND mean_elo >= {args.min_elo}
              AND mean_elo <= {args.max_elo}
              AND active_months >= {args.min_active_months}
              AND coverage_days >= {args.min_coverage_days}
              AND elo_drop_steps >= 1
        )
        SELECT *
        FROM base
        WHERE elo_bin_idx >= 0
        """
    )

    con.execute("DROP TABLE IF EXISTS selected_players")
    con.execute(
        f"""
        CREATE TEMP TABLE selected_players AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY elo_bin_lo, elo_bin_hi
                    ORDER BY selection_score DESC, n_sessions DESC, n_games DESC
                ) AS bin_rank
            FROM eligible_players
        )
        SELECT *
        FROM ranked
        WHERE bin_rank <= {args.per_bin}
        ORDER BY elo_bin_lo, bin_rank, player_name
        """
    )

    # 5) Export selected players, eligible players, and selected games subset
    selected_players_parquet = out_dir / "selected_players.parquet"
    selected_players_csv = out_dir / "selected_players.csv"
    eligible_players_parquet = out_dir / "eligible_players.parquet"
    eligible_players_csv = out_dir / "eligible_players.csv"
    selected_games_parquet = out_dir / "selected_games.parquet"
    summary_json = out_dir / "selection_summary.json"
    bin_summary_csv = out_dir / "selected_players_by_bin.csv"

    con.execute(
        f"""
        COPY (
            SELECT * FROM selected_players
        )
        TO {json.dumps(selected_players_parquet.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM selected_players
        )
        TO {json.dumps(selected_players_csv.as_posix())}
        (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM eligible_players
        )
        TO {json.dumps(eligible_players_parquet.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM eligible_players
        )
        TO {json.dumps(eligible_players_csv.as_posix())}
        (HEADER, DELIMITER ',')
        """
    )

    # Export the game subset for selected players
    con.execute(
        f"""
        COPY (
            SELECT g.*
            FROM {source} g
            INNER JOIN selected_players s
              ON g.white_player = s.player_name
              OR g.black_player = s.player_name
        )
        TO {json.dumps(selected_games_parquet.as_posix())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    con.execute(
        f"""
        COPY (
            SELECT
                CONCAT(CAST(elo_bin_lo AS VARCHAR), '-', CAST(elo_bin_hi AS VARCHAR)) AS elo_bin,
                COUNT(*) AS n_selected,
                ROUND(AVG(mean_elo), 2) AS avg_mean_elo,
                ROUND(AVG(n_games), 2) AS avg_games,
                ROUND(AVG(n_sessions), 2) AS avg_sessions,
                ROUND(AVG(selection_score), 2) AS avg_selection_score
            FROM selected_players
            GROUP BY 1
            ORDER BY MIN(elo_bin_lo)
        )
        TO {json.dumps(bin_summary_csv.as_posix())}
        (HEADER, DELIMITER ',')
        """
    )

    n_eligible = con.execute("SELECT COUNT(*) FROM eligible_players").fetchone()[0]
    n_selected = con.execute("SELECT COUNT(*) FROM selected_players").fetchone()[0]
    bins = con.execute(
        """
        SELECT
            elo_bin_lo,
            elo_bin_hi,
            COUNT(*) AS n_selected
        FROM selected_players
        GROUP BY 1, 2
        ORDER BY 1
        """
    ).fetchall()

    summary = {
        "db_path": str(db_path),
        "source_table": args.source_table,
        "filters": {
            "min_games": args.min_games,
            "min_sessions": args.min_sessions,
            "session_gap_hours": args.session_gap_hours,
            "min_elo": args.min_elo,
            "max_elo": args.max_elo,
            "elo_bin_size": args.elo_bin_size,
            "per_bin": args.per_bin,
            "min_active_months": args.min_active_months,
            "min_coverage_days": args.min_coverage_days,
            "rated_only": args.rated_only,
        },
        "n_eligible_players": int(n_eligible),
        "n_selected_players": int(n_selected),
        "selected_bins": [
            {
                "elo_bin_lo": int(row[0]),
                "elo_bin_hi": int(row[1]),
                "n_selected": int(row[2]),
            }
            for row in bins
        ],
        "outputs": {
            "selected_players_parquet": str(selected_players_parquet),
            "selected_players_csv": str(selected_players_csv),
            "eligible_players_parquet": str(eligible_players_parquet),
            "eligible_players_csv": str(eligible_players_csv),
            "selected_games_parquet": str(selected_games_parquet),
            "selected_players_by_bin_csv": str(bin_summary_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())