from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
from tqdm.auto import tqdm

from chessquant_ml.settings import settings
from chessquant_ml.utils.io import read_json, write_json


@dataclass
class EngineSummary:
    acpl: float
    blunders: int
    mistakes: int
    inaccuracies: int
    largest_eval_drop_cp: float
    avg_eval_drop_cp: float
    eval_volatility: float
    threw_winning_position: int


def _score_to_cp(score: chess.engine.PovScore) -> int:
    try:
        cp = score.white().score(mate_score=10000)
        return int(cp if cp is not None else 0)
    except Exception:
        return 0


def _evaluate_game_moves(
    engine: chess.engine.SimpleEngine,
    moves_san_or_uci: str,
    pbar: tqdm | None = None,
) -> list[int]:
    board = chess.Board()
    evals = [0]
    limit = chess.engine.Limit(time=max(settings.engine_movetime_ms, 5) / 1000.0)

    tokens = [t for t in moves_san_or_uci.split() if t]
    for idx, token in enumerate(tokens, start=1):
        try:
            move = board.parse_san(token)
        except Exception:
            move = chess.Move.from_uci(token)

        board.push(move)
        info = engine.analyse(board, limit)
        evals.append(_score_to_cp(info["score"]))

        if pbar is not None:
            pbar.set_postfix(move=f"{idx}/{len(tokens)}")

    return evals


def _derive_summary(evals: list[int], user_color: str) -> EngineSummary:
    user_evals = [cp if user_color == "white" else -cp for cp in evals]
    losses: list[int] = []
    inaccuracies = 0
    mistakes = 0
    blunders = 0
    threw_winning = 0

    own_parity = 1 if user_color == "white" else 0
    for ply in range(1, len(user_evals)):
        if ply % 2 != own_parity:
            continue
        before = user_evals[ply - 1]
        after = user_evals[ply]
        loss = max(0, before - after)
        losses.append(loss)
        if loss >= 50:
            inaccuracies += 1
        if loss >= 100:
            mistakes += 1
        if loss >= 300:
            blunders += 1
        if before >= 300 and after < 100:
            threw_winning = 1

    acpl = float(sum(losses) / len(losses)) if losses else 0.0
    avg_drop = acpl
    largest_drop = float(max(losses)) if losses else 0.0
    mean_eval = float(sum(user_evals) / len(user_evals)) if user_evals else 0.0
    variance = (
        sum((x - mean_eval) ** 2 for x in user_evals) / len(user_evals)
        if user_evals
        else 0.0
    )
    volatility = variance ** 0.5

    return EngineSummary(
        acpl=acpl,
        blunders=blunders,
        mistakes=mistakes,
        inaccuracies=inaccuracies,
        largest_eval_drop_cp=largest_drop,
        avg_eval_drop_cp=avg_drop,
        eval_volatility=float(volatility),
        threw_winning_position=threw_winning,
    )


def enrich_games_with_engine(raw_path: Path, enriched_path: Path):
    games: list[dict[str, Any]] = read_json(raw_path)
    enriched: list[dict[str, Any]] = []

    reused_lichess = 0
    computed_local = 0
    skipped = 0

    engine = chess.engine.SimpleEngine.popen_uci(settings.stockfish_path)
    engine.configure({
        "Threads": settings.engine_threads,
        "Hash": settings.engine_hash_mb,
    })

    bar = tqdm(games, desc="Enriching games", unit="game")
    try:
        for game in bar:
            game_id = str(game.get("id", "unknown"))
            white_name = str(
                game.get("players", {}).get("white", {}).get("user", {}).get("name", "")
            )
            user_color = (
                "white"
                if white_name.lower() == settings.lichess_username.lower()
                else "black"
            )

            white_analysis = game.get("players", {}).get("white", {}).get("analysis") or {}
            black_analysis = game.get("players", {}).get("black", {}).get("analysis") or {}

            has_eval = bool(game.get("analysis")) or bool(white_analysis) or bool(black_analysis)
            evals = None
            summary = None

            if has_eval:
                hero_analysis = white_analysis if user_color == "white" else black_analysis
                summary = {
                    "my_acpl": float(hero_analysis.get("acpl", 0) or 0),
                    "my_blunder_count": int(hero_analysis.get("blunder", hero_analysis.get("blunders", 0)) or 0),
                    "my_mistake_count": int(hero_analysis.get("mistake", hero_analysis.get("mistakes", 0)) or 0),
                    "my_inaccuracy_count": int(hero_analysis.get("inaccuracy", hero_analysis.get("inaccuracies", 0)) or 0),
                    "largest_eval_drop_cp": None,
                    "avg_eval_drop_cp": None,
                    "eval_volatility": None,
                    "threw_winning_position": None,
                    "engine_source": "lichess",
                }
                reused_lichess += 1
            else:
                moves = str(game.get("moves", "") or "")
                if moves:
                    evals = _evaluate_game_moves(engine, moves, pbar=bar)
                    derived = _derive_summary(evals, user_color)
                    summary = {
                        "my_acpl": derived.acpl,
                        "my_blunder_count": derived.blunders,
                        "my_mistake_count": derived.mistakes,
                        "my_inaccuracy_count": derived.inaccuracies,
                        "largest_eval_drop_cp": derived.largest_eval_drop_cp,
                        "avg_eval_drop_cp": derived.avg_eval_drop_cp,
                        "eval_volatility": derived.eval_volatility,
                        "threw_winning_position": derived.threw_winning_position,
                        "engine_source": "local_stockfish",
                    }
                    computed_local += 1
                else:
                    skipped += 1

            game["cq_engine"] = {
                "eval_timeline_cp": evals,
                **(summary or {}),
            }
            enriched.append(game)

            bar.set_postfix(
                id=game_id[:8],
                lichess=reused_lichess,
                local=computed_local,
                skipped=skipped,
            )
    finally:
        engine.quit()
        bar.close()

    write_json(enriched_path, enriched)
    return enriched_path