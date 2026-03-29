from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from tqdm.auto import tqdm

from chessquant_ml.settings import settings
from chessquant_ml.utils.io import write_json


class LichessClient:
    def __init__(self) -> None:
        self.base_url = settings.lichess_api_base_url.rstrip("/")
        self.headers = {
            "Accept": "application/x-ndjson",
        }
        if settings.lichess_api_token:
            self.headers["Authorization"] = f"Bearer {settings.lichess_api_token}"

    def _fetch_batch(self, since: int | None = None, max_games: int = 300) -> list[dict[str, Any]]:
        url = f"{self.base_url}/{settings.lichess_username}"
        params = {
            "max": max_games,
            "finished": "true",
            "ongoing": "false",
            "clocks": "true",
            "evals": "true",
            "accuracy": "true",
            "opening": "true",
            "pgnInJson": "true",
            "literate": "false",
        }
        if since is not None:
            params["since"] = str(since)

        with httpx.Client(timeout=60.0, headers=self.headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            raw_text = response.text.strip()

        if not raw_text:
            return []

        if raw_text.startswith('[Event "'):
            raise RuntimeError(
                "Lichess returned PGN instead of NDJSON. "
                "Check that Accept is exactly application/x-ndjson."
            )

        rows: list[dict[str, Any]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows

    def fetch_all_recent_games(self, max_games: int | None = None) -> list[dict[str, Any]]:
        target = max_games or settings.lichess_max_games
        batch_size = min(settings.lichess_fetch_batch, target)
        all_games: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        since: int | None = None

        bar = tqdm(total=target, desc="Fetching games", unit="game")
        try:
            while len(all_games) < target:
                batch = self._fetch_batch(since=since, max_games=batch_size)
                if not batch:
                    break

                new_count = 0
                for game in batch:
                    game_id = str(game.get("id", ""))
                    if game_id and game_id not in seen_ids:
                        seen_ids.add(game_id)
                        all_games.append(game)
                        new_count += 1

                last_created = max(int(g.get("createdAt", 0) or 0) for g in batch)
                since = last_created + 1 if last_created > 0 else since

                bar.update(new_count)
                bar.set_postfix(
                    batch=len(batch),
                    new=new_count,
                    total=len(all_games),
                )

                if new_count == 0:
                    break
                if len(batch) < batch_size:
                    break
        finally:
            bar.close()

        all_games.sort(key=lambda g: int(g.get("createdAt", 0) or 0))
        return all_games[:target]

    def dump_games(self, path: Path, max_games: int | None = None) -> Path:
        games = self.fetch_all_recent_games(max_games=max_games)
        write_json(path, games)
        return path