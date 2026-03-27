from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    lichess_username: str = "Matumnich"
    lichess_api_token: str | None = None
    lichess_api_base_url: str = "https://lichess.org/api/games/user"
    lichess_max_games: int = 5000
    lichess_fetch_batch: int = 300

    session_gap_minutes: int = 30
    local_timezone: str = "Europe/Warsaw"

    stockfish_path: str = "stockfish"
    engine_movetime_ms: int = 40
    engine_threads: int = 4
    engine_hash_mb: int = 256

    mlflow_tracking_uri: str | None = None
    wandb_project: str = "chessquant-tilt-v2"
    wandb_mode: str = "disabled"

    raw_dir: Path = Field(default=ROOT / "data/raw")
    enriched_dir: Path = Field(default=ROOT / "data/enriched")
    features_dir: Path = Field(default=ROOT / "data/features")
    models_dir: Path = Field(default=ROOT / "data/models")
    artifacts_dir: Path = Field(default=ROOT / "data/artifacts")
    logs_dir: Path = Field(default=ROOT / "data/logs")
    feature_schema_path: Path = Field(default=ROOT / "config/feature_schema.json")


settings = Settings()
