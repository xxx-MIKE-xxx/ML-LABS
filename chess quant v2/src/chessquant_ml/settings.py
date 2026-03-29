from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str = "chessquant-lab"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    lichess_username: str = Field(default="Matumnich", alias="LICHESS_USERNAME")
    lichess_token: str | None = Field(default=None, alias="LICHESS_TOKEN")
    stockfish_path: str | None = Field(default=None, alias="STOCKFISH_PATH")

    # Resolve from: project_root/src/chessquant_ml/settings.py -> project_root
    project_root: Path = Path(__file__).resolve().parents[2]

    config_dir: Path = project_root / "config"
    data_dir: Path = project_root / "data"
    outputs_dir: Path = project_root / "outputs"

    raw_dir: Path = data_dir / "raw"
    enriched_dir: Path = data_dir / "enriched"
    features_dir: Path = data_dir / "features"
    models_dir: Path = data_dir / "models"
    artifacts_dir: Path = data_dir / "artifacts"
    logs_dir: Path = data_dir / "logs"
    cache_dir: Path = data_dir / "cache"
    registry_dir: Path = data_dir / "registry"
    snapshots_dir: Path = data_dir / "snapshots"

    feature_schema_path: Path = config_dir / "feature_schema.json"

    def ensure_dirs(self) -> None:
        for path in [
            self.config_dir,
            self.data_dir,
            self.outputs_dir,
            self.raw_dir,
            self.enriched_dir,
            self.features_dir,
            self.models_dir,
            self.artifacts_dir,
            self.logs_dir,
            self.cache_dir,
            self.registry_dir,
            self.snapshots_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
