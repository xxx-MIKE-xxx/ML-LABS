from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow



def configure_mlflow(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)



def log_config(config: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(path))



def log_metrics_flat(metrics: dict[str, Any], prefix: str = "") -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"{prefix}{key}", float(value))
