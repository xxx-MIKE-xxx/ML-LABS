from __future__ import annotations

from pathlib import Path
from time import perf_counter

import typer
from typing import Annotated


from chessquant_ml.pipeline.engine_fill import enrich_games_with_engine
from chessquant_ml.pipeline.features import build_feature_table
from chessquant_ml.pipeline.lichess_client import LichessClient
from chessquant_ml.exporting.export_web import export_to_onnx
from chessquant_ml.settings import settings
from chessquant_ml.training.train import train_tilt_model

app = typer.Typer(help="ChessQuant ML lab CLI")


def step(msg: str) -> None:
    typer.secho(f"\n==> {msg}", fg=typer.colors.CYAN, bold=True)


def done(msg: str, started_at: float) -> None:
    typer.secho(
        f"✓ {msg} ({perf_counter() - started_at:.1f}s)",
        fg=typer.colors.GREEN,
        bold=True,
    )


@app.command()
def fetch(
    username: str | None = None,
    out: str | None = None,
    max_games: Annotated[int | None, typer.Option("--max-games")] = None,
) -> None:
    if username:
        settings.lichess_username = username
    path = Path(out) if out else settings.raw_dir / f"{settings.lichess_username}_games_raw.json"
    client = LichessClient()
    t0 = perf_counter()
    step(f"Fetching raw games for {settings.lichess_username}")
    dumped = client.dump_games(path, max_games=max_games)
    done(f"Saved raw games to {dumped}", t0)


@app.command()
def enrich(raw_path: str | None = None, out: str | None = None) -> None:
    in_path = Path(raw_path) if raw_path else settings.raw_dir / f"{settings.lichess_username}_games_raw.json"
    out_path = Path(out) if out else settings.enriched_dir / f"{settings.lichess_username}_games_enriched.json"
    t0 = perf_counter()
    step("Enriching games with engine summaries")
    result = enrich_games_with_engine(in_path, out_path)
    done(f"Saved enriched games to {result}", t0)


@app.command()
def features(enriched_path: str | None = None, out: str | None = None) -> None:
    in_path = Path(enriched_path) if enriched_path else settings.enriched_dir / f"{settings.lichess_username}_games_enriched.json"
    out_path = Path(out) if out else settings.features_dir / f"{settings.lichess_username}_features.parquet"
    t0 = perf_counter()
    step("Building feature table")
    result = build_feature_table(in_path, out_path)
    done(f"Saved feature table to {result}", t0)


@app.command()
def train(feature_path: str | None = None) -> None:
    in_path = Path(feature_path) if feature_path else settings.features_dir / f"{settings.lichess_username}_features.parquet"
    t0 = perf_counter()
    step("Training tilt model")
    model_path, config_path = train_tilt_model(in_path)
    done(f"Saved model to {model_path}", t0)
    typer.echo(f"Config: {config_path}")


@app.command()
def export_web(model_path: str | None = None, config_path: str | None = None, feature_path: str | None = None) -> None:
    mp = Path(model_path) if model_path else settings.models_dir / "tilt_xgb.json"
    cp = Path(config_path) if config_path else settings.models_dir / "tilt_config.joblib"
    fp = Path(feature_path) if feature_path else settings.features_dir / f"{settings.lichess_username}_features.parquet"
    t0 = perf_counter()
    step("Exporting model to ONNX")
    out = export_to_onnx(mp, cp, fp)
    done(f"Saved ONNX model to {out}", t0)

@app.command()

def all(
    max_games: Annotated[int | None, typer.Option("--max-games")] = None,
) -> None:
    raw = settings.raw_dir / f"{settings.lichess_username}_games_raw.json"
    enriched = settings.enriched_dir / f"{settings.lichess_username}_games_enriched.json"
    feat = settings.features_dir / f"{settings.lichess_username}_features.parquet"

    total_t0 = perf_counter()

    client = LichessClient()

    t0 = perf_counter()
    step(f"[1/5] Fetching raw games for {settings.lichess_username}")
    client.dump_games(raw, max_games=max_games)
    done("Raw fetch complete", t0)

    t0 = perf_counter()
    step("[2/5] Enriching games with engine summaries")
    enrich_games_with_engine(raw, enriched)
    done("Engine enrichment complete", t0)

    t0 = perf_counter()
    step("[3/5] Building feature table")
    build_feature_table(enriched, feat)
    done("Feature build complete", t0)

    t0 = perf_counter()
    step("[4/5] Training model")
    model_path, config_path = train_tilt_model(feat)
    done("Training complete", t0)

    t0 = perf_counter()
    step("[5/5] Exporting web model")
    export_to_onnx(model_path, config_path, feat)
    done("Web export complete", t0)

    typer.secho(
        f"\nEND-TO-END PIPELINE COMPLETE in {perf_counter() - total_t0:.1f}s",
        fg=typer.colors.GREEN,
        bold=True,
    )


if __name__ == "__main__":
    app()