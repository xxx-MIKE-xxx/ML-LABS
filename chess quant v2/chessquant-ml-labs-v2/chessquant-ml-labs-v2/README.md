# ChessQuant ML Labs v2

Local ML lab for retraining the ChessQuant live tilt detector from scratch.

## What this lab does

1. Downloads as many finished Lichess games as possible for one user.
2. Preserves all Lichess metadata available from the export endpoint.
3. Fills missing engine summaries with local Stockfish.
4. Builds a training table for the tilt detector.
5. Trains an XGBoost model.
6. Optimizes the stop threshold against session P/L.
7. Exports the model to a web-friendly ONNX artifact.

## Folder layout

- `data/raw/` raw Lichess JSON exports
- `data/enriched/` raw games plus `cq_engine` summary block
- `data/features/` model-ready parquet tables
- `data/models/` XGBoost and ONNX artifacts
- `data/artifacts/` summaries and manifests
- `config/feature_schema.json` feature order for training and web inference

## Quick start

```bash
poetry install
copy .env.example .env
```

Edit `.env` and set:
- `LICHESS_USERNAME=Matumnich`
- `STOCKFISH_PATH` to your local Stockfish binary

Then run:

```bash
poetry run chessquant-ml all
```

Or step by step:

```bash
poetry run chessquant-ml fetch
poetry run chessquant-ml enrich
poetry run chessquant-ml features
poetry run chessquant-ml train
poetry run chessquant-ml export-web
```

## Output contract for the web app

The web app should load:
- `data/models/tilt_xgb.onnx`
- `data/artifacts/web_export_manifest.json`

The manifest contains the exact feature order and threshold.

## Notes

- This lab intentionally mirrors the older tilt-detector feature shape first.
- It keeps the current MVP focused on summary engine features instead of full sequence modeling.
- You can add richer eval-derived features later by extending `config/feature_schema.json` and `pipeline/features.py`.
