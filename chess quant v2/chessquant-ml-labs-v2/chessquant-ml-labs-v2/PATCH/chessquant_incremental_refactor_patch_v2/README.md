# ChessQuant Incremental Refactor Patch v2

This patch is meant to be merged into the existing ChessQuant project after the move to `src/`.

## Goals
- Fix `settings.py` so project paths resolve from the project root, not from `src/`
- Add a registry for incremental fetch/enrich/features
- Add snapshot helpers for reproducible datasets
- Add a minimal incremental utility module
- Provide commands to merge the patch safely

## Important
Before applying the incremental patch, first fix the current path bug in
`src/chessquant_ml/settings.py`.

The symptom is:
- code tries to read `src/config/feature_schema.json`
- or writes artifacts under `src/data/...`

The correct locations are:
- `config/feature_schema.json`
- `data/raw`, `data/enriched`, `data/features`, `data/models`, `data/artifacts`
