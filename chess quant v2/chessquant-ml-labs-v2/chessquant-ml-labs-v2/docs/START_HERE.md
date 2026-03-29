# ChessQuant ML Labs - Start Here

## Daily workflow

1. Fetch / enrich / build features with the package CLI.
2. Train the production-style model from `src/chessquant_ml/training/`.
3. Export ONNX from `src/chessquant_ml/exporting/`.
4. Use the research sandbox for objective comparison and policy evaluation.

## Important locations

- Production-like model artifacts: `data/models/`
- Export manifests: `data/artifacts/`
- Research outputs copied or compared later: `outputs/`
- Handoff bundle for implementation team: `production_handoff/`

## Cleanup policy

- Remove cache-like Python artifacts (`__pycache__`, `*.pyc`) freely.
- Keep `data/` and `outputs/` unless you intentionally want to reset experiments.
- Keep `production_handoff/` because it is part of delivery.
