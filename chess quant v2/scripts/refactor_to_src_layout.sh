#!/usr/bin/env bash
set -euo pipefail
mkdir -p src configs/datasets configs/experiments configs/policies outputs/figures outputs/metrics outputs/models outputs/reports outputs/web_exports scripts notebooks tests data/cache
if [ -d chessquant_ml ] && [ ! -d src/chessquant_ml ]; then
  mv chessquant_ml src/chessquant_ml
fi
if [ -d experiments/configs ]; then
  cp -r experiments/configs/. configs/experiments/
fi
echo "Refactor skeleton created."
echo "Next manual step: update pyproject.toml packages = [{ include = \"chessquant_ml\", from = \"src\" }]"
