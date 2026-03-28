# ChessQuant visualization patch

This patch adds lightweight visualization utilities to the research repo.

Files:
- `src/chessquant_ml/lab/viz.py`
- `scripts/visualize_lab.py`

Recommended commands:

```powershell
Copy-Item .\PATCH_VIS\src\chessquant_ml\lab\viz.py .\src\chessquant_ml\lab\viz.py -Force
Copy-Item .\PATCH_VIS\scripts\visualize_lab.py .\scripts\visualize_lab.py -Force

poetry run python scripts/visualize_lab.py dataset-report --features-parquet data/features/Matumnich_features.parquet --output-dir outputs/figures/dataset_report --target-col target

poetry run python scripts/visualize_lab.py compare-runs --run-dirs outputs/models/20260327_201008_peak_binary_xgb outputs/models/20260327_201012_meaningful_upside_xgb outputs/models/20260327_201017_future_upside_xgb outputs/models/20260327_201021_future_drawdown_xgb --output-dir outputs/reports/run_compare
```
