# Commands

## 0) Go to the real project root
```powershell
cd "C:\Users\User\Desktop\ML LABS\chess quant v2\chessquant-ml-labs-v2\chessquant-ml-labs-v2"
```

## 1) Backup the current settings file
```powershell
Copy-Item .\src\chessquant_ml\settings.py .\src\chessquant_ml\settings.py.bak -Force
```

## 2) Replace `src/chessquant_ml/settings.py` with the example from this patch
```powershell
Copy-Item .\PATCH_V2\patches\settings.py .\src\chessquant_ml\settings.py -Force
```

## 3) Create registry and snapshot folders
```powershell
New-Item -ItemType Directory -Force .\data\registry | Out-Null
New-Item -ItemType Directory -Force .\data\snapshots | Out-Null
New-Item -ItemType Directory -Force .\outputs\runs | Out-Null
```

## 4) Copy new files into the project
```powershell
Copy-Item .\PATCH_V2\src\chessquant_ml\data\registry.py .\src\chessquant_ml\data\registry.py -Force
Copy-Item .\PATCH_V2\src\chessquant_ml\data\snapshot.py .\src\chessquant_ml\data\snapshot.py -Force
Copy-Item .\PATCH_V2\src\chessquant_ml\pipeline\incremental.py .\src\chessquant_ml\pipeline\incremental.py -Force
Copy-Item .\PATCH_V2\scripts\bootstrap_registry.py .\scripts\bootstrap_registry.py -Force
Copy-Item .\PATCH_V2\configs\datasets\incremental_default.yaml .\configs\datasets\incremental_default.yaml -Force
```

## 5) Bootstrap the registry from current raw/enriched/features files
```powershell
poetry run python .\scripts\bootstrap_registry.py
```

## 6) Quick verification
```powershell
poetry run python -c "from chessquant_ml.settings import settings; print(settings.project_root); print(settings.feature_schema_path)"
```

The second printed path should end with:
```text
config\feature_schema.json
```

## 7) Smoke test without retraining everything
```powershell
poetry run chessquant-ml fetch --max-games 20
poetry run chessquant-ml enrich
poetry run chessquant-ml features
```

At this point the path bug should be fixed. After that, wire the incremental helpers into
`cli/main.py`, `pipeline/lichess_client.py`, `pipeline/engine_fill.py`, and `pipeline/features.py`.
