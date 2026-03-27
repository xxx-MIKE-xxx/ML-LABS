from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import polars as pl
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from xgboost import XGBClassifier

from chessquant_ml.settings import settings
from chessquant_ml.utils.io import write_json


def export_to_onnx(model_path: Path, config_path: Path, feature_path: Path) -> Path:
    print(f"Loading config from {config_path} ...")
    config = joblib.load(config_path)
    features = list(config["features"])

    sklearn_model_path = config.get("sklearn_model_path")
    if sklearn_model_path:
        print(f"Loading sklearn model from {sklearn_model_path} ...")
        model = joblib.load(sklearn_model_path)
    else:
        print(f"Loading model from {model_path} ...")
        model = XGBClassifier()
        setattr(model, "_estimator_type", "classifier")
        model.load_model(model_path)

    print("Preparing sample batch for ONNX export validation ...")
    df = pl.read_parquet(feature_path).select(features).head(32).to_pandas().astype(np.float32)
    _ = model.predict_proba(df)

    print("Converting model to ONNX ...")
    initial_types = [("float_input", FloatTensorType([None, len(features)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)

    out_path = settings.models_dir / "tilt_xgb.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Validating exported ONNX model ...")
    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    _ = session.run(None, {input_name: df.to_numpy(dtype=np.float32)})

    write_json(
        settings.artifacts_dir / "web_export_manifest.json",
        {
            "onnx_model": str(out_path),
            "feature_order": features,
            "threshold": float(config["threshold"]),
            "notes": "Feed float32 tensor in this exact feature order on web.",
        },
    )
    print(f"ONNX export ready: {out_path}")
    return out_path