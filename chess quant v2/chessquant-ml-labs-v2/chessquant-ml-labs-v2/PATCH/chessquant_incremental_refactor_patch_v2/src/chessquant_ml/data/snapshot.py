from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import hashlib

import polars as pl


def dataframe_fingerprint(df: pl.DataFrame) -> str:
    payload = {
        "rows": df.height,
        "cols": df.columns,
        "schema": {k: str(v) for k, v in df.schema.items()},
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_snapshot(df: pl.DataFrame, snapshots_dir: Path, prefix: str = "features") -> tuple[Path, Path]:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = dataframe_fingerprint(df)
    parquet_path = snapshots_dir / f"{prefix}_{stamp}_{fp}.parquet"
    manifest_path = snapshots_dir / f"{prefix}_{stamp}_{fp}.json"
    df.write_parquet(parquet_path)
    manifest = {
        "snapshot_name": parquet_path.name,
        "rows": df.height,
        "columns": df.columns,
        "fingerprint": fp,
        "created_at": stamp,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return parquet_path, manifest_path
