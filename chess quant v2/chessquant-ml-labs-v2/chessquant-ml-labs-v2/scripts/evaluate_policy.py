from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from chessquant_ml.lab.backtester import backtest_stop_policy
from chessquant_ml.lab.plots import save_session_plot
from chessquant_ml.lab.policy import classification_to_action, score_to_action
from chessquant_ml.lab.targets import add_target



def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))



def resolve_run_dir(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_dir():
        return p
    if p.is_file():
        return Path(p.read_text(encoding="utf-8").strip())
    raise FileNotFoundError(path_str)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    policy_cfg = load_yaml(args.policy)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    model = joblib.load(run_dir / "model.joblib")
    df = pd.read_parquet(manifest["features_path"])

    objective = manifest["objective"]
    schema = manifest["schema"]
    df, target_col, task_type = add_target(
        df,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        objective_kind=objective["kind"],
        min_gain_elo=float(objective.get("min_gain_elo", 8.0)),
    )
    X = df[manifest["feature_order"]].fillna(0.0)

    if task_type == "classification":
        score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.asarray(model.predict(X), dtype=float)
        df["policy_action"] = classification_to_action(score, continue_threshold=float(policy_cfg["classification_continue_threshold"]))
    else:
        score = np.asarray(model.predict(X), dtype=float)
        df["policy_action"] = score_to_action(score, continue_threshold=float(policy_cfg.get("continue_threshold", 0.0)))

    summary = backtest_stop_policy(
        df,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        action_col="policy_action",
    )
    out = {
        "policy": policy_cfg["name"],
        "n_sessions": summary.n_sessions,
        "mean_realized_pl": summary.mean_realized_pl,
        "mean_peak_pl": summary.mean_peak_pl,
        "mean_capture_ratio": summary.mean_capture_ratio,
        "mean_stop_game_index": summary.mean_stop_game_index,
    }
    out_path = run_dir / f"policy_{policy_cfg['name']}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_session_plot(
        df,
        group_col=schema["group_col"],
        order_col=schema["order_col"],
        cum_pl_col=schema["cum_pl_col"],
        action_col="policy_action",
        out_path=run_dir / f"policy_{policy_cfg['name']}_preview.png",
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
