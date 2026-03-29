from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def continue_score(
    upside: np.ndarray,
    downside: np.ndarray,
    *,
    risk_lambda: float,
    friction_cost: float,
) -> np.ndarray:
    return (
        np.asarray(upside, dtype=float)
        - (float(risk_lambda) * np.asarray(downside, dtype=float))
        - float(friction_cost)
    )


def classification_to_action(prob: np.ndarray, *, continue_threshold: float) -> np.ndarray:
    return np.where(np.asarray(prob, dtype=float) >= float(continue_threshold), "continue", "stop")


def score_to_action(score: np.ndarray, *, continue_threshold: float = 0.0) -> np.ndarray:
    return np.where(np.asarray(score, dtype=float) > float(continue_threshold), "continue", "stop")


def objective_threshold_name(objective_kind: str) -> str:
    if objective_kind == "peak_binary":
        return "peak_stop_threshold"
    if objective_kind == "meaningful_upside_binary":
        return "classification_continue_threshold"
    if objective_kind == "future_upside_regression":
        return "continue_threshold"
    if objective_kind == "future_drawdown_regression":
        return "future_drawdown_stop_threshold"
    if objective_kind == "games_to_peak_regression":
        return "games_to_peak_continue_threshold"
    raise ValueError(f"Unsupported objective_kind: {objective_kind}")


def objective_default_threshold(objective_kind: str) -> float:
    if objective_kind == "peak_binary":
        return 0.60
    if objective_kind == "meaningful_upside_binary":
        return 0.50
    if objective_kind == "future_upside_regression":
        return 0.0
    if objective_kind == "future_drawdown_regression":
        return 6.0
    if objective_kind == "games_to_peak_regression":
        return 1.0
    raise ValueError(f"Unsupported objective_kind: {objective_kind}")


def threshold_from_policy(policy_cfg: Mapping[str, Any], objective_kind: str) -> float:
    key = objective_threshold_name(objective_kind)
    if key in policy_cfg:
        return float(policy_cfg[key])
    if objective_kind == "future_upside_regression" and "future_upside_continue_threshold" in policy_cfg:
        return float(policy_cfg["future_upside_continue_threshold"])
    if objective_kind == "future_drawdown_regression" and "regression_stop_threshold" in policy_cfg:
        return float(policy_cfg["regression_stop_threshold"])
    if objective_kind == "peak_binary" and "classification_stop_threshold" in policy_cfg:
        return float(policy_cfg["classification_stop_threshold"])
    if objective_kind == "games_to_peak_regression" and "regression_stop_threshold" in policy_cfg:
        return float(policy_cfg["regression_stop_threshold"])
    if objective_kind in {"future_upside_regression", "games_to_peak_regression"} and "continue_threshold" in policy_cfg:
        return float(policy_cfg["continue_threshold"])
    return objective_default_threshold(objective_kind)


def action_from_objective(
    score: np.ndarray,
    *,
    objective_kind: str,
    policy_cfg: Mapping[str, Any],
    threshold_override: float | None = None,
) -> np.ndarray:
    values = np.asarray(score, dtype=float)
    threshold = float(threshold_override) if threshold_override is not None else threshold_from_policy(policy_cfg, objective_kind)

    if objective_kind == "peak_binary":
        return np.where(values >= threshold, "stop", "continue")
    if objective_kind == "meaningful_upside_binary":
        return np.where(values >= threshold, "continue", "stop")
    if objective_kind == "future_upside_regression":
        return np.where(values >= threshold, "continue", "stop")
    if objective_kind == "future_drawdown_regression":
        return np.where(values >= threshold, "stop", "continue")
    if objective_kind == "games_to_peak_regression":
        return np.where(values <= threshold, "stop", "continue")
    raise ValueError(f"Unsupported objective_kind: {objective_kind}")
