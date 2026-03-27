from __future__ import annotations

import numpy as np



def continue_score(upside: np.ndarray, downside: np.ndarray, *, risk_lambda: float, friction_cost: float) -> np.ndarray:
    return np.asarray(upside, dtype=float) - (float(risk_lambda) * np.asarray(downside, dtype=float)) - float(friction_cost)



def classification_to_action(prob: np.ndarray, *, continue_threshold: float) -> np.ndarray:
    return np.where(np.asarray(prob, dtype=float) >= float(continue_threshold), "continue", "stop")



def score_to_action(score: np.ndarray, *, continue_threshold: float = 0.0) -> np.ndarray:
    return np.where(np.asarray(score, dtype=float) > float(continue_threshold), "continue", "stop")
