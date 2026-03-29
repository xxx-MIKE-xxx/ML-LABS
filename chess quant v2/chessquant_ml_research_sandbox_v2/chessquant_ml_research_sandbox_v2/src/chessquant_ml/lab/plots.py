from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_session_plot(
    df: pd.DataFrame,
    *,
    group_col: str,
    order_col: str,
    cum_pl_col: str,
    action_col: str,
    out_path: str | Path,
) -> None:
    work = df.sort_values([group_col, order_col]).copy()
    first_group = work[group_col].astype(str).iloc[0]
    g = work[work[group_col].astype(str) == first_group].copy().sort_values(order_col).reset_index(drop=True)
    x = range(1, len(g) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, g[cum_pl_col], marker="o", label=cum_pl_col)

    stop_rows = g[g[action_col] == "stop"]
    if not stop_rows.empty:
        stop_game = int(stop_rows.index[0] + 1)
        ax.axvline(stop_game, linestyle="--", linewidth=1.5, label="predicted_stop")
        ax.scatter([stop_game], [float(stop_rows[cum_pl_col].iloc[0])], marker="x", s=80)

    peak_game = int(pd.to_numeric(g[cum_pl_col], errors="coerce").idxmax() + 1)
    peak_val = float(pd.to_numeric(g[cum_pl_col], errors="coerce").max())
    ax.scatter([peak_game], [peak_val], s=50, marker="o", label="actual_peak")

    ax.set_title(f"Session preview: {first_group}")
    ax.set_xlabel("game_in_session")
    ax.set_ylabel(cum_pl_col)
    ax.legend()
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
