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
    g = work[work[group_col].astype(str) == first_group].copy().sort_values(order_col)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(g[order_col], g[cum_pl_col], marker="o")
    stop_rows = g[g[action_col] == "stop"]
    if not stop_rows.empty:
        ax.axvline(stop_rows[order_col].iloc[0], linestyle="--")
    ax.set_title(f"Session preview: {first_group}")
    ax.set_xlabel(order_col)
    ax.set_ylabel(cum_pl_col)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
