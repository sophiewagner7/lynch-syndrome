# icer.py
"""
Calculate ICER for a set of runs.
"""
from typing import Iterable, Tuple

import pandas as pd
import numpy as np


# ---------- ICERs across strategies ----------
def summarize_for_icer(
    results: Iterable[Tuple[str, pd.DataFrame, pd.DataFrame]],
    use_discounted: bool = True,
) -> pd.DataFrame:
    """
    Build a summary table for ICERs from multiple runs.
    Each item in `results` is (label, per_cycle_df, lifetime_df).
    """
    rows = []
    total_cost = "per_person_cost_disc" if use_discounted else "per_person_cost"
    total_qalys = "per_person_qalys_disc" if use_discounted else "per_person_qalys"
    for label, _per_cycle, life in results:
        rows.append(
            {
                "label": label,
                "cost": float(life[total_cost].iloc[0]),
                "qalys": float(life[total_qalys].iloc[0]),
            }
        )
    df = (
        pd.DataFrame(rows)
        .sort_values(by=["qalys", "cost"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return df


def compute_icers(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ICERs with simple dominance checks.
    Expects columns: ['label', 'cost', 'qalys'] sorted by increasing qalys.
    Removes strictly dominated strategies; then computes incremental pairs.
    Note: Extended dominance handling here is basic; adjust as needed.
    """
    df = summary.copy().reset_index(drop=True)

    # Remove strictly dominated (higher cost AND lower/equal qalys)
    keep = []
    best_cost = np.inf
    best_q = -np.inf
    for i, row in df.iterrows():
        dominated = (row["cost"] >= best_cost) and (row["qalys"] <= best_q)
        if not dominated:
            keep.append(i)
            best_cost = min(best_cost, row["cost"])
            best_q = max(best_q, row["qalys"])
    df = df.loc[keep].reset_index(drop=True)

    # Incremental pairs
    inc_cost = [np.nan]
    inc_q = [np.nan]
    icer = [np.nan]
    for i in range(1, len(df)):
        dC = df.loc[i, "cost"] - df.loc[i - 1, "cost"]
        dE = df.loc[i, "qalys"] - df.loc[i - 1, "qalys"]
        inc_cost.append(dC)
        inc_q.append(dE)
        icer.append(dC / dE if dE > 0 else np.nan)

    df["incr_cost"] = inc_cost
    df["incr_qalys"] = inc_q
    df["ICER"] = icer
    return df
