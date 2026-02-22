"""Rules-based generation of short candidate universe."""

from __future__ import annotations

import pandas as pd


def screen_short_candidates(
    panel: pd.DataFrame,
    risk_col: str = "hybrid_decision_score",
    min_risk: float = 0.7,
) -> pd.DataFrame:
    """Return rows that satisfy deterministic short-candidate criteria.

    Missing ``risk_col`` returns an empty DataFrame with original columns.
    """

    if risk_col not in panel.columns:
        return panel.iloc[0:0].copy()
    return panel.loc[panel[risk_col].fillna(float("-inf")).ge(min_risk)].copy()
