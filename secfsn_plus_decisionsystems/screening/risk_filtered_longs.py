"""Risk-filtered long candidate screen."""

from __future__ import annotations

import pandas as pd


def screen_risk_filtered_longs(
    panel: pd.DataFrame,
    risk_col: str = "hybrid_decision_score",
    max_risk: float = 0.3,
) -> pd.DataFrame:
    """Return rows with risk below threshold for long-side consideration."""

    if risk_col not in panel.columns:
        return panel.iloc[0:0].copy()
    return panel.loc[panel[risk_col].fillna(float("inf")).le(max_risk)].copy()
