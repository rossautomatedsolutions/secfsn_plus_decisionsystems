"""Risk-filtered long candidate screen."""

from __future__ import annotations

import pandas as pd


def _resolve_risk_col(panel: pd.DataFrame, risk_col: str) -> str | None:
    if risk_col in panel.columns:
        return risk_col
    if risk_col == "hybrid_risk_score" and "hybrid_decision_score" in panel.columns:
        return "hybrid_decision_score"
    return None


def screen_risk_filtered_longs(
    panel: pd.DataFrame,
    risk_col: str = "hybrid_risk_score",
    allowed_col: str | None = "long_allowed",
    max_risk: float = 0.3,
) -> pd.DataFrame:
    """Return rows with risk below threshold for long-side consideration."""

    if allowed_col is not None and allowed_col in panel.columns:
        return panel.loc[panel[allowed_col].fillna(False).astype(bool)].copy()
    resolved_risk_col = _resolve_risk_col(panel, risk_col)
    if resolved_risk_col is None:
        return panel.iloc[0:0].copy()
    return panel.loc[panel[resolved_risk_col].fillna(float("inf")).le(max_risk)].copy()
