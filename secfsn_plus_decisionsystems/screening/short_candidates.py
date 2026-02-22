"""Rules-based generation of short candidate universe."""

from __future__ import annotations

import pandas as pd


def build_short_candidates(
    panel: pd.DataFrame,
    top_n: int,
    risk_col: str = "hybrid_risk_score",
    structural_flag_col: str = "is_high_structural_risk",
    anomaly_flag_col: str = "is_anomaly_flagged",
) -> pd.DataFrame:
    """Return top-N high-risk shorts requiring structural or anomaly confirmation."""

    result = panel.copy()
    if risk_col not in result.columns or top_n <= 0:
        return result.iloc[0:0].copy()

    structural_flags = result.get(structural_flag_col, pd.Series(False, index=result.index)).fillna(False)
    anomaly_flags = result.get(anomaly_flag_col, pd.Series(False, index=result.index)).fillna(False)
    eligibility = structural_flags | anomaly_flags

    shortlisted = result.loc[eligibility].copy()
    if shortlisted.empty:
        return shortlisted

    shortlisted = shortlisted.sort_values(by=risk_col, ascending=False)
    return shortlisted.head(top_n)


def screen_short_candidates(
    panel: pd.DataFrame,
    risk_col: str = "hybrid_decision_score",
    min_risk: float = 0.7,
) -> pd.DataFrame:
    """Backward-compatible threshold-based short-candidate screening."""

    if risk_col not in panel.columns:
        return panel.iloc[0:0].copy()
    return panel.loc[panel[risk_col].fillna(float("-inf")).ge(min_risk)].copy()
