"""Rules-based generation of short candidate universe."""

from __future__ import annotations

import pandas as pd


def _resolve_risk_col(panel: pd.DataFrame, risk_col: str) -> str | None:
    if risk_col in panel.columns:
        return risk_col
    if risk_col == "hybrid_risk_score" and "hybrid_decision_score" in panel.columns:
        return "hybrid_decision_score"
    return None


def build_short_candidates(
    panel: pd.DataFrame,
    top_n: int,
    score_col: str = "hybrid_risk_score",
    structural_flag_col: str = "is_structural_break",
    anomaly_flag_cols: tuple[str, ...] = (
        "is_cross_sectional_z_anomaly",
        "is_robust_mad_anomaly",
        "is_rolling_instability",
    ),
) -> pd.DataFrame:
    """Return top-N ranked short candidates filtered by structural/anomaly criteria."""

    if top_n <= 0 or score_col not in panel.columns:
        return panel.iloc[0:0].copy()

    result = panel.copy()
    structural_flag = (
        result[structural_flag_col].fillna(False).astype(bool)
        if structural_flag_col in result.columns
        else pd.Series(False, index=result.index)
    )

    available_anomaly_flags = [column for column in anomaly_flag_cols if column in result.columns]
    if available_anomaly_flags:
        anomaly_flag = result[available_anomaly_flags].fillna(False).astype(bool).any(axis=1)
    else:
        anomaly_flag = pd.Series(False, index=result.index)

    eligible = result.loc[structural_flag | anomaly_flag].copy()
    if eligible.empty:
        return eligible

    eligible[score_col] = pd.to_numeric(eligible[score_col], errors="coerce")
    ranked = eligible.sort_values(by=score_col, ascending=False, kind="mergesort")
    return ranked.head(top_n)


def screen_short_candidates(
    panel: pd.DataFrame,
    risk_col: str = "hybrid_risk_score",
    allowed_col: str | None = "short_allowed",
    min_risk: float = 0.7,
) -> pd.DataFrame:
    """Return rows that satisfy deterministic short-candidate criteria.

    Missing ``risk_col`` returns an empty DataFrame with original columns.
    """

    if allowed_col is not None and allowed_col in panel.columns:
        return panel.loc[panel[allowed_col].fillna(False).astype(bool)].copy()
    resolved_risk_col = _resolve_risk_col(panel, risk_col)
    if resolved_risk_col is None:
        return panel.iloc[0:0].copy()
    return panel.loc[panel[resolved_risk_col].fillna(float("-inf")).ge(min_risk)].copy()
