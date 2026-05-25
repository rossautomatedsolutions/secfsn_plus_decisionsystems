"""Deterministic structural-risk scoring on prepared feature panels."""

from __future__ import annotations

import pandas as pd

STRUCTURAL_FEATURE_WEIGHTS: dict[str, float] = {
    "equity_ni_residual_scaled": 1.0,
    "cash_vs_earnings_gap_scaled": 1.0,
    "accrual_intensity": 1.0,
    "structural_drift_1q": 1.0,
    "margin_volatility_4q": 1.0,
    "balance_sheet_identity_residual_scaled": 1.0,
}


def build_structural_risk_score(
    panel: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    output_col: str = "structural_risk_score",
) -> pd.DataFrame:
    """Build a deterministic structural-risk score from available feature columns.

    Missing configured columns are skipped. Missing values in available columns are
    treated as neutral (0.0). The score is a normalized weighted sum.
    """

    result = panel.copy()
    weights = STRUCTURAL_FEATURE_WEIGHTS if feature_weights is None else feature_weights

    available = [column for column in weights if column in result.columns]
    if not available:
        if output_col in result.columns:
            return result
        result[output_col] = 0.0
        return result

    weight_series = pd.Series({column: float(weights[column]) for column in available})
    numerator = result[available].fillna(0.0).mul(weight_series, axis=1).sum(axis=1)
    denominator = float(weight_series.abs().sum()) or 1.0

    result[output_col] = numerator / denominator
    return result


def compute_structural_risk_score(
    panel: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    output_col: str = "structural_risk_score",
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`build_structural_risk_score`."""

    return build_structural_risk_score(
        panel=panel,
        feature_weights=feature_weights,
        output_col=output_col,
    )
