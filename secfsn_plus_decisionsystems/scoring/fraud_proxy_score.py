"""Deterministic fraud-proxy scoring on prepared panels."""

from __future__ import annotations

import pandas as pd

FRAUD_FEATURE_WEIGHTS: dict[str, float] = {
    "accrual_intensity": 1.0,
    "asset_revenue_growth_mismatch_scaled": 1.0,
    "non_cash_earnings_ratio": 1.0,
    "retained_earnings_gap_proxy_scaled": 1.0,
}


def build_fraud_proxy_score(
    panel: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    output_col: str = "fraud_proxy_score",
) -> pd.DataFrame:
    """Build a deterministic Beneish-style fraud proxy from available features."""

    result = panel.copy()
    weights = feature_weights or FRAUD_FEATURE_WEIGHTS

    available = [column for column in weights if column in result.columns]
    if not available:
        result[output_col] = 0.0
        return result

    weight_series = pd.Series({column: float(weights[column]) for column in available})
    numerator = result[available].fillna(0.0).mul(weight_series, axis=1).sum(axis=1)
    denominator = float(weight_series.abs().sum()) or 1.0

    result[output_col] = numerator / denominator
    return result


def compute_fraud_proxy_score(
    panel: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    output_col: str = "fraud_proxy_score",
) -> pd.DataFrame:
    """Backward-compatible alias for :func:`build_fraud_proxy_score`."""

    return build_fraud_proxy_score(
        panel=panel,
        feature_weights=feature_weights,
        output_col=output_col,
    )
