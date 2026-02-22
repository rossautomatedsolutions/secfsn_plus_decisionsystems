"""Hybrid deterministic score combining multiple precomputed components."""

from __future__ import annotations

import pandas as pd


def _clamp_weight(weight: float) -> float:
    """Clamp blend weights into [0, 1] for deterministic composition."""

    return min(1.0, max(0.0, float(weight)))


def build_hybrid_risk_score(
    panel: pd.DataFrame,
    structural_col: str = "structural_risk_score",
    fraud_col: str = "fraud_proxy_score",
    model_col: str = "model_risk_probability",
    anomaly_cols: tuple[str, ...] = (
        "cross_sectional_z_anomaly_score",
        "robust_mad_anomaly_score",
        "rolling_instability_score",
    ),
    anomaly_weight: float = 0.25,
    output_col: str = "hybrid_risk_score",
) -> pd.DataFrame:
    """Build deterministic hybrid risk score with optional anomaly contribution.

    Base score is the mean of available structural/fraud/model components.
    If anomaly magnitudes are present, their mean is blended in using
    ``anomaly_weight`` to emphasize instability/outlier severity.
    """

    result = panel.copy()
    base_cols = [c for c in (structural_col, fraud_col, model_col) if c in result.columns]

    if base_cols:
        base_score = result[base_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
    else:
        base_score = pd.Series(0.0, index=result.index)

    weight = _clamp_weight(anomaly_weight)
    available_anomaly_cols = [c for c in anomaly_cols if c in result.columns]
    if available_anomaly_cols and weight > 0.0:
        anomaly_magnitude = (
            result[available_anomaly_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)
        )
        result[output_col] = ((1.0 - weight) * base_score) + (weight * anomaly_magnitude)
    else:
        result[output_col] = base_score

    return result


def compute_hybrid_score(
    panel: pd.DataFrame,
    structural_col: str = "structural_risk_score",
    fraud_col: str = "fraud_proxy_score",
    model_col: str = "model_risk_probability",
    output_col: str = "hybrid_decision_score",
) -> pd.DataFrame:
    """Backward-compatible wrapper for prior hybrid score naming."""

    result = build_hybrid_risk_score(
        panel,
        structural_col=structural_col,
        fraud_col=fraud_col,
        model_col=model_col,
        output_col=output_col,
    )
    return result
