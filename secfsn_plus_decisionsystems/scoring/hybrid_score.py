"""Hybrid deterministic score combining multiple precomputed components."""

from __future__ import annotations

import pandas as pd


def compute_hybrid_score(
    panel: pd.DataFrame,
    structural_col: str = "structural_risk_score",
    fraud_col: str = "fraud_proxy_score",
    model_col: str = "model_risk_probability",
    output_col: str = "hybrid_decision_score",
) -> pd.DataFrame:
    """Combine deterministic and optional model signals into one hybrid score.

    This function does not train or infer models. It only consumes columns if
    they already exist in the incoming panel.

    Args:
        panel: MultiIndex ``(cik, period)`` feature DataFrame.
        structural_col: Deterministic structural score column.
        fraud_col: Deterministic fraud-proxy score column.
        model_col: Optional Layer 3A model output column.
        output_col: Output hybrid score column name.

    Returns:
        New DataFrame with ``output_col``.
    """

    result = panel.copy()
    cols = [c for c in (structural_col, fraud_col, model_col) if c in result.columns]
    if not cols:
        result[output_col] = 0.0
        return result

    result[output_col] = result[cols].fillna(0.0).mean(axis=1)
    return result
