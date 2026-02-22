"""Deterministic structural-risk scoring on prepared feature panels."""

from __future__ import annotations

import pandas as pd


def compute_structural_risk_score(
    panel: pd.DataFrame,
    feature_weights: dict[str, float] | None = None,
    output_col: str = "structural_risk_score",
) -> pd.DataFrame:
    """Compute a deterministic weighted score from existing structural features.

    The function expects a MultiIndex DataFrame keyed by ``(cik, period)`` and
    combines available columns in ``feature_weights`` via a normalized weighted
    average. Missing columns are ignored gracefully.

    Args:
        panel: Prepared feature panel indexed by ``(cik, period)``.
        feature_weights: Mapping of feature name to deterministic weight.
            Defaults to an empty mapping, resulting in a zero score.
        output_col: Name of the generated score column.

    Returns:
        A new DataFrame with ``output_col`` added. Input is never mutated.

    Notes:
        - Fully vectorized over rows.
        - Deterministic (no random state).
        - Extension point: replace static weights with versioned policy tables.
    """

    result = panel.copy()
    weights = feature_weights or {}

    available = [c for c in weights if c in result.columns]
    if not available:
        result[output_col] = 0.0
        return result

    weight_series = pd.Series({c: float(weights[c]) for c in available})
    numer = result[available].fillna(0.0).mul(weight_series, axis=1).sum(axis=1)
    denom = abs(weight_series).sum() or 1.0
    result[output_col] = numer / denom
    return result
