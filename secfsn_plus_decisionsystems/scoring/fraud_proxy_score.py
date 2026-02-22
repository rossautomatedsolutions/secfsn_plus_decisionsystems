"""Deterministic fraud-proxy scoring on prepared panels."""

from __future__ import annotations

import pandas as pd


def compute_fraud_proxy_score(
    panel: pd.DataFrame,
    proxy_columns: tuple[str, ...] = ("accruals_signal", "restatement_signal"),
    output_col: str = "fraud_proxy_score",
) -> pd.DataFrame:
    """Compute a simple deterministic fraud-proxy score.

    Available proxy columns are averaged row-wise. Missing columns are skipped,
    and missing values are treated as neutral (0.0).

    Args:
        panel: MultiIndex ``(cik, period)`` feature DataFrame.
        proxy_columns: Candidate proxy columns expected from upstream layers.
        output_col: Output score column name.

    Returns:
        New DataFrame containing ``output_col``.
    """

    result = panel.copy()
    available = [c for c in proxy_columns if c in result.columns]
    if not available:
        result[output_col] = 0.0
        return result

    result[output_col] = result[available].fillna(0.0).mean(axis=1)
    return result
