"""Rolling-instability diagnostics across periods per CIK."""

from __future__ import annotations

import pandas as pd


def compute_rolling_instability(
    panel: pd.DataFrame,
    value_col: str,
    window: int = 4,
    output_col: str = "rolling_instability",
) -> pd.DataFrame:
    """Compute per-entity rolling standard deviation as instability proxy.

    Args:
        panel: MultiIndex DataFrame indexed by ``(cik, period)``.
        value_col: Column to evaluate for instability.
        window: Rolling window length per ``cik`` group.
        output_col: Name of generated instability column.

    Returns:
        New DataFrame with rolling instability metric.
    """

    result = panel.copy()
    if value_col not in result.columns:
        result[output_col] = 0.0
        return result

    grouped = result[value_col].groupby(level=0)
    result[output_col] = grouped.transform(
        lambda s: s.astype(float).rolling(window=window, min_periods=1).std().fillna(0.0)
    )
    return result
