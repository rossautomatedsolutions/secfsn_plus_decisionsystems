"""Threshold-based deterministic risk flags."""

from __future__ import annotations

import pandas as pd


def apply_threshold_flag(
    panel: pd.DataFrame,
    score_col: str,
    threshold: float,
    output_col: str = "is_high_risk_threshold",
) -> pd.DataFrame:
    """Flag rows where a score exceeds a deterministic threshold.

    Missing ``score_col`` yields all-False output.
    """

    result = panel.copy()
    if score_col not in result.columns:
        result[output_col] = False
        return result

    result[output_col] = result[score_col].fillna(float("-inf")).ge(threshold)
    return result
