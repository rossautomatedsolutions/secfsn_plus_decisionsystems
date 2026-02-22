"""Statistical outlier detection for deterministic anomaly flags."""

from __future__ import annotations

import pandas as pd


def flag_statistical_outliers(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    z_threshold: float = 3.0,
    output_col: str = "is_statistical_outlier",
) -> pd.DataFrame:
    """Flag row-level outliers using cross-sectional z-scores.

    For each selected column, computes z-scores over the full panel and flags a
    row if any absolute z-score exceeds ``z_threshold``.

    Missing columns are ignored gracefully.
    """

    result = panel.copy()
    available = [c for c in columns if c in result.columns]
    if not available:
        result[output_col] = False
        return result

    z = result[available].apply(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)
    )
    result[output_col] = z.abs().gt(z_threshold).any(axis=1)
    return result
