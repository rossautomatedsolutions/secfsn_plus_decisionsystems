"""Stability diagnostics utilities for longitudinal signal behavior."""

from __future__ import annotations

import pandas as pd


def summarize_group_stability(
    panel: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Return per-``cik`` deterministic stability summaries for one column.

    Output columns include mean, standard deviation, and observation count.
    Missing input columns yield an empty summary DataFrame.
    """

    if value_col not in panel.columns:
        return pd.DataFrame(columns=["mean", "std", "count"])

    grouped = panel[value_col].groupby(level=0)
    return pd.DataFrame(
        {
            "mean": grouped.mean(),
            "std": grouped.std(ddof=0).fillna(0.0),
            "count": grouped.count(),
        }
    )
