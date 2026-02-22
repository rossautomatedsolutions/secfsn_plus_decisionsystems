"""Distribution diagnostics utilities for deterministic QA."""

from __future__ import annotations

import pandas as pd


def summarize_distribution(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    """Return deterministic summary stats for selected columns.

    Missing columns are ignored, and an empty summary is returned when none are
    available.
    """

    available = [c for c in columns if c in panel.columns]
    if not available:
        return pd.DataFrame(columns=["count", "mean", "std", "min", "max"])

    summary = panel[available].describe().T.loc[:, ["count", "mean", "std", "min", "max"]]
    return summary
