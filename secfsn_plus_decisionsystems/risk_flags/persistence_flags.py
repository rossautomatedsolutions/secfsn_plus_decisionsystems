"""Persistence-based flags for repeated risk behavior."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems._panel_utils import sort_panel_by_cik_period


def apply_persistence_flag(
    panel: pd.DataFrame,
    flag_col: str,
    min_consecutive: int = 2,
    output_col: str = "is_persistent_risk",
) -> pd.DataFrame:
    """Flag rows where ``flag_col`` has persisted for consecutive periods.

    Uses per-``cik`` cumulative streak counting in a fully vectorized manner.
    """

    original_index = panel.index
    result = sort_panel_by_cik_period(panel)
    if flag_col not in result.columns:
        result[output_col] = False
        return result.reindex(original_index)

    mask = result[flag_col].fillna(False).astype(bool)
    g = mask.groupby(level=0)
    streak = g.cumsum() - g.cumsum().where(~mask).ffill().fillna(0).astype(int)
    result[output_col] = streak.ge(min_consecutive)
    return result.reindex(original_index)
