"""Rules-based generation of put-option candidate universe."""

from __future__ import annotations

import pandas as pd


def screen_put_candidates(
    panel: pd.DataFrame,
    instability_col: str = "rolling_instability",
    min_instability: float = 0.2,
) -> pd.DataFrame:
    """Return deterministic put candidates from instability diagnostics."""

    if instability_col not in panel.columns:
        return panel.iloc[0:0].copy()
    return panel.loc[
        panel[instability_col].fillna(float("-inf")).ge(min_instability)
    ].copy()
