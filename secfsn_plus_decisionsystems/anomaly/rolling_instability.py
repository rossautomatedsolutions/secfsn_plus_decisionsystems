"""Rolling-instability diagnostics across periods per CIK."""

from __future__ import annotations

import pandas as pd


def build_rolling_instability_score(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    window: int = 4,
    score_col: str = "rolling_instability_score",
    flag_col: str = "is_instability_anomaly",
    instability_threshold: float = 2.0,
) -> pd.DataFrame:
    """Measure rolling instability from per-CIK acceleration in structural signals.

    For each selected column and ``cik`` group:
      1) compute first differences over period order,
      2) compute acceleration as absolute change in first differences,
      3) compute rolling mean of acceleration with the given ``window``.

    Row-level instability score is the maximum rolling acceleration magnitude
    across selected columns.
    """

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be indexed by a 2-level MultiIndex: (cik, period)")

    result = panel.copy()
    available = [c for c in columns if c in result.columns]
    if not available:
        result[score_col] = 0.0
        result[flag_col] = False
        return result

    scores = pd.DataFrame(index=result.index)
    for col in available:
        numeric = pd.to_numeric(result[col], errors="coerce")
        first_diff = numeric.groupby(level=0, group_keys=False).diff()
        acceleration = first_diff.groupby(level=0, group_keys=False).diff().abs()
        rolling_acc = acceleration.groupby(level=0, group_keys=False).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        scores[col] = rolling_acc.fillna(0.0)

    instability = scores.max(axis=1).fillna(0.0)
    result[score_col] = instability
    result[flag_col] = instability.ge(instability_threshold)
    return result


def compute_rolling_instability(
    panel: pd.DataFrame,
    value_col: str,
    window: int = 4,
    output_col: str = "rolling_instability",
) -> pd.DataFrame:
    """Backward-compatible rolling standard deviation instability proxy."""

    result = panel.copy()
    if value_col not in result.columns:
        result[output_col] = 0.0
        return result

    grouped = result[value_col].groupby(level=0)
    result[output_col] = grouped.transform(
        lambda s: s.astype(float).rolling(window=window, min_periods=1).std().fillna(0.0)
    )
    return result
