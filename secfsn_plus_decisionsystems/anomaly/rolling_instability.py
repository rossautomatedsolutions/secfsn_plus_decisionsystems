"""Rolling-instability diagnostics across periods per CIK."""

from __future__ import annotations

import pandas as pd


_DEFAULT_INSTABILITY_THRESHOLD = 1.0


def _validate_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
        raise ValueError("panel must be indexed by a (cik, period) MultiIndex")


def build_rolling_instability_score(
    panel: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    window: int = 4,
    threshold: float = _DEFAULT_INSTABILITY_THRESHOLD,
    magnitude_col: str = "rolling_instability_magnitude",
    flag_col: str = "is_rolling_instability",
) -> pd.DataFrame:
    """Build per-CIK rolling instability magnitude and flag from structural signals."""

    _validate_panel(panel)
    result = panel.copy()
    available = [column for column in columns if column in result.columns]
    if not available:
        result[magnitude_col] = 0.0
        result[flag_col] = False
        return result

    numeric = result[available].apply(pd.to_numeric, errors="coerce")
    abs_diff = numeric.groupby(level=0).diff().abs().fillna(0.0)
    acceleration = abs_diff.groupby(level=0).diff().abs().fillna(0.0)

    rolling_std = (
        abs_diff.groupby(level=0)
        .transform(lambda s: s.rolling(window=window, min_periods=1).std(ddof=0))
        .fillna(0.0)
    )
    rolling_acc = (
        acceleration.groupby(level=0)
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        .fillna(0.0)
    )

    combined = rolling_std + rolling_acc
    result[magnitude_col] = combined.max(axis=1)
    result[flag_col] = result[magnitude_col].gt(float(threshold))
    return result


def compute_rolling_instability(
    panel: pd.DataFrame,
    value_col: str,
    window: int = 4,
    output_col: str = "rolling_instability",
) -> pd.DataFrame:
    """Backward-compatible wrapper for single-column rolling instability metric."""

    legacy_flag_col = "_legacy_is_rolling_instability"
    result = build_rolling_instability_score(
        panel,
        columns=(value_col,),
        window=window,
        magnitude_col=output_col,
        flag_col=legacy_flag_col,
    )
    result = result.drop(columns=[legacy_flag_col], errors="ignore")
    return result
