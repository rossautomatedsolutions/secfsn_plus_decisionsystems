"""Rolling-instability diagnostics across periods per CIK."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems._panel_utils import sort_panel_by_cik_period, validate_panel_index


_DEFAULT_INSTABILITY_THRESHOLD = 1.0


def _validate_panel(panel: pd.DataFrame) -> None:
    validate_panel_index(panel)


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
    original_index = panel.index
    ordered = sort_panel_by_cik_period(panel)
    result = ordered.copy()
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
    return result.reindex(original_index)


def compute_rolling_instability(
    panel: pd.DataFrame,
    value_col: str,
    window: int = 4,
    output_col: str = "rolling_instability",
) -> pd.DataFrame:
    """Backward-compatible wrapper for single-column rolling instability metric."""

    original_index = panel.index
    ordered = sort_panel_by_cik_period(panel)

    public_flag_col = "is_rolling_instability"
    had_public_flag = public_flag_col in panel.columns
    legacy_flag_col = "_legacy_is_rolling_instability"
    result = build_rolling_instability_score(
        ordered,
        columns=(value_col,),
        window=window,
        magnitude_col=output_col,
        flag_col=legacy_flag_col,
    )
    if not had_public_flag and legacy_flag_col in result.columns:
        result[public_flag_col] = result[legacy_flag_col]
    result = result.drop(columns=[legacy_flag_col], errors="ignore")
    return result.reindex(original_index)
