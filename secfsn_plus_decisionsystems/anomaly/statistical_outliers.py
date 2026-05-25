"""Statistical outlier detection for deterministic anomaly flags."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems._panel_utils import validate_panel_index


_CROSS_SECTIONAL_THRESHOLD = 3.0
_ROBUST_SCALE_FACTOR = 0.6745


def _validate_panel(panel: pd.DataFrame) -> None:
    validate_panel_index(panel)


def _ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].apply(pd.to_numeric, errors="coerce")


def build_cross_sectional_z_anomaly(
    panel: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    threshold: float = _CROSS_SECTIONAL_THRESHOLD,
    magnitude_col: str = "cross_sectional_z_anomaly_magnitude",
    flag_col: str = "is_cross_sectional_z_anomaly",
) -> pd.DataFrame:
    """Build cross-sectional anomaly magnitude and flag using per-period z-scores."""

    _validate_panel(panel)
    result = panel.copy()
    available = [column for column in columns if column in result.columns]
    if not available:
        result[magnitude_col] = 0.0
        result[flag_col] = False
        return result

    values = _ensure_numeric(result, available)
    period_mean = values.groupby(level=1).transform("mean")
    period_std = values.groupby(level=1).transform(lambda s: s.std(ddof=0)).replace(0.0, pd.NA)
    z_scores = ((values - period_mean) / period_std).fillna(0.0)

    abs_z = z_scores.abs()
    result[magnitude_col] = abs_z.max(axis=1)
    result[flag_col] = abs_z.gt(float(threshold)).any(axis=1)
    return result


def build_robust_mad_anomaly(
    panel: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    threshold: float = _CROSS_SECTIONAL_THRESHOLD,
    magnitude_col: str = "robust_mad_anomaly_magnitude",
    flag_col: str = "is_robust_mad_anomaly",
) -> pd.DataFrame:
    """Build robust anomaly magnitude and flag using per-period median/MAD scores."""

    _validate_panel(panel)
    result = panel.copy()
    available = [column for column in columns if column in result.columns]
    if not available:
        result[magnitude_col] = 0.0
        result[flag_col] = False
        return result

    values = _ensure_numeric(result, available)
    period_median = values.groupby(level=1).transform("median")
    abs_dev = (values - period_median).abs()
    period_mad = abs_dev.groupby(level=1).transform("median").replace(0.0, pd.NA)
    robust_scores = ((_ROBUST_SCALE_FACTOR * (values - period_median)) / period_mad).fillna(0.0)

    abs_robust = robust_scores.abs()
    result[magnitude_col] = abs_robust.max(axis=1)
    result[flag_col] = abs_robust.gt(float(threshold)).any(axis=1)
    return result


def flag_statistical_outliers(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    z_threshold: float = 3.0,
    output_col: str = "is_statistical_outlier",
) -> pd.DataFrame:
    """Backward-compatible wrapper for cross-sectional z anomaly flags."""

    result = build_cross_sectional_z_anomaly(
        panel,
        columns=columns,
        threshold=z_threshold,
        magnitude_col="statistical_outlier_magnitude",
        flag_col=output_col,
    )
    return result
