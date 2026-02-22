"""Statistical outlier detection for deterministic anomaly scoring."""

from __future__ import annotations

import pandas as pd


def _validate_multiindex(panel: pd.DataFrame) -> None:
    """Validate the expected ``(cik, period)`` MultiIndex contract."""

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel must be indexed by a 2-level MultiIndex: (cik, period)")


def build_cross_sectional_z_anomaly(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    z_threshold: float = 3.0,
    score_col: str = "cross_sectional_z_anomaly_score",
    flag_col: str = "is_cross_sectional_z_anomaly",
) -> pd.DataFrame:
    """Build period-wise cross-sectional z-score anomaly magnitude and flag.

    For each period and selected column, z-scores are computed across available
    entities. The row-level anomaly magnitude is the maximum absolute z-score
    across selected columns for that row. A row is flagged when the magnitude
    exceeds ``z_threshold``.
    """

    _validate_multiindex(panel)

    result = panel.copy()
    available = [c for c in columns if c in result.columns]
    if not available:
        result[score_col] = 0.0
        result[flag_col] = False
        return result

    def _zscore(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        std = numeric.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return (numeric - numeric.mean()) / std

    z_scores = result[available].groupby(level=1, group_keys=False).transform(_zscore)
    magnitude = z_scores.abs().max(axis=1).fillna(0.0)
    result[score_col] = magnitude
    result[flag_col] = magnitude.gt(z_threshold)
    return result


def build_robust_mad_anomaly(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    mad_threshold: float = 3.5,
    score_col: str = "robust_mad_anomaly_score",
    flag_col: str = "is_robust_mad_anomaly",
) -> pd.DataFrame:
    """Build robust outlier score using period-wise median absolute deviation.

    The robust z-score is defined as ``0.6745 * (x - median) / MAD`` computed
    cross-sectionally per period. The row-level anomaly magnitude is the maximum
    absolute robust z-score across selected columns.
    """

    _validate_multiindex(panel)

    result = panel.copy()
    available = [c for c in columns if c in result.columns]
    if not available:
        result[score_col] = 0.0
        result[flag_col] = False
        return result

    def _robust_z(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        median = numeric.median()
        mad = (numeric - median).abs().median()
        if pd.isna(mad) or mad == 0:
            return pd.Series(0.0, index=series.index)
        return 0.6745 * (numeric - median) / mad

    robust_scores = result[available].groupby(level=1, group_keys=False).transform(_robust_z)
    magnitude = robust_scores.abs().max(axis=1).fillna(0.0)
    result[score_col] = magnitude
    result[flag_col] = magnitude.gt(mad_threshold)
    return result


def flag_statistical_outliers(
    panel: pd.DataFrame,
    columns: tuple[str, ...],
    z_threshold: float = 3.0,
    output_col: str = "is_statistical_outlier",
) -> pd.DataFrame:
    """Backward-compatible outlier flag derived from cross-sectional z anomaly."""

    result = build_cross_sectional_z_anomaly(panel, columns=columns, z_threshold=z_threshold)
    result[output_col] = result["is_cross_sectional_z_anomaly"]
    return result
