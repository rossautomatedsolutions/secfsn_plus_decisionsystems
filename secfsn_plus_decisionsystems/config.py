"""Configuration objects for deterministic decision system components."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionSystemConfig:
    """Container for default parameters used by deterministic decision modules.

    Attributes:
        min_history_periods: Minimum observations required for persistence checks.
        zscore_threshold: Default absolute z-score cutoff for outlier detection.
        instability_window: Rolling window length for instability diagnostics.
        high_risk_cutoff: Generic high-risk score threshold used by screeners.
        long_risk_cutoff: Maximum hybrid risk score allowed for long candidates.
        short_risk_cutoff: Minimum hybrid risk score required for short candidates.
        persistence_min_consecutive: Consecutive flagged periods required to raise
            the persistence override.
    """

    min_history_periods: int = 4
    zscore_threshold: float = 3.0
    instability_window: int = 4
    high_risk_cutoff: float = 0.7
    long_risk_cutoff: float = 0.3
    short_risk_cutoff: float = 0.7
    persistence_min_consecutive: int = 2
