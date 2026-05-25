"""Top-level deterministic decision engine orchestration."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems.anomaly.rolling_instability import (
    build_rolling_instability_score,
    compute_rolling_instability,
)
from secfsn_plus_decisionsystems.anomaly.statistical_outliers import (
    build_cross_sectional_z_anomaly,
    build_robust_mad_anomaly,
)
from secfsn_plus_decisionsystems.config import DecisionSystemConfig
from secfsn_plus_decisionsystems.risk_flags.persistence_flags import apply_persistence_flag
from secfsn_plus_decisionsystems.scoring.fraud_proxy_score import build_fraud_proxy_score
from secfsn_plus_decisionsystems.scoring.hybrid_score import HybridRiskConfig, build_hybrid_risk_score
from secfsn_plus_decisionsystems.scoring.structural_risk_score import build_structural_risk_score


def run_decision_engine(
    panel: pd.DataFrame,
    config: DecisionSystemConfig | None = None,
) -> pd.DataFrame:
    """Run the deterministic Layer 3B decision pipeline on a prepared panel."""

    cfg = config or DecisionSystemConfig()

    result = build_structural_risk_score(panel, output_col="structural_risk_score")
    result = build_fraud_proxy_score(result, output_col="fraud_proxy_score")
    result = build_hybrid_risk_score(
        result,
        HybridRiskConfig(
            structural_col="structural_risk_score",
            fraud_col="fraud_proxy_score",
            model_risk_col="model_risk_probability",
            hybrid_col="hybrid_risk_score",
        ),
    )

    anomaly_inputs = (
        "structural_risk_score",
        "fraud_proxy_score",
        "model_risk_probability",
    )
    result = build_cross_sectional_z_anomaly(
        result,
        columns=anomaly_inputs,
        threshold=cfg.zscore_threshold,
    )
    result = build_robust_mad_anomaly(
        result,
        columns=anomaly_inputs,
        threshold=cfg.zscore_threshold,
    )
    result = build_rolling_instability_score(
        result,
        columns=anomaly_inputs,
        window=cfg.instability_window,
        threshold=cfg.high_risk_cutoff,
    )

    # Preserve the legacy column for existing put-screening integrations.
    result = compute_rolling_instability(
        result,
        value_col="hybrid_risk_score",
        window=cfg.instability_window,
    )

    result["high_structural_risk_flag"] = result["structural_risk_score"].ge(cfg.high_risk_cutoff)
    result["high_fraud_proxy_flag"] = result["fraud_proxy_score"].ge(cfg.high_risk_cutoff)
    result["high_hybrid_risk_flag"] = result["hybrid_risk_score"].ge(cfg.high_risk_cutoff)

    result["high_anomaly_flag"] = result[
        ["is_cross_sectional_z_anomaly", "is_robust_mad_anomaly", "is_rolling_instability"]
    ].any(axis=1)
    result = apply_persistence_flag(
        result,
        flag_col="high_hybrid_risk_flag",
        min_consecutive=cfg.persistence_min_consecutive,
        output_col="persistent_high_hybrid_risk_flag",
    )

    result["risk_override"] = result["high_anomaly_flag"] | result["persistent_high_hybrid_risk_flag"]
    result["short_allowed"] = result["hybrid_risk_score"].ge(cfg.short_risk_cutoff) | result["risk_override"]
    result["long_allowed"] = (
        result["hybrid_risk_score"].le(cfg.long_risk_cutoff) & ~result["risk_override"]
    )

    result["hybrid_decision_score"] = result["hybrid_risk_score"]
    return result
