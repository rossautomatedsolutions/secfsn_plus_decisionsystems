"""Top-level deterministic decision engine orchestration."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems.anomaly.rolling_instability import build_rolling_instability_score
from secfsn_plus_decisionsystems.anomaly.statistical_outliers import (
    build_cross_sectional_z_anomaly,
    build_robust_mad_anomaly,
)
from secfsn_plus_decisionsystems.risk_flags.threshold_flags import apply_threshold_flag
from secfsn_plus_decisionsystems.scoring.hybrid_score import build_hybrid_risk_score, compute_hybrid_score


def run_decision_engine(panel: pd.DataFrame) -> pd.DataFrame:
    """Run a deterministic decision pipeline with anomaly-aware hybrid scoring."""

    result = panel.copy()

    anomaly_inputs = (
        "structural_risk_score",
        "fraud_proxy_score",
        "model_risk_probability",
    )
    result = build_cross_sectional_z_anomaly(result, columns=anomaly_inputs)
    result = build_robust_mad_anomaly(result, columns=anomaly_inputs)
    result = build_rolling_instability_score(result, columns=anomaly_inputs)

    result["is_anomaly_flagged"] = (
        result.get("is_cross_sectional_z_anomaly", False)
        | result.get("is_robust_mad_anomaly", False)
        | result.get("is_instability_anomaly", False)
    )

    result = build_hybrid_risk_score(result)
    # Backward compatibility for existing integrations.
    result = compute_hybrid_score(result)

    result = apply_threshold_flag(result, score_col="hybrid_risk_score", threshold=0.7)
    result = apply_threshold_flag(
        result,
        score_col="structural_risk_score",
        threshold=0.7,
        output_col="is_high_structural_risk",
    )
    return result
