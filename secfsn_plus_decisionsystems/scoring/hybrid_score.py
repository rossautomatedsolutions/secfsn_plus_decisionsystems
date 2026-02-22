"""Hybrid deterministic score combining structural and fraud risk components."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from secfsn_plus_decisionsystems.scoring.fraud_proxy_score import build_fraud_proxy_score
from secfsn_plus_decisionsystems.scoring.structural_risk_score import build_structural_risk_score


@dataclass(frozen=True)
class HybridRiskConfig:
    """Configuration for deterministic hybrid risk scoring."""

    structural_weight: float = 0.5
    fraud_weight: float = 0.5
    model_drawdown_weight: float = 0.0
    anomaly_weight: float = 0.0
    use_percentile: bool = False
    high_risk_threshold: float | None = None
    medium_risk_threshold: float | None = None
    structural_col: str = "structural_risk_score"
    fraud_col: str = "fraud_proxy_score"
    model_drawdown_col: str = "model_drawdown_probability"
    hybrid_col: str = "hybrid_risk_score"
    percentile_col: str = "hybrid_risk_percentile"
    high_flag_col: str = "is_high_risk"
    medium_flag_col: str = "is_medium_risk"
    anomaly_magnitude_cols: tuple[str, ...] = (
        "cross_sectional_z_anomaly_magnitude",
        "robust_mad_anomaly_magnitude",
        "rolling_instability_magnitude",
    )


def build_hybrid_risk_score(
    panel: pd.DataFrame,
    config: HybridRiskConfig | None = None,
) -> pd.DataFrame:
    """Build deterministic hybrid risk score with optional percentile + flags."""

    cfg = config or HybridRiskConfig()

    result = build_structural_risk_score(panel, output_col=cfg.structural_col)
    result = build_fraud_proxy_score(result, output_col=cfg.fraud_col)

    components: list[str] = []
    component_weights: dict[str, float] = {}

    if cfg.structural_col in result.columns:
        components.append(cfg.structural_col)
        component_weights[cfg.structural_col] = cfg.structural_weight
    if cfg.fraud_col in result.columns:
        components.append(cfg.fraud_col)
        component_weights[cfg.fraud_col] = cfg.fraud_weight
    if cfg.model_drawdown_col in result.columns:
        components.append(cfg.model_drawdown_col)
        component_weights[cfg.model_drawdown_col] = cfg.model_drawdown_weight

    available_anomaly_cols = [column for column in cfg.anomaly_magnitude_cols if column in result.columns]
    if available_anomaly_cols and cfg.anomaly_weight != 0.0:
        anomaly_blend_col = "_anomaly_blend_component"
        result[anomaly_blend_col] = result[available_anomaly_cols].fillna(0.0).mean(axis=1)
        components.append(anomaly_blend_col)
        component_weights[anomaly_blend_col] = cfg.anomaly_weight

    if not components:
        result[cfg.hybrid_col] = 0.0
    else:
        weight_series = pd.Series({column: float(component_weights[column]) for column in components})
        numerator = result[components].fillna(0.0).mul(weight_series, axis=1).sum(axis=1)
        denominator = float(weight_series.abs().sum()) or 1.0
        result[cfg.hybrid_col] = numerator / denominator

    if "_anomaly_blend_component" in result.columns:
        result = result.drop(columns=["_anomaly_blend_component"])

    if cfg.use_percentile:
        result[cfg.percentile_col] = result[cfg.hybrid_col].rank(method="average", pct=True)
        threshold_col = cfg.percentile_col
    else:
        threshold_col = cfg.hybrid_col

    if cfg.high_risk_threshold is not None:
        result[cfg.high_flag_col] = result[threshold_col] >= cfg.high_risk_threshold

    if cfg.medium_risk_threshold is not None:
        result[cfg.medium_flag_col] = result[threshold_col] >= cfg.medium_risk_threshold

    return result


def compute_hybrid_score(
    panel: pd.DataFrame,
    structural_col: str = "structural_risk_score",
    fraud_col: str = "fraud_proxy_score",
    model_col: str = "model_risk_probability",
    output_col: str = "hybrid_decision_score",
) -> pd.DataFrame:
    """Backward-compatible wrapper for legacy hybrid score API."""

    result = panel.copy()
    cols = [c for c in (structural_col, fraud_col, model_col) if c in result.columns]
    if not cols:
        result[output_col] = 0.0
        return result

    result[output_col] = result[cols].fillna(0.0).mean(axis=1)
    return result
