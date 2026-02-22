import pandas as pd

from secfsn_plus_decisionsystems.anomaly.rolling_instability import build_rolling_instability_score
from secfsn_plus_decisionsystems.anomaly.statistical_outliers import (
    build_cross_sectional_z_anomaly,
    build_robust_mad_anomaly,
)
from secfsn_plus_decisionsystems.scoring.fraud_proxy_score import build_fraud_proxy_score
from secfsn_plus_decisionsystems.scoring.hybrid_score import HybridRiskConfig, build_hybrid_risk_score
from secfsn_plus_decisionsystems.scoring.structural_risk_score import build_structural_risk_score
from secfsn_plus_decisionsystems.screening.short_candidates import build_short_candidates


def _sample_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (1001, "2024Q1"),
            (1002, "2024Q1"),
            (1003, "2024Q1"),
            (1001, "2024Q2"),
            (1002, "2024Q2"),
            (1003, "2024Q2"),
        ],
        names=["cik", "period"],
    )
    return pd.DataFrame(
        {
            "equity_ni_residual_scaled": [0.1, 0.2, 1.5, 0.1, 0.2, 1.6],
            "cash_vs_earnings_gap_scaled": [0.2, 0.1, 1.4, 0.2, 0.1, 1.5],
            "accrual_intensity": [0.5, 0.0, -0.2, 0.4, 0.1, -0.3],
            "structural_drift_1q": [0.0, 0.1, 0.8, 0.2, 0.1, 1.0],
            "margin_volatility_4q": [0.3, 0.2, 0.9, 0.3, 0.2, 1.1],
            "balance_sheet_identity_residual_scaled": [0.1, 0.0, 1.2, 0.1, 0.0, 1.3],
            "asset_revenue_growth_mismatch_scaled": [0.05, 0.10, 0.9, 0.06, 0.11, 1.0],
            "non_cash_earnings_ratio": [0.20, 0.10, 0.7, 0.21, 0.09, 0.8],
            "retained_earnings_gap_proxy_scaled": [0.01, 0.02, 0.7, 0.01, 0.03, 0.8],
            "model_drawdown_probability": [0.9, 0.1, 0.5, 0.85, 0.15, 0.55],
        },
        index=index,
    )


def test_scores_are_deterministic():
    panel = _sample_panel()
    cfg = HybridRiskConfig(
        structural_weight=0.4,
        fraud_weight=0.4,
        model_drawdown_weight=0.2,
        use_percentile=True,
        high_risk_threshold=0.8,
        medium_risk_threshold=0.5,
    )

    first = build_hybrid_risk_score(panel, cfg)
    second = build_hybrid_risk_score(panel, cfg)

    pd.testing.assert_frame_equal(first, second)


def test_scoring_does_not_mutate_input():
    panel = _sample_panel()
    baseline = panel.copy(deep=True)

    _ = build_structural_risk_score(panel)
    _ = build_fraud_proxy_score(panel)
    _ = build_hybrid_risk_score(panel, HybridRiskConfig())

    pd.testing.assert_frame_equal(panel, baseline)


def test_missing_columns_are_handled_gracefully():
    panel = _sample_panel()[["accrual_intensity"]]
    result = build_hybrid_risk_score(
        panel,
        HybridRiskConfig(
            use_percentile=False,
            high_risk_threshold=0.1,
            medium_risk_threshold=0.0,
        ),
    )

    assert "structural_risk_score" in result.columns
    assert "fraud_proxy_score" in result.columns
    assert "hybrid_risk_score" in result.columns
    assert "is_high_risk" in result.columns
    assert "is_medium_risk" in result.columns
    assert result["hybrid_risk_score"].notna().all()


def test_anomaly_builders_and_short_candidates_smoke():
    panel = _sample_panel()
    panel = build_cross_sectional_z_anomaly(
        panel,
        columns=("equity_ni_residual_scaled", "cash_vs_earnings_gap_scaled"),
        threshold=1.0,
    )
    panel = build_robust_mad_anomaly(
        panel,
        columns=("equity_ni_residual_scaled", "cash_vs_earnings_gap_scaled"),
        threshold=1.0,
    )
    panel = build_rolling_instability_score(
        panel,
        columns=("structural_drift_1q", "margin_volatility_4q"),
        window=2,
        threshold=0.2,
    )

    cfg = HybridRiskConfig(anomaly_weight=0.2)
    panel = build_hybrid_risk_score(panel, cfg)
    panel["is_structural_break"] = [False, False, True, False, False, True]

    ranked = build_short_candidates(panel, top_n=2)

    for col in (
        "cross_sectional_z_anomaly_magnitude",
        "is_cross_sectional_z_anomaly",
        "robust_mad_anomaly_magnitude",
        "is_robust_mad_anomaly",
        "rolling_instability_magnitude",
        "is_rolling_instability",
    ):
        assert col in panel.columns

    assert panel.index.equals(_sample_panel().index)
    assert "hybrid_risk_score" in panel.columns
    assert not ranked.empty
    assert len(ranked) == 2
    assert ranked["hybrid_risk_score"].is_monotonic_decreasing
