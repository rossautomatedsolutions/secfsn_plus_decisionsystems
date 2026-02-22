import pandas as pd

from secfsn_plus_decisionsystems.scoring.fraud_proxy_score import build_fraud_proxy_score
from secfsn_plus_decisionsystems.scoring.hybrid_score import HybridRiskConfig, build_hybrid_risk_score
from secfsn_plus_decisionsystems.scoring.structural_risk_score import build_structural_risk_score


def _sample_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(1001, "2024Q1"), (1002, "2024Q1"), (1003, "2024Q1")],
        names=["cik", "period"],
    )
    return pd.DataFrame(
        {
            "equity_ni_residual_scaled": [0.1, 0.2, 0.3],
            "cash_vs_earnings_gap_scaled": [0.2, 0.1, 0.0],
            "accrual_intensity": [0.5, 0.0, -0.2],
            "structural_drift_1q": [0.0, 0.1, 0.2],
            "margin_volatility_4q": [0.3, 0.2, 0.1],
            "balance_sheet_identity_residual_scaled": [0.1, 0.0, 0.3],
            "asset_revenue_growth_mismatch_scaled": [0.05, 0.10, 0.15],
            "non_cash_earnings_ratio": [0.20, 0.10, 0.00],
            "retained_earnings_gap_proxy_scaled": [0.01, 0.02, 0.03],
            "model_drawdown_probability": [0.9, 0.1, 0.5],
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
