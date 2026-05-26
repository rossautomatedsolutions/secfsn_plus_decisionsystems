import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from secfsn_plus_decisionsystems.anomaly.rolling_instability import (
    build_rolling_instability_score,
    compute_rolling_instability,
)
from secfsn_plus_decisionsystems.anomaly.statistical_outliers import (
    build_cross_sectional_z_anomaly,
    build_robust_mad_anomaly,
)
from secfsn_plus_decisionsystems.scoring.fraud_proxy_score import (
    build_fraud_proxy_score,
    compute_fraud_proxy_score,
)
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


def test_hybrid_risk_config_prefers_canonical_model_risk_column():
    panel = _sample_panel().rename(columns={"model_drawdown_probability": "model_risk_probability"})
    panel["model_drawdown_probability"] = 0.0

    result = build_hybrid_risk_score(
        panel,
        HybridRiskConfig(
            structural_weight=0.0,
            fraud_weight=0.0,
            model_drawdown_weight=1.0,
        ),
    )

    pd.testing.assert_series_equal(
        result["hybrid_risk_score"],
        panel["model_risk_probability"],
        check_names=False,
    )


def test_hybrid_risk_config_supports_legacy_model_drawdown_alias_when_canonical_missing():
    panel = _sample_panel()

    result = build_hybrid_risk_score(
        panel,
        HybridRiskConfig(
            structural_weight=0.0,
            fraud_weight=0.0,
            model_drawdown_weight=1.0,
        ),
    )

    pd.testing.assert_series_equal(
        result["hybrid_risk_score"],
        panel["model_drawdown_probability"],
        check_names=False,
    )


def test_empty_weight_configuration_is_respected():
    panel = _sample_panel()

    structural = build_structural_risk_score(panel, feature_weights={})
    fraud = build_fraud_proxy_score(panel, feature_weights={})

    assert structural["structural_risk_score"].eq(0.0).all()
    assert fraud["fraud_proxy_score"].eq(0.0).all()


def test_compute_fraud_proxy_score_supports_legacy_proxy_columns_argument():
    panel = _sample_panel()
    columns = ("accrual_intensity", "non_cash_earnings_ratio")

    result = compute_fraud_proxy_score(panel, columns)
    expected = panel[list(columns)].mean(axis=1)

    pd.testing.assert_series_equal(
        result["fraud_proxy_score"],
        expected,
        check_names=False,
    )


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


def test_obvious_cross_sectional_outlier_is_detected():
    index = pd.MultiIndex.from_tuples(
        [
            ("1001", "2024Q1"),
            ("1002", "2024Q1"),
            ("1003", "2024Q1"),
            ("1004", "2024Q1"),
        ],
        names=["cik", "period"],
    )
    panel = pd.DataFrame(
        {
            "equity_ni_residual_scaled": [0.0, 0.1, 0.0, 100.0],
        },
        index=index,
    )

    result = build_robust_mad_anomaly(
        panel,
        columns=("equity_ni_residual_scaled",),
        threshold=3.0,
    )

    assert result.loc[("1004", "2024Q1"), "is_robust_mad_anomaly"]
    assert not result.loc[("1001", "2024Q1"), "is_robust_mad_anomaly"]


def test_cross_sectional_z_anomaly_output_remains_numeric():
    panel = _sample_panel()
    panel.loc[(1001, "2024Q1"), "equity_ni_residual_scaled"] = None

    result = build_cross_sectional_z_anomaly(
        panel,
        columns=("equity_ni_residual_scaled", "cash_vs_earnings_gap_scaled"),
        threshold=1.0,
    )

    assert is_numeric_dtype(result["cross_sectional_z_anomaly_magnitude"])
    assert is_bool_dtype(result["is_cross_sectional_z_anomaly"])
    assert result["cross_sectional_z_anomaly_magnitude"].notna().all()


def test_robust_mad_anomaly_output_remains_numeric():
    panel = _sample_panel()
    panel.loc[(1002, "2024Q2"), "cash_vs_earnings_gap_scaled"] = None

    result = build_robust_mad_anomaly(
        panel,
        columns=("equity_ni_residual_scaled", "cash_vs_earnings_gap_scaled"),
        threshold=1.0,
    )

    assert is_numeric_dtype(result["robust_mad_anomaly_magnitude"])
    assert is_bool_dtype(result["is_robust_mad_anomaly"])
    assert result["robust_mad_anomaly_magnitude"].notna().all()


def test_statistical_outlier_builders_handle_missing_columns_gracefully():
    panel = _sample_panel()[["accrual_intensity"]]

    z_result = build_cross_sectional_z_anomaly(
        panel,
        columns=("equity_ni_residual_scaled",),
    )
    robust_result = build_robust_mad_anomaly(
        panel,
        columns=("cash_vs_earnings_gap_scaled",),
    )

    assert z_result["cross_sectional_z_anomaly_magnitude"].eq(0.0).all()
    assert robust_result["robust_mad_anomaly_magnitude"].eq(0.0).all()
    assert not z_result["is_cross_sectional_z_anomaly"].any()
    assert not robust_result["is_robust_mad_anomaly"].any()


def test_compute_rolling_instability_restores_public_flag_when_missing():
    panel = _sample_panel()

    result = compute_rolling_instability(panel, value_col="structural_drift_1q", window=2)

    assert "is_rolling_instability" in result.columns
    assert result["is_rolling_instability"].dtype == bool


def test_compute_rolling_instability_preserves_existing_public_flag():
    panel = _sample_panel()
    panel["is_rolling_instability"] = True

    result = compute_rolling_instability(panel, value_col="structural_drift_1q", window=2)

    assert "is_rolling_instability" in result.columns
    assert result["is_rolling_instability"].all()
