import pandas as pd

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine
from secfsn_plus_decisionsystems.risk_flags.persistence_flags import apply_persistence_flag
from secfsn_plus_decisionsystems.screening.put_candidates import screen_put_candidates
from secfsn_plus_decisionsystems.screening.risk_filtered_longs import screen_risk_filtered_longs
from secfsn_plus_decisionsystems.screening.short_candidates import screen_short_candidates


def _engine_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            (1001, "2024Q1"),
            (1002, "2024Q1"),
            (1001, "2024Q2"),
            (1002, "2024Q2"),
            (1001, "2024Q3"),
            (1002, "2024Q3"),
        ],
        names=["cik", "period"],
    )
    return pd.DataFrame(
        {
            "structural_risk_score": [0.10, 0.30, 0.35, 0.20, 0.85, 0.25],
            "fraud_proxy_score": [0.05, 0.20, 0.40, 0.20, 0.90, 0.10],
            "model_risk_probability": [0.08, 0.25, 0.20, 0.25, 0.95, 0.20],
        },
        index=index,
    )


def test_run_decision_engine_is_deterministic_and_does_not_mutate_input():
    panel = _engine_panel()
    baseline = panel.copy(deep=True)

    first = run_decision_engine(panel)
    second = run_decision_engine(panel)

    pd.testing.assert_frame_equal(panel, baseline)
    pd.testing.assert_frame_equal(first, second)


def test_run_decision_engine_returns_required_columns():
    panel = _engine_panel()

    result = run_decision_engine(panel)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "high_structural_risk_flag",
        "high_fraud_proxy_flag",
        "high_hybrid_risk_flag",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }

    assert required_columns.issubset(result.columns)
    assert result.index.equals(panel.index)


def test_run_decision_engine_handles_missing_columns_gracefully():
    panel = _engine_panel()[["model_risk_probability"]].copy()

    result = run_decision_engine(panel)

    assert "structural_risk_score" in result.columns
    assert "fraud_proxy_score" in result.columns
    assert "hybrid_risk_score" in result.columns
    assert result["risk_override"].isin([True, False]).all()


def test_run_decision_engine_preserves_legacy_instability_output_for_put_screening():
    panel = _engine_panel()

    result = run_decision_engine(panel)

    assert "rolling_instability" in result.columns
    screened = screen_put_candidates(result, min_instability=0.01)

    assert not screened.empty


def test_screeners_consume_canonical_decision_outputs():
    panel = _engine_panel()

    result = run_decision_engine(panel)

    short_candidates = screen_short_candidates(result)
    long_candidates = screen_risk_filtered_longs(result)

    pd.testing.assert_index_equal(short_candidates.index, result.index[result["short_allowed"]])
    pd.testing.assert_index_equal(long_candidates.index, result.index[result["long_allowed"]])


def test_legacy_screening_alias_does_not_override_canonical_outputs():
    panel = _engine_panel()

    result = run_decision_engine(panel)
    result["hybrid_decision_score"] = 0.0

    short_candidates = screen_short_candidates(result)
    long_candidates = screen_risk_filtered_longs(result)

    pd.testing.assert_index_equal(short_candidates.index, result.index[result["short_allowed"]])
    pd.testing.assert_index_equal(long_candidates.index, result.index[result["long_allowed"]])


def test_run_decision_engine_keeps_component_rolling_instability_flag():
    panel = _engine_panel()

    result = run_decision_engine(panel)

    expected_flag = panel.copy()
    expected_flag = expected_flag.assign(
        hybrid_decision_score=(
            expected_flag["structural_risk_score"]
            + expected_flag["fraud_proxy_score"]
            + expected_flag["model_risk_probability"]
        )
        / 3.0
    )

    from secfsn_plus_decisionsystems.anomaly.rolling_instability import build_rolling_instability_score

    expected_flag = build_rolling_instability_score(
        expected_flag,
        columns=("structural_risk_score", "fraud_proxy_score", "model_risk_probability"),
    )

    pd.testing.assert_series_equal(
        result["is_rolling_instability"],
        expected_flag["is_rolling_instability"],
        check_names=False,
    )


def test_persistence_flag_uses_period_order_within_each_cik():
    index = pd.MultiIndex.from_tuples(
        [
            ("1001", "2024Q2"),
            ("1001", "2024Q1"),
            ("1001", "2024Q3"),
            ("1002", "2024Q1"),
            ("1002", "2024Q2"),
        ],
        names=["cik", "period"],
    )
    panel = pd.DataFrame(
        {
            "high_hybrid_risk_flag": [True, True, True, False, True],
        },
        index=index,
    )

    result = apply_persistence_flag(panel, "high_hybrid_risk_flag", min_consecutive=2)

    expected = pd.Series([True, False, True, False, False], index=index, name="is_persistent_risk")
    pd.testing.assert_series_equal(result["is_persistent_risk"], expected)
