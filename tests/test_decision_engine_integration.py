import pandas as pd

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine
from secfsn_plus_decisionsystems.screening.put_candidates import screen_put_candidates


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


def test_run_decision_engine_preserves_legacy_instability_output_for_put_screening():
    panel = _engine_panel()

    result = run_decision_engine(panel)

    assert "rolling_instability" in result.columns
    screened = screen_put_candidates(result, min_instability=0.01)

    assert not screened.empty


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
