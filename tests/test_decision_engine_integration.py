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
