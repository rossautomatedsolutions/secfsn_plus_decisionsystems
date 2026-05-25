"""Small deterministic smoke test for first real panel integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine

CANONICAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "structural_risk_score",
    "fraud_proxy_score",
    "hybrid_risk_score",
    "high_structural_risk_flag",
    "high_fraud_proxy_flag",
    "high_hybrid_risk_flag",
    "long_allowed",
    "short_allowed",
    "risk_override",
)


def build_sample_feature_panel() -> pd.DataFrame:
    """Return a small prepared Layer 2.5-style panel for smoke testing."""

    index = pd.MultiIndex.from_tuples(
        [
            ("1001", "2024Q1"),
            ("1001", "2024Q2"),
            ("1001", "2024Q3"),
            ("2002", "2024Q1"),
            ("2002", "2024Q2"),
            ("2002", "2024Q3"),
        ],
        names=["cik", "period"],
    )
    return pd.DataFrame(
        {
            "equity_ni_residual_scaled": [0.05, 0.07, 0.10, 0.45, 0.78, 0.92],
            "cash_vs_earnings_gap_scaled": [0.08, 0.09, 0.12, 0.50, 0.82, 0.95],
            "accrual_intensity": [0.10, 0.11, 0.13, 0.55, 0.85, 0.98],
            "structural_drift_1q": [0.04, 0.05, 0.06, 0.50, 0.88, 1.05],
            "margin_volatility_4q": [0.06, 0.07, 0.08, 0.48, 0.84, 0.96],
            "balance_sheet_identity_residual_scaled": [0.03, 0.04, 0.05, 0.46, 0.80, 0.94],
            "retained_earnings_gap_proxy_scaled": [0.05, 0.06, 0.07, 0.40, 0.75, 0.90],
            "model_risk_probability": [0.10, 0.12, 0.15, 0.60, 0.90, 0.97],
        },
        index=index,
    )


def run_integration_smoke_example() -> pd.DataFrame:
    """Run the decision engine on a realistic prepared sample panel."""

    panel = build_sample_feature_panel()
    return run_decision_engine(panel)


def format_canonical_outputs(result: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical decision columns for compact display."""

    return result.loc[:, list(CANONICAL_OUTPUT_COLUMNS)].copy()


def main() -> None:
    """Print canonical decision-engine outputs for the sample panel."""

    result = run_integration_smoke_example()
    print(format_canonical_outputs(result).to_string())


if __name__ == "__main__":
    main()
