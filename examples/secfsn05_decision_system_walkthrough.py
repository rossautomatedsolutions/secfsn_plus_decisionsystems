"""Tiny SECFSN 05 / Layer 3B walkthrough for a short screen recording."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine
from secfsn_plus_decisionsystems.integration.model_bridge import merge_model_outputs

FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "upstream_prepared_panel.csv"
DISPLAY_COLUMNS: tuple[str, ...] = (
    "structural_risk_score",
    "fraud_proxy_score",
    "model_risk_probability",
    "hybrid_risk_score",
    "long_allowed",
    "short_allowed",
    "risk_override",
)
PREVIEW_COLUMNS: tuple[str, ...] = (
    "equity_ni_residual_scaled",
    "cash_vs_earnings_gap_scaled",
    "accrual_intensity",
)


def load_prepared_panel() -> pd.DataFrame:
    """Load the committed upstream-shaped fixture and enforce the SECFSN 05 index."""

    fixture = pd.read_csv(FIXTURE_PATH, dtype={"cik": str, "period": str})
    return fixture.set_index(["cik", "period"]).sort_index()


def build_partial_model_outputs(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a tiny optional SECFSN 04 / Layer 3A frame with partial coverage."""

    covered_index = pd.MultiIndex.from_tuples(
        [("1001", "2024Q2"), ("2002", "2024Q2")],
        names=panel.index.names,
    )
    return pd.DataFrame(
        {"model_risk_probability": [0.18, 0.93]},
        index=covered_index,
    )


def format_final_table(result: pd.DataFrame) -> pd.DataFrame:
    """Keep the screen-recording view compact and keyed by CIK/period."""

    available_columns = [column for column in DISPLAY_COLUMNS if column in result.columns]
    compact = result.loc[:, available_columns].copy()
    if "model_risk_probability" not in compact.columns:
        compact.insert(2, "model_risk_probability", pd.NA)
    return compact.loc[:, list(DISPLAY_COLUMNS)].round(3)


def format_prepared_panel_preview(panel: pd.DataFrame) -> pd.DataFrame:
    """Show only the smallest useful upstream view for the walkthrough."""

    return panel.loc[:, list(PREVIEW_COLUMNS)].round(3)


def main() -> None:
    """Walk through deterministic-only and optional-model SECFSN 05 usage."""

    print("SECFSN 05 demo: prepared panel -> optional model bridge -> decision engine")
    print()
    print("SECFSN 01-04 live upstream and provide prepared evidence plus optional model outputs.")
    print("SECFSN 05 consumes available evidence without retraining or rewriting upstream logic.")
    print()

    prepared_panel = load_prepared_panel()
    print("1. Load the committed upstream-shaped fixture")
    print(f"   fixture: {FIXTURE_PATH.relative_to(REPO_ROOT)}")
    print(f"   rows: {len(prepared_panel)}")
    print(f"   index: {prepared_panel.index.names}")
    print()
    print("Prepared panel preview (compact):")
    print(format_prepared_panel_preview(prepared_panel).to_string())
    print()

    deterministic_only = run_decision_engine(prepared_panel)
    print("2. Run deterministic-only SECFSN 05")
    print("   model_risk_probability is optional; SECFSN 05 still runs when it is absent.")
    print(format_final_table(deterministic_only).to_string())
    print()

    model_outputs = build_partial_model_outputs(prepared_panel)
    print("3. Add a tiny optional SECFSN 04 model-output frame")
    print("   Partial model coverage is allowed; uncovered rows simply keep null model risk values.")
    print(model_outputs.to_string())
    print()

    merged = merge_model_outputs(prepared_panel, model_outputs)
    with_optional_model = run_decision_engine(merged)
    print("4. Merge model outputs, then run the same SECFSN 05 engine")
    print("   Output remains keyed by the same (cik, period) MultiIndex.")
    print(format_final_table(with_optional_model).to_string())


if __name__ == "__main__":
    main()
