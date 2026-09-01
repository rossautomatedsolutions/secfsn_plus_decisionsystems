from pathlib import Path

import pandas as pd

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine
from secfsn_plus_decisionsystems.integration.model_bridge import merge_model_outputs
from secfsn_plus_decisionsystems.risk_flags.persistence_flags import apply_persistence_flag
from secfsn_plus_decisionsystems.screening.put_candidates import screen_put_candidates
from secfsn_plus_decisionsystems.screening.risk_filtered_longs import screen_risk_filtered_longs
from secfsn_plus_decisionsystems.screening.short_candidates import screen_short_candidates

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


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


def test_canonical_secfsn05_path_merges_optional_secfsn04_outputs_before_decision_engine():
    index = pd.MultiIndex.from_tuples(
        [
            ("1001", "2024Q1"),
            ("1001", "2024Q2"),
            ("1002", "2024Q1"),
        ],
        names=["cik", "period"],
    )
    prepared_panel = pd.DataFrame(
        {
            "equity_ni_residual_scaled": [0.10, 0.20, 0.90],
            "cash_vs_earnings_gap_scaled": [0.15, 0.25, 0.85],
            "accrual_intensity": [0.12, 0.22, 0.80],
        },
        index=index,
    )
    model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.05, 0.55, 0.95]},
        index=index,
    )

    merged = merge_model_outputs(prepared_panel, model_outputs)
    result = run_decision_engine(merged)

    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    pd.testing.assert_series_equal(
        result["model_risk_probability"],
        model_outputs["model_risk_probability"],
        check_names=False,
    )
    assert result.loc[("1002", "2024Q1"), "hybrid_risk_score"] > result.loc[("1001", "2024Q1"), "hybrid_risk_score"]


def test_realistic_upstream_fixture_adapts_to_secfsn05_contract():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.10, 0.18, 0.62, 0.93]},
        index=prepared_panel.index,
    )

    merged = merge_model_outputs(prepared_panel, model_outputs)
    result = run_decision_engine(merged)

    expected_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert merged.index.names == ["cik", "period"]
    assert "model_risk_probability" in merged.columns
    pd.testing.assert_series_equal(
        merged["model_risk_probability"],
        model_outputs["model_risk_probability"],
        check_names=False,
    )
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert expected_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


def test_unsorted_upstream_fixture_preserves_secfsn05_keys_and_outputs():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    unsorted_fixture = fixture.iloc[[2, 0, 3, 1]].reset_index(drop=True)
    prepared_panel = unsorted_fixture.set_index(["cik", "period"])
    model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.62, 0.10, 0.93, 0.18]},
        index=prepared_panel.index,
    )

    merged = merge_model_outputs(prepared_panel, model_outputs)
    result = run_decision_engine(merged)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    assert set(merged.index.tolist()) == set(prepared_panel.index.tolist())
    assert merged.index.tolist() == prepared_panel.index.tolist()
    assert "model_risk_probability" in merged.columns
    pd.testing.assert_series_equal(
        merged["model_risk_probability"],
        model_outputs["model_risk_probability"],
        check_names=False,
    )
    assert not result.empty
    assert set(result.index.tolist()) == set(prepared_panel.index.tolist())
    assert result.index.tolist() == prepared_panel.index.tolist()
    assert required_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


def test_merge_model_outputs_none_preserves_prepared_panel_and_allows_deterministic_only_engine():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()

    merged = merge_model_outputs(prepared_panel, None)
    result = run_decision_engine(merged)

    core_input_columns = [
        "equity_ni_residual_scaled",
        "cash_vs_earnings_gap_scaled",
        "accrual_intensity",
    ]
    core_output_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    pd.testing.assert_frame_equal(merged, prepared_panel)
    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert merged[core_input_columns].equals(prepared_panel[core_input_columns])
    assert "model_risk_probability" not in merged.columns
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert result[core_output_columns].notna().all().all()


def test_merge_model_outputs_missing_model_column_preserves_prepared_panel_and_allows_deterministic_only_engine():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    incomplete_model_outputs = pd.DataFrame(
        {"other_model_signal": [0.1, 0.2, 0.6, 0.9]},
        index=prepared_panel.index,
    )

    merged = merge_model_outputs(prepared_panel, incomplete_model_outputs)
    result = run_decision_engine(merged)

    core_input_columns = [
        "equity_ni_residual_scaled",
        "cash_vs_earnings_gap_scaled",
        "accrual_intensity",
    ]
    core_output_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    pd.testing.assert_frame_equal(merged, prepared_panel)
    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert merged[core_input_columns].equals(prepared_panel[core_input_columns])
    assert "model_risk_probability" not in merged.columns
    assert "other_model_signal" not in merged.columns
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert result[core_output_columns].notna().all().all()


def test_partial_model_coverage_preserves_full_panel_and_supports_decision_engine():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    covered_index = prepared_panel.index[[0, 3]]
    partial_model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.12, 0.91]},
        index=covered_index,
    )

    merged = merge_model_outputs(prepared_panel, partial_model_outputs)
    result = run_decision_engine(merged)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]
    uncovered_index = prepared_panel.index.difference(covered_index)

    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert "model_risk_probability" in merged.columns
    assert merged.loc[list(covered_index), "model_risk_probability"].notna().all()
    assert merged.loc[list(uncovered_index), "model_risk_probability"].isna().all()
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert required_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


def test_custom_model_col_bridges_alternate_upstream_probability_name_to_canonical_field():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    alternate_model_outputs = pd.DataFrame(
        {"upstream_probability_score": [0.11, 0.19, 0.65, 0.94]},
        index=prepared_panel.index,
    )

    merged = merge_model_outputs(
        prepared_panel,
        alternate_model_outputs,
        model_col="upstream_probability_score",
    )
    result = run_decision_engine(merged)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert "model_risk_probability" in merged.columns
    assert "upstream_probability_score" not in merged.columns
    pd.testing.assert_series_equal(
        merged["model_risk_probability"],
        alternate_model_outputs["upstream_probability_score"],
        check_names=False,
    )
    assert merged["model_risk_probability"].notna().all()
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert required_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


def test_supplied_model_outputs_overwrite_existing_canonical_model_risk_probability():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    prepared_panel["model_risk_probability"] = [0.01, 0.02, 0.03, 0.04]
    model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.21, 0.22, 0.83, 0.94]},
        index=prepared_panel.index,
    )

    merged = merge_model_outputs(prepared_panel, model_outputs)
    result = run_decision_engine(merged)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]

    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert list(merged.columns).count("model_risk_probability") == 1
    pd.testing.assert_series_equal(
        merged["model_risk_probability"],
        model_outputs["model_risk_probability"],
        check_names=False,
    )
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert required_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


def test_partial_model_outputs_overwrite_only_matching_rows_and_preserve_existing_unmatched_values():
    fixture = pd.read_csv(FIXTURE_DIR / "upstream_prepared_panel.csv", dtype={"cik": str, "period": str})
    prepared_panel = fixture.set_index(["cik", "period"]).sort_index()
    prepared_panel["model_risk_probability"] = [0.01, 0.02, 0.03, 0.04]
    covered_index = prepared_panel.index[[0, 3]]
    model_outputs = pd.DataFrame(
        {"model_risk_probability": [0.21, 0.94]},
        index=covered_index,
    )

    merged = merge_model_outputs(prepared_panel, model_outputs)
    result = run_decision_engine(merged)

    required_columns = {
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    }
    core_decision_columns = [
        "structural_risk_score",
        "fraud_proxy_score",
        "hybrid_risk_score",
        "long_allowed",
        "short_allowed",
        "risk_override",
    ]
    uncovered_index = prepared_panel.index.difference(covered_index)

    pd.testing.assert_index_equal(merged.index, prepared_panel.index)
    assert list(merged.columns).count("model_risk_probability") == 1
    pd.testing.assert_series_equal(
        merged.loc[list(covered_index), "model_risk_probability"],
        model_outputs["model_risk_probability"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        merged.loc[list(uncovered_index), "model_risk_probability"],
        prepared_panel.loc[list(uncovered_index), "model_risk_probability"],
        check_names=False,
    )
    assert not result.empty
    pd.testing.assert_index_equal(result.index, prepared_panel.index)
    assert required_columns.issubset(result.columns)
    assert result[core_decision_columns].notna().all().all()


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
