import pandas as pd

from examples.integration_smoke import (
    CANONICAL_OUTPUT_COLUMNS,
    build_sample_feature_panel,
    format_canonical_outputs,
    run_integration_smoke_example,
)


def test_integration_smoke_example_runs_and_preserves_multiindex():
    panel = build_sample_feature_panel()
    baseline = panel.copy(deep=True)

    result = run_integration_smoke_example()
    displayed = format_canonical_outputs(result)

    pd.testing.assert_frame_equal(panel, baseline)
    pd.testing.assert_index_equal(result.index, panel.index)
    pd.testing.assert_index_equal(displayed.index, panel.index)
    assert result.index.names == ["cik", "period"]
    assert all(isinstance(cik, str) for cik in result.index.get_level_values("cik"))
    assert all(
        isinstance(period, str) and len(period) == 6 and period[4] == "Q"
        for period in result.index.get_level_values("period")
    )
    assert set(CANONICAL_OUTPUT_COLUMNS).issubset(result.columns)
    assert list(displayed.columns) == list(CANONICAL_OUTPUT_COLUMNS)
