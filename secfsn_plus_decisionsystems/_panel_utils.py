"""Internal helpers for validating and ordering prepared SECFSN+ panels."""

from __future__ import annotations

import re

import pandas as pd


_PERIOD_PATTERN = re.compile(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")


def validate_panel_index(panel: pd.DataFrame) -> None:
    """Ensure the panel follows the public ``(cik, period)`` MultiIndex contract."""

    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels < 2:
        raise ValueError("panel must be indexed by a (cik, period) MultiIndex")


def sort_panel_by_cik_period(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a copy sorted by ``cik`` and parsed ``YYYYQ#`` period order."""

    validate_panel_index(panel)

    index_frame = panel.index.to_frame(index=False)
    periods = index_frame.iloc[:, 1].astype(str)
    extracted = periods.str.extract(_PERIOD_PATTERN)
    if extracted.isna().any().any():
        raise ValueError("period index level must use YYYYQ# format")

    ordered = panel.copy()
    ordered["_period_year"] = extracted["year"].astype(int).to_numpy()
    ordered["_period_quarter"] = extracted["quarter"].astype(int).to_numpy()
    ordered = ordered.sort_values(
        by=[ordered.index.names[0], "_period_year", "_period_quarter"],
        kind="mergesort",
    )
    return ordered.drop(columns=["_period_year", "_period_quarter"])
