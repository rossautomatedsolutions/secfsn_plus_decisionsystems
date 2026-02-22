"""Optional integration helpers for Layer 3A model outputs."""

from __future__ import annotations

import pandas as pd


def merge_model_outputs(
    panel: pd.DataFrame,
    model_outputs: pd.DataFrame | None,
    model_col: str = "model_risk_probability",
) -> pd.DataFrame:
    """Merge optional model outputs into the decision panel.

    Args:
        panel: Deterministic feature panel indexed by ``(cik, period)``.
        model_outputs: Optional DataFrame with matching MultiIndex and a model
            probability column. If ``None`` or missing column, panel is returned
            unchanged (copied).
        model_col: Name of model-probability column expected from Layer 3A.

    Returns:
        New DataFrame containing the merged model column when available.
    """

    result = panel.copy()
    if model_outputs is None or model_col not in model_outputs.columns:
        return result

    result[model_col] = model_outputs.reindex(result.index)[model_col]
    return result
