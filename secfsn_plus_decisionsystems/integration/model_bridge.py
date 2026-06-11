"""Optional integration helpers for Layer 3A model outputs."""

from __future__ import annotations

import pandas as pd

CANONICAL_MODEL_RISK_COL = "model_risk_probability"


def merge_model_outputs(
    panel: pd.DataFrame,
    model_outputs: pd.DataFrame | None,
    model_col: str = CANONICAL_MODEL_RISK_COL,
) -> pd.DataFrame:
    """Merge optional model outputs into the decision panel.

    Args:
        panel: Deterministic feature panel indexed by ``(cik, period)``.
        model_outputs: Optional DataFrame with matching MultiIndex and a model
            probability column. If ``None`` or missing column, panel is returned
            unchanged (copied).
        model_col: Name of the upstream model-probability column to read from
            ``model_outputs``. The merged output is exposed under the canonical
            Stage 04 field ``model_risk_probability``.

    Returns:
        New DataFrame containing the merged canonical model-risk column when
        available.
    """

    result = panel.copy()
    if model_outputs is None or model_col not in model_outputs.columns:
        return result

    incoming = model_outputs.reindex(result.index)[model_col]
    if CANONICAL_MODEL_RISK_COL in result.columns:
        result[CANONICAL_MODEL_RISK_COL] = incoming.combine_first(result[CANONICAL_MODEL_RISK_COL])
    else:
        result[CANONICAL_MODEL_RISK_COL] = incoming
    return result
