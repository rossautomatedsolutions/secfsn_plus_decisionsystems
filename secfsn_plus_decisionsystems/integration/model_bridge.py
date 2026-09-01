"""SECFSN 05 bridge for optional upstream SECFSN 04 / Layer 3A model outputs."""

from __future__ import annotations

import pandas as pd

CANONICAL_MODEL_RISK_COL = "model_risk_probability"


def merge_model_outputs(
    panel: pd.DataFrame,
    model_outputs: pd.DataFrame | None,
    model_col: str = CANONICAL_MODEL_RISK_COL,
) -> pd.DataFrame:
    """Merge optional upstream SECFSN 04 model outputs into an SECFSN 05 panel.

    Args:
        panel: SECFSN 05 decision-input panel indexed by ``(cik, period)``.
        model_outputs: Optional SECFSN 04 DataFrame with matching MultiIndex and
            a model-probability column. If ``None`` or missing column, the panel
            is returned unchanged (copied).
        model_col: Name of the SECFSN 04 model-probability column to read from
            ``model_outputs``. SECFSN 05 exposes it under the canonical
            ``model_risk_probability`` field.

    Returns:
        New SECFSN 05 decision-input DataFrame containing the merged canonical
        model-risk column when available. Scoring, screening, veto/override, and
        decision-state behavior are owned by the SECFSN 05 decision engine.
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
