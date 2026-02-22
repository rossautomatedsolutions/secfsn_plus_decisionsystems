"""Top-level deterministic decision engine orchestration."""

from __future__ import annotations

import pandas as pd

from secfsn_plus_decisionsystems.anomaly.rolling_instability import compute_rolling_instability
from secfsn_plus_decisionsystems.risk_flags.threshold_flags import apply_threshold_flag
from secfsn_plus_decisionsystems.scoring.hybrid_score import compute_hybrid_score


def run_decision_engine(panel: pd.DataFrame) -> pd.DataFrame:
    """Run a minimal deterministic decision pipeline.

    This is a stub orchestration entry-point. It demonstrates composition of
    deterministic components without introducing any model training logic.

    Extension notes:
        - Plug additional scores and flag modules in a versioned sequence.
        - Externalize thresholds/config via :mod:`secfsn_plus_decisionsystems.config`.
    """

    result = panel.copy()
    result = compute_hybrid_score(result)
    result = compute_rolling_instability(result, value_col="hybrid_decision_score")
    result = apply_threshold_flag(result, score_col="hybrid_decision_score", threshold=0.7)
    return result
