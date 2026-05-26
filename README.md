# secfsn_plus_decisionsystems

`secfsn_plus_decisionsystems` is **Layer 3B** in a modular quantitative research stack:

1. Layer 1: `secfsn`
2. Layer 2: `secfsn_plus`
3. Layer 2.5: `secfsn_plus_structural`
4. Layer 3A: `secfsn_plus_models`
5. Layer 3B: `secfsn_plus_decisionsystems` (this repository)

## Architectural role

This repository provides deterministic, rules-based decision functionality:

- deterministic scoring from already-prepared features
- anomaly/outlier and instability detection
- threshold/persistence risk flags
- screening outputs for short, put, and risk-filtered long candidates
- optional consumption of model outputs produced upstream by Layer 3A

## Upstream/downstream contract

### Inputs expected

All public functions expect a `pandas.DataFrame` indexed by a MultiIndex:

- level 0: `cik`
- level 1: `period`
- `period` format: `YYYYQ#`
- `cik` should be treated as a string identifier even when stored numerically

The panel should already be prepared by upstream layers. This repo:

- **does not** parse SEC filings
- **does not** perform raw wrangling
- **does not** derive canonical accounting ratios

### Outputs produced

Functions return new DataFrames with added deterministic columns or filtered rows,
without mutating the input DataFrame.

The decision engine preserves the input MultiIndex and adds deterministic
decision columns such as:

- `structural_risk_score`
- `fraud_proxy_score`
- `hybrid_risk_score`
- `high_structural_risk_flag`
- `high_fraud_proxy_flag`
- `high_hybrid_risk_flag`
- `long_allowed`
- `short_allowed`
- `risk_override`

The options-oriented `screen_put_candidates()` helper may also consume
`rolling_instability` as an extra diagnostic. That column is supported as an
options-specific enhancement, not part of the canonical decision-output list
for general downstream integrations.

Downstream consumers can include portfolio construction, reporting, and execution
layers that require transparent, reproducible decision signals.

## Deterministic vs modeling separation

This repository intentionally excludes model training logic.

- Allowed: consume existing model outputs (for example, a column like
  `model_risk_probability`) and combine them deterministically.
- Not allowed: fit models, tune hyperparameters, derive upstream features, or
  run non-deterministic learning pipelines.

Legacy aliases such as `hybrid_decision_score` remain available only for
backward compatibility. Canonical downstream integrations should consume
`hybrid_risk_score` and the decision-engine flags directly.

## Package structure

- `scoring/`: structural, fraud-proxy, and hybrid deterministic scores
- `anomaly/`: cross-sectional outlier and per-CIK instability diagnostics
- `risk_flags/`: threshold and persistence flags
- `screening/`: short/put/long candidate screens
- `integration/`: model bridge and orchestration entrypoint
- `diagnostics/`: distribution and stability checks

## Example usage

```python
import pandas as pd

from secfsn_plus_decisionsystems.integration.decision_engine import run_decision_engine
from secfsn_plus_decisionsystems.integration.model_bridge import merge_model_outputs

# MultiIndex (cik, period) panel prepared upstream
index = pd.MultiIndex.from_tuples(
    [("1001", "2023Q4"), ("1001", "2024Q1"), ("1002", "2024Q1")],
    names=["cik", "period"],
)
panel = pd.DataFrame(
    {
        "equity_ni_residual_scaled": [0.1, 0.2, 1.2],
        "cash_vs_earnings_gap_scaled": [0.0, 0.1, 1.0],
        "accrual_intensity": [0.1, 0.3, 0.8],
        "asset_revenue_growth_mismatch_scaled": [0.0, 0.2, 0.7],
        "non_cash_earnings_ratio": [0.1, 0.2, 0.6],
    },
    index=index,
)

model_outputs = pd.DataFrame(
    {"model_risk_probability": [0.15, 0.55, 0.95]},
    index=index,
)

merged = merge_model_outputs(panel, model_outputs)
scored = run_decision_engine(merged)

print(
    scored[
        [
            "structural_risk_score",
            "fraud_proxy_score",
            "hybrid_risk_score",
            "long_allowed",
            "short_allowed",
            "risk_override",
        ]
    ]
)
```

A repo-level smoke example is also available at
[`examples/integration_smoke.py`](C:/GitHub_Review_Folder/secfsn_plus_decisionsystems/examples/integration_smoke.py)
to demonstrate a small realistic prepared panel flowing through the decision
engine without raw SEC loading or model training.

## Current validation status

- synthetic integration smoke test passes
- production/main-machine clone validated
- real-data integration remains a future milestone

## Extension guidance

Current modules are stubs with clear extension points and full docstrings.
When extending:

1. Keep operations vectorized.
2. Handle missing columns gracefully.
3. Return new DataFrames (no in-place mutation).
4. Preserve deterministic behavior.

## Limitations

This package is intended for research workflows, not investment advice or live
trade execution. Scores and screens are deterministic diagnostics built from
upstream prepared features and should be validated within the broader stack.
