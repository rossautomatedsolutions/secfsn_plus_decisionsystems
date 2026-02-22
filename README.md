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

The panel should already be prepared by upstream layers. This repo:

- **does not** parse SEC filings
- **does not** perform raw wrangling
- **does not** derive canonical accounting ratios

### Outputs produced

Functions return new DataFrames with added deterministic columns or filtered rows,
without mutating the input DataFrame.

Downstream consumers can include portfolio construction, reporting, and execution
layers that require transparent, reproducible decision signals.

## Deterministic vs modeling separation

This repository intentionally excludes model training/inference logic.

- Allowed: consume existing model outputs (for example, a column like
  `model_risk_probability`) and combine them deterministically.
- Not allowed: fit models, tune hyperparameters, or run non-deterministic
  learning pipelines.

## Package structure

- `scoring/`: structural, fraud-proxy, and hybrid score stubs
- `anomaly/`: outlier and rolling instability stubs
- `risk_flags/`: threshold and persistence flag stubs
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
    [(1001, "2023Q4"), (1001, "2024Q1"), (1002, "2024Q1")],
    names=["cik", "period"],
)
panel = pd.DataFrame(
    {
        "structural_risk_score": [0.2, 0.4, 0.8],
        "fraud_proxy_score": [0.1, 0.5, 0.9],
    },
    index=index,
)

model_outputs = pd.DataFrame(
    {"model_risk_probability": [0.15, 0.55, 0.95]},
    index=index,
)

merged = merge_model_outputs(panel, model_outputs)
scored = run_decision_engine(merged)

print(scored[["hybrid_decision_score", "rolling_instability", "is_high_risk_threshold"]])
```

## Extension guidance

Current modules are stubs with clear extension points and full docstrings.
When extending:

1. Keep operations vectorized.
2. Handle missing columns gracefully.
3. Return new DataFrames (no in-place mutation).
4. Preserve deterministic behavior.
