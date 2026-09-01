# secfsn_plus_decisionsystems — SECFSN 05 / Layer 3B

`secfsn_plus_decisionsystems` is **SECFSN 05 / Layer 3B** in a modular quantitative research stack:

1. SECFSN 01 / Layer 1: `secfsn` — normalized SEC financial-statement facts
2. SECFSN 02 / Layer 2: `secfsn_plus` — derived financial features and analytical enrichment
3. SECFSN 03 / Layer 2.5: `secfsn_plus_structural` — structural accounting and filing-pattern evidence
4. SECFSN 04 / Layer 3A: `secfsn_plus_models` — model-derived evidence and validated model-output contracts
5. SECFSN 05 / Layer 3B: `secfsn_plus_decisionsystems` (this repository) — governed decision outputs

## Architectural role

This repository provides deterministic, rules-based decision functionality:

- deterministic scoring from already-prepared features
- anomaly/outlier and instability detection
- threshold/persistence risk flags
- screening outputs for short, put, and risk-filtered long candidates
- optional consumption of model-derived evidence produced upstream by SECFSN 04 / Layer 3A

## Upstream/downstream contract

### Inputs expected

All public functions expect a `pandas.DataFrame` indexed by a MultiIndex:

- level 0: `cik`
- level 1: `period`
- `period` format: `YYYYQ#`
- `cik` should be treated as a string identifier even when stored numerically

The panel should already reflect the available SECFSN 01–04 upstream analytical stack:

- SECFSN 01 normalized SEC financial-statement data
- SECFSN 02 derived financial features and analytical enrichment
- SECFSN 03 structural accounting and filing-pattern evidence
- optional SECFSN 04 model-derived evidence under a validated output contract

SECFSN 05 consumes that available evidence. This repo:

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

- Allowed: consume SECFSN 04 model-derived evidence (for example, a column like
  `model_risk_probability`) under the documented deterministic contract.
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

A legacy-named decision-systems walkthrough for short screen recordings is available at
[`examples/stage04_demo_walkthrough.py`](C:/GitHub/secfsn_plus_decisionsystems/examples/stage04_demo_walkthrough.py).
Its filename predates the SECFSN 05 naming convention; it is not an SECFSN 04 demo.
Run it with:

```powershell
python examples\stage04_demo_walkthrough.py
```

## Running the SECFSN 05 demo

From the repository root, create an environment with Python 3.10 or newer and
install the package plus Jupyter's notebook launcher:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e . notebook
```

Launch the recording-ready notebook with:

```powershell
jupyter notebook notebooks\05_secfsn_plus_decisionsystems_demo.ipynb
```

The demo uses only the committed synthetic SECFSN 05 fixture at
`tests/fixtures/upstream_prepared_panel.csv`; it does not require network access,
credentials, real SEC data, or an actual SECFSN 04 trained-model artifact. SECFSN 04
is a supported upstream model-evidence source, but its missing fixture artifact is
displayed explicitly and never fabricated or silently weighted. For a clean execution
check that saves displayed outputs in place, run:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\05_secfsn_plus_decisionsystems_demo.ipynb
```

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
