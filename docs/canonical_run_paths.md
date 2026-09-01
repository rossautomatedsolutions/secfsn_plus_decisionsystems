# Canonical SECFSN 05 run paths and status

## Current status

The canonical analytical stack is:

1. SECFSN 01 / Layer 1 (`secfsn`): normalized SEC financial-statement data.
2. SECFSN 02 / Layer 2 (`secfsn_plus`): derived financial features and analytical enrichment.
3. SECFSN 03 / Layer 2.5 (`secfsn_plus_structural`): structural accounting and filing-pattern evidence.
4. SECFSN 04 / Layer 3A (`secfsn_plus_models`): optional model-derived evidence and validated model-output contracts.
5. SECFSN 05 / Layer 3B (`secfsn_plus_decisionsystems`): deterministic, governed decision outputs. This repository owns this layer.

SECFSN 01–04 are not implemented here. This repository consumes their available
prepared outputs and does not rerun their pipelines.

## Canonical SECFSN 05 workflow

The supported handoff is:

1. A prepared `pandas.DataFrame` indexed by `("cik", "period")` arrives from the SECFSN 01–03 evidence layers.
2. Optional SECFSN 04 model-derived evidence is aligned through `secfsn_plus_decisionsystems.integration.model_bridge.merge_model_outputs()` under the canonical `model_risk_probability` column.
3. `secfsn_plus_decisionsystems.integration.decision_engine.run_decision_engine()` applies SECFSN 05 deterministic scoring, diagnostics, veto/override handling, and screening-state outputs.

The package preserves the prepared panel index and does not mutate the caller's
input DataFrame.

## What SECFSN 05 consumes

Required upstream contract:

- MultiIndex levels: `cik`, `period`
- `period` format: `YYYYQ#`
- prepared feature columns such as `equity_ni_residual_scaled`, `cash_vs_earnings_gap_scaled`, and `accrual_intensity`

Optional SECFSN 04 / Layer 3A contract:

- `model_risk_probability`, merged by the public bridge when an upstream artifact is available

The committed `tests/fixtures/upstream_prepared_panel.csv` is a small,
hand-reviewable synthetic fixture. It is not a real SECFSN 01–04 artifact and
does not contain a trained SECFSN 04 model output.

Validated optional-model behavior is deterministic:

- `run_decision_engine(...)` supports evidence with no model probability.
- `merge_model_outputs(panel, None)` returns a copy of the panel unchanged.
- Missing or partial SECFSN 04 coverage preserves the full SECFSN 05 panel index.
- An alternate upstream probability field can be mapped to canonical `model_risk_probability` with `model_col`.
- Existing canonical model values are overwritten only where supplied upstream values align.

## SECFSN 05 outputs

Canonical decision-engine outputs include:

- `structural_risk_score`
- `fraud_proxy_score`
- `hybrid_risk_score`
- `high_structural_risk_flag`
- `high_fraud_proxy_flag`
- `high_hybrid_risk_flag`
- `long_allowed`
- `short_allowed`
- `risk_override`

SECFSN 05 owns the deterministic synthesis, anomaly and persistence diagnostics,
veto/override state, and public screening APIs. SECFSN 04 remains an optional
upstream evidence source; this repository does not train models.

## Canonical validation commands

Run from the repository root:

```powershell
pytest
python examples\integration_smoke.py
python examples\secfsn05_decision_system_walkthrough.py
python -m compileall secfsn_plus_decisionsystems examples tests
```

Run the canonical recording notebook with:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks\05_secfsn_plus_decisionsystems_demo.ipynb
```

## Assumptions and limitations

- SECFSN 01–04 are owned and validated in their upstream repositories.
- `model_risk_probability` remains the canonical optional SECFSN 04 evidence field.
- Duplicate upstream `("cik", "period")` keys are outside the validated contract.
- This repository has no real upstream artifact snapshot, so validation here is limited to synthetic fixtures unless an external prepared panel is supplied.
