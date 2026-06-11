# Canonical Run Paths And Stage Status

## Current Status

- Stage 01 (`secfsn` / raw SEC parsing and base extraction): not implemented in this repo; assumed complete upstream per project-owner direction.
- Stage 02 (`secfsn_plus` / feature engineering): not implemented in this repo; assumed complete upstream per project-owner direction.
- Stage 03 (`secfsn_plus_structural` plus optional `secfsn_plus_models` Layer 3A outputs): not implemented in this repo; this repo only consumes their prepared outputs.
- Stage 04 (`secfsn_plus_decisionsystems` / Layer 3B deterministic decision logic): this repo's scope.

## Stage Relationship

This repository does not regenerate Stages 01-03 outputs. The canonical Stage 04 input contract is:

1. A prepared `pandas.DataFrame` indexed by `("cik", "period")`.
2. Upstream structural/accounting features already materialized on that panel.
3. Optional Layer 3A model output column `model_risk_probability`, merged by `secfsn_plus_decisionsystems.integration.model_bridge.merge_model_outputs()`.

Stage 04 then runs `secfsn_plus_decisionsystems.integration.decision_engine.run_decision_engine()` to produce deterministic decision outputs.

## Canonical Commands Used For Validation

The following commands were run from `C:\GitHub\secfsn_plus_decisionsystems`:

```powershell
pytest
python examples\integration_smoke.py
python -m compileall secfsn_plus_decisionsystems examples tests
```

## Validation Outcome

- `pytest`: passed (`32 passed`).
- `python examples\integration_smoke.py`: passed and printed canonical decision columns for a synthetic prepared panel.
- `python -m compileall secfsn_plus_decisionsystems examples tests`: passed.

These validations confirm the repo's own canonical Stage 04 path without re-running Stage 01, Stage 02, or Stage 03 pipelines inside this repository.

## What Stage 04 Consumes

Canonical required upstream inputs:

- MultiIndex levels: `cik`, `period`
- Prepared feature columns used by deterministic scorers, such as:
  - `equity_ni_residual_scaled`
  - `cash_vs_earnings_gap_scaled`
  - `accrual_intensity`
- Optional canonical model column from Stage 03 / Layer 3A:
  - `model_risk_probability`

A tiny synthetic upstream-shaped fixture now lives at
`tests/fixtures/upstream_prepared_panel.csv`. It is a hand-reviewable test
fixture only, not canonical Stage 01/02/03 output.

Stage 04 should not rely on upstream row ordering. Current contract tests
cover both sorted and unsorted upstream-prepared panels as long as the input
still conforms to the `("cik", "period")` MultiIndex contract.

Validated optional-model behavior is narrow and deterministic:

- `run_decision_engine(...)` supports deterministic-only input; `model_risk_probability` is optional.
- `merge_model_outputs(panel, None)` returns a copy of the prepared panel unchanged.
- `merge_model_outputs(panel, model_outputs_without_model_risk_probability)` also returns a copy unchanged.
- `merge_model_outputs(panel, partial_model_outputs)` preserves the full prepared panel index. If the prepared panel does not already contain `model_risk_probability`, covered rows are populated and uncovered rows remain null.
- `merge_model_outputs(panel, model_outputs, model_col="upstream_probability_score")` reads the alternate upstream probability field and exposes it under the canonical Stage 04 column `model_risk_probability`.
- `merge_model_outputs(panel_with_model_risk_probability, model_outputs_with_model_risk_probability)` keeps exactly one canonical `model_risk_probability` column and overwrites matching prepared-panel values with the separately supplied `model_outputs` values.
- `merge_model_outputs(panel_with_model_risk_probability, partial_model_outputs_with_model_risk_probability)` overwrites only matching keys; unmatched rows retain their existing prepared-panel `model_risk_probability` values.

Partial Stage 03 / Layer 3A model coverage is allowed as long as deterministic
prepared-panel inputs are present for every `("cik", "period")` row.

Canonical Stage 04 outputs:

- `structural_risk_score`
- `fraud_proxy_score`
- `hybrid_risk_score`
- `high_structural_risk_flag`
- `high_fraud_proxy_flag`
- `high_hybrid_risk_flag`
- `long_allowed`
- `short_allowed`
- `risk_override`

## First Narrow Stage 04 Step

The first low-risk Stage 04 implementation step in this repo is to harden the canonical upstream handoff path:

- verify `prepared panel -> merge_model_outputs(...) -> run_decision_engine(...)`
- keep Stage 04 scoped to deterministic integration rather than adding speculative new features

This is implemented by the integration test covering the canonical Stage 04 handoff path.

## Assumptions And Blockers

- Assumption: Stages 01-03 are owned and validated in their upstream repositories, not here.
- Assumption: `model_risk_probability` remains the canonical optional Layer 3A output column.
- Duplicate upstream `("cik", "period")` keys are not yet part of the validated
  Stage 04 contract and should be treated as out of contract unless explicitly
  validated in a future change.
- Blocker: no real upstream artifact snapshot is stored in this repo, so validation here is limited to synthetic smoke/tests unless an external prepared panel is provided.
