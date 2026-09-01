# SECFSN 05 / Layer 3B demo narration script

This 2–3 minute recording demonstrates `secfsn_plus_decisionsystems` as implemented: **SECFSN 05 / Layer 3B**. SECFSN 01–04 are the upstream analytical stack; SECFSN 05 converts their available evidence into deterministic, auditable research-decision states. It does not train a predictive model.

| Time | Notebook section and on-screen view | Narration / transition |
| --- | --- | --- |
| 0:00–0:18 | Title and three-bullet overview | “This is SECFSN 05, Layer 3B: the decision-system layer. Its job is to turn available upstream SEC-derived evidence into transparent decision states. It does not parse filings or train models.” |
| 0:18–0:36 | Five-row architecture table | “The stack is deliberate: SECFSN 01 normalizes SEC facts; 02 derives financial features; 03 adds structural accounting and filing-pattern evidence; SECFSN 04—upstream `secfsn_plus_models`—adds optional model-derived evidence and validated output contracts; and SECFSN 05 owns the governed decision output. The highlighted final row is this repository.” |
| 0:36–0:50 | Setup confirmation | “Everything here is deterministic: no network calls, credentials, live APIs, or model training. The demonstration uses the small committed SECFSN 05 fixture that supports the repository’s tests.” |
| 0:50–1:15 | **1. Upstream evidence** and the compact evidence table | “The input is a panel keyed by CIK and reporting period. The visible accounting signals originate across SECFSN 01 through 03. SECFSN 04 is a supported upstream model-evidence source, but this fixture contains no trained-model artifact, so its evidence state is explicitly missing.” |
| 1:15–1:42 | **2. Deterministic rules and synthesis** plus completion message | “SECFSN 05 owns one canonical API that builds structural and fraud-proxy scores, computes anomaly, instability, and persistence diagnostics, and produces decision flags. The default hybrid score equally blends its two deterministic components. A risk override is this repository’s veto or exception state.” |
| 1:42–2:17 | **3. Resulting decision state** final table | “This table is the SECFSN 05 product output. It separates the available upstream evidence, the missing SECFSN 04 model-evidence state, the Layer 3B veto state, and the resulting research state. In this fixture, both 1001 rows pass the long-side screen, while 2002 in 2024 Q2 reaches the short-side screen.” |
| 2:17–2:35 | Keep final table visible | “The result is a traceable bridge from upstream analytical evidence to downstream research, reporting, portfolio, or execution systems. It identifies where to investigate next; it is not a prediction, fraud conclusion, or investment recommendation.” |

## Optional 15-second closing

“SECFSN 01–04 provide the available analytical evidence; SECFSN 05 applies deterministic and auditable decision rules and emits downstream-ready research states. SECFSN 04 model evidence is optional and is never fabricated when its artifact is absent.”

## Recording cues

- Begin with the title cell visible.
- Execute the four code cells in order; all outputs fit without scrolling.
- Hold the final decision table on screen for the closing statement.
