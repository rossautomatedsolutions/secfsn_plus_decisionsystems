# SECFSN 05 / Layer 3B demo narration script

This 2–3 minute recording demonstrates `secfsn_plus_decisionsystems` as implemented: SECFSN 05, Layer 3B of the research stack. It converts prepared upstream evidence into deterministic, auditable research-decision states. It does not train a predictive model.

| Time | Notebook section and on-screen view | Narration / transition |
| --- | --- | --- |
| 0:00–0:18 | Title and three-bullet overview | “This is SECFSN 05, Layer 3B: the decision-system layer. Its job is to turn upstream SEC-derived evidence into transparent decision states. It does not parse filings or train models.” |
| 0:18–0:35 | First execution confirmation | “Everything here is deterministic: no network calls, credentials, live APIs, or model training. The demonstration uses the small committed fixture that supports the repository’s tests.” |
| 0:35–1:03 | **1. Upstream evidence** and the compact evidence table | “The input is a panel keyed by CIK and reporting period. These are prepared accounting signals: equity-to-net-income residual, cash-versus-earnings gap, and accrual intensity. Their derivation belongs to upstream SEC and feature-engineering layers.” |
| 1:03–1:18 | Layer 3A model-evidence column | “A Layer 3A probability can be merged through this repository’s public model bridge. The committed fixture has none, so the absence is displayed rather than filled with a made-up value. The deterministic decision path still runs.” |
| 1:18–1:43 | **2. Deterministic rules and synthesis** plus completion message | “One canonical API builds structural and fraud-proxy scores, computes anomaly, instability, and persistence diagnostics, and then produces decision flags. The default hybrid score equally blends the two deterministic scores. A risk override is the documented veto or exception state.” |
| 1:43–2:17 | **3. Resulting decision state** final table | “This table is the product output. It keeps the component evidence, the optional-model evidence state, the veto state, and the resulting research state together. In this fixture, both 1001 rows pass the long-side screen, while 2002 in 2024 Q2 reaches the short-side screen. The other 2002 row remains monitored.” |
| 2:17–2:35 | Keep final table visible | “The result is a traceable bridge from fundamental evidence to downstream research, reporting, portfolio, or execution systems. It identifies where to investigate next; it is not a prediction, fraud conclusion, or investment recommendation.” |

## Optional 15-second closing

“SECFSN 05 consumes upstream evidence, applies deterministic and auditable decision rules, and emits downstream-ready research states. Any SEC parsing, feature engineering, real-data validation, and model training remain upstream responsibilities.”

## Recording cues

- Begin with the title cell visible.
- Execute the four code cells in order; all outputs fit without scrolling.
- Hold the final decision table on screen for the closing statement.
