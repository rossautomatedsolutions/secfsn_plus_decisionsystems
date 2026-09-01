"""SECFSN 05 / Layer 3B deterministic decision logic.

``secfsn_plus_decisionsystems`` consumes available evidence from the SECFSN
01–04 upstream stack and owns the governed decision-system responsibilities:

- deterministic scoring from prepared feature panels
- anomaly and instability diagnostics
- rules-based risk flags and screening outputs
- optional consumption of SECFSN 04 / Layer 3A model-derived evidence

It explicitly excludes SEC parsing, ratio engineering, and model training.
"""

from .config import DecisionSystemConfig

__all__ = ["DecisionSystemConfig"]
