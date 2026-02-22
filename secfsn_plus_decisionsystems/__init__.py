"""Top-level package for deterministic decision logic on SECFSN+ feature panels.

This package is intentionally limited to **Layer 3B** responsibilities:

- deterministic scoring from prepared feature panels
- anomaly and instability diagnostics
- rules-based risk flags and screening outputs
- optional consumption of model outputs from Layer 3A

It explicitly excludes SEC parsing, ratio engineering, and model training.
"""

from .config import DecisionSystemConfig

__all__ = ["DecisionSystemConfig"]
