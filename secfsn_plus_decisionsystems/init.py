"""Compatibility module mirroring package exports.

Some tooling requests a literal ``init.py`` in addition to ``__init__.py``.
This file re-exports the same public symbols.
"""

from .config import DecisionSystemConfig

__all__ = ["DecisionSystemConfig"]
