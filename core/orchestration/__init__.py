"""
Orchestration layer for multi-step workflows (hypothesis suites, etc.).

Currently focused on hypothesis grid-search runs across multiple periods.
"""

from .hypothesis import HypothesisRunConfig, run_hypothesis_suite

__all__ = ["HypothesisRunConfig", "run_hypothesis_suite"]

