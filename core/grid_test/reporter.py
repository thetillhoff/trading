"""
Comparison reporter for backtesting results.

Generates comparison reports and visualizations for walk-forward evaluation results.

This module re-exports from split submodules for backward compatibility.
Import from here: ComparisonReporter, trades_to_dataframe, compute_alpha_over_time_series,
_daily_rate_from_pa, _is_new_month, CASH_DAILY_RATE_2PA, AlphaOverTimeSeries, MAX_LEGEND_INSTRUMENTS.
"""
from .reporter_utils import (
    CASH_DAILY_RATE_2PA,
    MAX_LEGEND_INSTRUMENTS,
    _daily_rate_from_pa,
    _is_new_month,
    trades_to_dataframe,
)
from .reporter_analysis import AlphaOverTimeSeries, compute_alpha_over_time_series
from .reporter_base import ComparisonReporter

__all__ = [
    "ComparisonReporter",
    "trades_to_dataframe",
    "compute_alpha_over_time_series",
    "_daily_rate_from_pa",
    "_is_new_month",
    "CASH_DAILY_RATE_2PA",
    "AlphaOverTimeSeries",
    "MAX_LEGEND_INSTRUMENTS",
]
