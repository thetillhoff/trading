"""
Walk-forward evaluation module.

Provides day-by-day evaluation that never has access to future data.
All evaluations are done on a "day-by-day" basis.
"""
from .walk_forward import WalkForwardEvaluator, WalkForwardResult
from .portfolio import PortfolioSimulator, SimulationResult, Position, PositionStatus
from .trade_analysis import (
    aggregate_positions_by_signal_type,
    aggregate_trades_dataframe_by_signal_type,
    analyze_pretrade_predictors,
)

__all__ = [
    'WalkForwardEvaluator',
    'WalkForwardResult',
    'PortfolioSimulator',
    'SimulationResult',
    'Position',
    'PositionStatus',
    'aggregate_positions_by_signal_type',
    'aggregate_trades_dataframe_by_signal_type',
    'analyze_pretrade_predictors',
]
