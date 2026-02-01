"""
Walk-forward evaluation types: trade outcomes, evaluation summary, result.

Extracted for reuse and to keep walk_forward.py focused on evaluation logic.
New evaluators or strategies can import these types without pulling in WalkForwardEvaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from ..signals.config import StrategyConfig
from .portfolio_types import SimulationResult


class TradeOutcome(Enum):
    """Possible outcomes for a trade."""
    TARGET_HIT = "target_hit"  # Target price reached before stop-loss
    STOP_LOSS_HIT = "stop_loss_hit"  # Stop-loss hit before target
    NO_OUTCOME = "no_outcome"  # Neither target nor stop-loss hit (still open or data ended)
    INVALID = "invalid"  # Invalid signal (missing target or stop-loss)


@dataclass
class TradeEvaluation:
    """Evaluation result for a single trade."""
    signal: object  # TradingSignal
    outcome: TradeOutcome
    exit_price: Optional[float]  # Price at which trade exited (target or stop-loss)
    exit_timestamp: Optional[pd.Timestamp]  # When the trade exited
    gain_percentage: float  # Percentage gain/loss (-100% if stop-loss hit)
    days_held: Optional[int]  # Number of days the trade was held
    max_favorable_excursion: float  # Maximum favorable price movement (%)
    max_adverse_excursion: float  # Maximum adverse price movement (%)


@dataclass
class EvaluationSummary:
    """Summary statistics for all evaluated trades."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    no_outcome_trades: int
    win_rate: float  # Percentage of trades that hit target
    average_gain: float  # Average percentage gain (only winning trades)
    average_loss: float  # Average percentage loss (only losing trades)
    total_gain: float  # Sum of all percentage gains/losses
    best_trade: Optional[TradeEvaluation] = None
    worst_trade: Optional[TradeEvaluation] = None
    average_days_held: Optional[float] = None


@dataclass
class WalkForwardResult:
    """Results from a walk-forward evaluation."""
    config: StrategyConfig

    # Portfolio simulation results
    simulation: SimulationResult

    # Walk-forward specific metrics
    evaluation_start_date: pd.Timestamp
    evaluation_end_date: pd.Timestamp
    lookback_days: int
    step_days: int

    # Performance metrics (derived from simulation)
    buy_and_hold_gain: float = 0.0  # Full 100% invested market return
    exposure_adjusted_market: float = 0.0  # Market return scaled to strategy's avg exposure
    outperformance: float = 0.0  # vs exposure-adjusted market (fair comparison)

    # Hybrid strategy metrics (active + passive in buy-and-hold)
    hybrid_return: float = 0.0  # Combined: active portion + passive portion earning market return
    active_alpha: float = 0.0   # Hybrid return - pure buy-and-hold (the value added by active trading)

    # Performance monitoring (optional): phase and per-indicator timings in seconds
    performance_timings: Optional[Dict[str, float]] = None

    @property
    def summary(self) -> EvaluationSummary:
        """Create a summary compatible with code that expects EvaluationSummary."""
        return EvaluationSummary(
            total_trades=self.simulation.total_trades,
            winning_trades=self.simulation.winning_trades,
            losing_trades=self.simulation.losing_trades,
            no_outcome_trades=0,
            win_rate=self.simulation.win_rate,
            average_gain=self.simulation.avg_win_pct,
            average_loss=self.simulation.avg_loss_pct,
            total_gain=self.simulation.total_return_pct,
            best_trade=None,
            worst_trade=None,
            average_days_held=self.simulation.avg_days_held,
        )
