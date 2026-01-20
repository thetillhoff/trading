"""
Evaluates trading signals by checking actual market outcomes.

For each signal, determines if the target price or stop-loss was hit first,
and calculates the percentage gain or loss.
"""
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
trading_signals_dir = Path('/app/trading_signals')
if not trading_signals_dir.exists():
    trading_signals_dir = current_dir.parent / 'trading_signals'
sys.path.insert(0, str(trading_signals_dir))
from signal_detector import TradingSignal, SignalType


class TradeOutcome(Enum):
    """Possible outcomes for a trade."""
    TARGET_HIT = "target_hit"  # Target price reached before stop-loss
    STOP_LOSS_HIT = "stop_loss_hit"  # Stop-loss hit before target
    NO_OUTCOME = "no_outcome"  # Neither target nor stop-loss hit (still open or data ended)
    INVALID = "invalid"  # Invalid signal (missing target or stop-loss)


@dataclass
class TradeEvaluation:
    """Evaluation result for a single trade."""
    signal: TradingSignal
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
    best_trade: Optional[TradeEvaluation]
    worst_trade: Optional[TradeEvaluation]
    average_days_held: Optional[float]


class TradeEvaluator:
    """Evaluates trading signals against actual market data."""
    
    def __init__(
        self,
        max_days: Optional[int] = None,
        require_both_targets: bool = True,
        hold_through_stop_loss: bool = False
    ):
        """
        Initialize the trade evaluator.
        
        Args:
            max_days: Maximum days to hold a trade (None = until target/stop-loss or data ends)
            require_both_targets: If True, only evaluate signals with both target and stop-loss
            hold_through_stop_loss: If True, when stop-loss is hit, continue holding until recovery
                                   (price returns above entry for buy, below entry for sell)
        """
        self.max_days = max_days
        self.require_both_targets = require_both_targets
        self.hold_through_stop_loss = hold_through_stop_loss
    
    def evaluate_signals(
        self,
        signals: List[TradingSignal],
        data: pd.Series
    ) -> List[TradeEvaluation]:
        """
        Evaluate all trading signals against market data.
        
        Args:
            signals: List of trading signals to evaluate
            data: Historical price data
            
        Returns:
            List of trade evaluations
        """
        evaluations = []
        
        for signal in signals:
            evaluation = self._evaluate_single_trade(signal, data)
            if evaluation:
                evaluations.append(evaluation)
        
        return evaluations
    
    def _evaluate_single_trade(
        self,
        signal: TradingSignal,
        data: pd.Series
    ) -> Optional[TradeEvaluation]:
        """Evaluate a single trade signal."""
        # Check if signal has required targets
        if self.require_both_targets:
            if not signal.target_price or not signal.stop_loss:
                return TradeEvaluation(
                    signal=signal,
                    outcome=TradeOutcome.INVALID,
                    exit_price=None,
                    exit_timestamp=None,
                    gain_percentage=0.0,
                    days_held=None,
                    max_favorable_excursion=0.0,
                    max_adverse_excursion=0.0
                )
        
        # Find entry point in data
        entry_idx = self._find_timestamp_index(data, signal.timestamp)
        if entry_idx is None or entry_idx >= len(data) - 1:
            return None
        
        # Get data after entry point
        future_data = data.iloc[entry_idx + 1:]
        if len(future_data) == 0:
            return TradeEvaluation(
                signal=signal,
                outcome=TradeOutcome.NO_OUTCOME,
                exit_price=None,
                exit_timestamp=None,
                gain_percentage=0.0,
                days_held=None,
                max_favorable_excursion=0.0,
                max_adverse_excursion=0.0
            )
        
        # Determine target and stop-loss prices based on signal type
        if signal.signal_type == SignalType.BUY:
            target_price = signal.target_price
            stop_loss_price = signal.stop_loss
            check_target = lambda price: price >= target_price if target_price else False
            check_stop = lambda price: price <= stop_loss_price if stop_loss_price else False
        else:  # SELL
            target_price = signal.target_price
            stop_loss_price = signal.stop_loss
            check_target = lambda price: price <= target_price if target_price else False
            check_stop = lambda price: price >= stop_loss_price if stop_loss_price else False
        
        # Track maximum favorable and adverse excursions
        entry_price = signal.price
        max_favorable = 0.0
        max_adverse = 0.0
        
        # Check each day after entry
        exit_idx = None
        exit_price = None
        outcome = TradeOutcome.NO_OUTCOME
        stop_loss_hit = False
        stop_loss_hit_idx = None
        recovery_price = None  # Price when recovery happens (back to entry or better)
        
        for i, (timestamp, price) in enumerate(future_data.items()):
            # Check max days limit
            if self.max_days and i >= self.max_days:
                exit_idx = entry_idx + 1 + i
                exit_price = price
                outcome = TradeOutcome.NO_OUTCOME
                break
            
            # Calculate current gain/loss percentage
            if signal.signal_type == SignalType.BUY:
                current_gain = ((price - entry_price) / entry_price) * 100
            else:  # SELL
                current_gain = ((entry_price - price) / entry_price) * 100
            
            # Update max favorable/adverse excursions
            if current_gain > max_favorable:
                max_favorable = current_gain
            if current_gain < max_adverse:
                max_adverse = current_gain
            
            # Check if target hit
            if target_price and check_target(price):
                exit_idx = entry_idx + 1 + i
                exit_price = target_price  # Use target price, not actual price
                if stop_loss_hit:
                    outcome = TradeOutcome.TARGET_HIT  # Recovered and hit target
                else:
                    outcome = TradeOutcome.TARGET_HIT
                break
            
            # Check if stop-loss hit
            if stop_loss_price and check_stop(price):
                if self.hold_through_stop_loss:
                    # Mark that stop-loss was hit, but continue holding
                    if not stop_loss_hit:
                        stop_loss_hit = True
                        stop_loss_hit_idx = entry_idx + 1 + i
                        # Continue holding - don't exit yet
                else:
                    # Normal behavior: exit at stop-loss
                    exit_idx = entry_idx + 1 + i
                    exit_price = stop_loss_price  # Use stop-loss price
                    outcome = TradeOutcome.STOP_LOSS_HIT
                    break
            
            # If holding through stop-loss, check for recovery
            if self.hold_through_stop_loss and stop_loss_hit:
                # Recovery means price returns to entry level or better
                if signal.signal_type == SignalType.BUY:
                    if price >= entry_price:
                        recovery_price = price
                        # Continue holding to see if target is reached
                else:  # SELL
                    if price <= entry_price:
                        recovery_price = price
                        # Continue holding to see if target is reached
        
        # Calculate final gain percentage
        if exit_price:
            if signal.signal_type == SignalType.BUY:
                gain_percentage = ((exit_price - entry_price) / entry_price) * 100
            else:  # SELL
                gain_percentage = ((entry_price - exit_price) / entry_price) * 100
        else:
            # No exit, use final price
            final_price = future_data.iloc[-1]
            if signal.signal_type == SignalType.BUY:
                gain_percentage = ((final_price - entry_price) / entry_price) * 100
            else:  # SELL
                gain_percentage = ((entry_price - final_price) / entry_price) * 100
        
        # If we held through stop-loss but didn't hit target, mark outcome appropriately
        if self.hold_through_stop_loss and stop_loss_hit and outcome != TradeOutcome.TARGET_HIT:
            if exit_price:
                # We exited at some point (max days or data end)
                if exit_price >= entry_price if signal.signal_type == SignalType.BUY else exit_price <= entry_price:
                    outcome = TradeOutcome.NO_OUTCOME  # Recovered but didn't hit target
                else:
                    outcome = TradeOutcome.STOP_LOSS_HIT  # Still below/above entry at exit
            else:
                # No exit yet, check final price
                final_price = future_data.iloc[-1]
                if final_price >= entry_price if signal.signal_type == SignalType.BUY else final_price <= entry_price:
                    outcome = TradeOutcome.NO_OUTCOME  # Recovered but didn't hit target
                else:
                    outcome = TradeOutcome.STOP_LOSS_HIT  # Still below/above entry
        
        # Calculate days held
        if exit_idx is not None:
            exit_timestamp = data.index[exit_idx]
            days_held = (exit_timestamp - signal.timestamp).days
        else:
            exit_timestamp = future_data.index[-1]
            days_held = (exit_timestamp - signal.timestamp).days
        
        return TradeEvaluation(
            signal=signal,
            outcome=outcome,
            exit_price=exit_price,
            exit_timestamp=exit_timestamp,
            gain_percentage=gain_percentage,
            days_held=days_held,
            max_favorable_excursion=max_favorable,
            max_adverse_excursion=max_adverse
        )
    
    def _find_timestamp_index(
        self,
        data: pd.Series,
        timestamp: pd.Timestamp
    ) -> Optional[int]:
        """Find the index of a timestamp in the data."""
        try:
            # Try exact match first
            if timestamp in data.index:
                return data.index.get_loc(timestamp)
            
            # Find nearest timestamp
            idx = data.index.get_indexer([timestamp], method='nearest')[0]
            if idx >= 0:
                return idx
        except (KeyError, IndexError):
            pass
        
        return None
    
    def summarize_evaluations(
        self,
        evaluations: List[TradeEvaluation]
    ) -> EvaluationSummary:
        """
        Generate summary statistics from trade evaluations.
        
        Args:
            evaluations: List of trade evaluations
            
        Returns:
            Evaluation summary with statistics
        """
        if not evaluations:
            return EvaluationSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                no_outcome_trades=0,
                win_rate=0.0,
                average_gain=0.0,
                average_loss=0.0,
                total_gain=0.0,
                best_trade=None,
                worst_trade=None,
                average_days_held=None
            )
        
        winning = [e for e in evaluations if e.outcome == TradeOutcome.TARGET_HIT]
        losing = [e for e in evaluations if e.outcome == TradeOutcome.STOP_LOSS_HIT]
        no_outcome = [e for e in evaluations if e.outcome == TradeOutcome.NO_OUTCOME]
        
        total_trades = len(evaluations)
        winning_trades = len(winning)
        losing_trades = len(losing)
        no_outcome_trades = len(no_outcome)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        average_gain = sum(e.gain_percentage for e in winning) / len(winning) if winning else 0.0
        average_loss = sum(e.gain_percentage for e in losing) / len(losing) if losing else 0.0
        total_gain = sum(e.gain_percentage for e in evaluations)
        
        best_trade = max(evaluations, key=lambda e: e.gain_percentage) if evaluations else None
        worst_trade = min(evaluations, key=lambda e: e.gain_percentage) if evaluations else None
        
        trades_with_days = [e for e in evaluations if e.days_held is not None]
        average_days_held = (
            sum(e.days_held for e in trades_with_days) / len(trades_with_days)
            if trades_with_days else None
        )
        
        return EvaluationSummary(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            no_outcome_trades=no_outcome_trades,
            win_rate=win_rate,
            average_gain=average_gain,
            average_loss=average_loss,
            total_gain=total_gain,
            best_trade=best_trade,
            worst_trade=worst_trade,
            average_days_held=average_days_held
        )
