"""
Calculates target prices and stop-loss levels for trading signals.

Uses Fibonacci retracements and Elliott Wave projections to determine
realistic price targets based on wave patterns.
"""
import pandas as pd
from typing import List, Optional
import sys
from pathlib import Path
from dataclasses import replace

# Use absolute import for direct script execution
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from signal_detector import TradingSignal, SignalType

# Import Wave for type hints and runtime usage
djia_dir = Path('/app/djia')
if not djia_dir.exists():
    djia_dir = current_dir.parent / 'djia'
sys.path.insert(0, str(djia_dir))
from elliott_wave_detector import Wave


class TargetCalculator:
    """Calculates target prices for trading signals."""
    
    def __init__(
        self,
        risk_reward_ratio: float = 2.0,
        use_fibonacci: bool = True
    ):
        """
        Initialize the target calculator.
        
        Args:
            risk_reward_ratio: Desired risk/reward ratio (default: 2.0 = 2:1)
            use_fibonacci: Whether to use Fibonacci levels for targets
        """
        self.risk_reward_ratio = risk_reward_ratio
        self.use_fibonacci = use_fibonacci
    
    def calculate_targets(
        self,
        signals: List[TradingSignal],
        data: pd.Series
    ) -> List[TradingSignal]:
        """
        Calculate target prices and stop-loss for trading signals.
        
        Args:
            signals: List of trading signals
            data: Time series data for context
            
        Returns:
            Signals with calculated targets and stop-loss levels
        """
        signals_with_targets = []
        
        for signal in signals:
            signal_with_targets = self._calculate_signal_targets(signal, data)
            signals_with_targets.append(signal_with_targets)
        
        return signals_with_targets
    
    def _calculate_signal_targets(
        self,
        signal: TradingSignal,
        data: pd.Series
    ) -> TradingSignal:
        """Calculate targets for a single signal."""
        wave = signal.wave
        entry_price = signal.price
        
        if signal.signal_type == SignalType.BUY:
            # For buy signals, target is typically end of next impulse wave
            target_price = self._calculate_buy_target(wave, data, entry_price)
            stop_loss = self._calculate_stop_loss(entry_price, target_price, SignalType.BUY)
        else:  # SELL
            # For sell signals, target is typically end of next corrective wave
            target_price = self._calculate_sell_target(wave, data, entry_price)
            stop_loss = self._calculate_stop_loss(entry_price, target_price, SignalType.SELL)
        
        reasoning = signal.reasoning
        if target_price:
            price_change = abs(target_price - entry_price) / entry_price * 100
            reasoning += f" | Target: {target_price:.2f} ({price_change:+.1f}%)"
        if stop_loss:
            risk = abs(stop_loss - entry_price) / entry_price * 100
            reasoning += f" | Stop: {stop_loss:.2f} ({risk:+.1f}%)"
        
        return replace(
            signal,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasoning
        )
    
    def _calculate_buy_target(
        self,
        wave: 'Wave',
        data: pd.Series,
        entry_price: float
    ) -> Optional[float]:
        """Calculate target price for buy signal."""
        # Find the next impulse wave (wave 3 or wave 5)
        # Use Fibonacci extensions: 1.618, 2.0, 2.618 of wave 1
        
        # Get wave 1 size if available
        wave_size = abs(wave.end_price - wave.start_price)
        
        if self.use_fibonacci:
            # Fibonacci extension levels
            fib_levels = [1.618, 2.0, 2.618]
            # Use conservative 1.618 extension
            target = entry_price + (wave_size * 1.618)
        else:
            # Simple approach: target is 1.5x the correction size
            target = entry_price + (wave_size * 1.5)
        
        return target
    
    def _calculate_sell_target(
        self,
        wave: 'Wave',
        data: pd.Series,
        entry_price: float
    ) -> Optional[float]:
        """Calculate target price for sell signal."""
        # For sell signals, target is typically a correction
        # Use Fibonacci retracement: 38.2%, 50%, 61.8% of the impulse
        
        wave_size = abs(wave.end_price - wave.start_price)
        
        if self.use_fibonacci:
            # Fibonacci retracement levels (conservative: 50%)
            target = entry_price - (wave_size * 0.5)
        else:
            # Simple approach: target is 0.382 of the wave size
            target = entry_price - (wave_size * 0.382)
        
        return target
    
    def _calculate_stop_loss(
        self,
        entry_price: float,
        target_price: Optional[float],
        signal_type: SignalType
    ) -> Optional[float]:
        """Calculate stop-loss based on risk/reward ratio."""
        if not target_price:
            return None
        
        # Calculate potential profit
        if signal_type == SignalType.BUY:
            profit = target_price - entry_price
            # Stop loss should be at risk_reward_ratio distance below entry
            stop_loss = entry_price - (profit / self.risk_reward_ratio)
        else:  # SELL
            profit = entry_price - target_price
            # Stop loss should be at risk_reward_ratio distance above entry
            stop_loss = entry_price + (profit / self.risk_reward_ratio)
        
        return stop_loss
