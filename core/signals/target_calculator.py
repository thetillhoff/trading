"""
Calculates target prices and stop-loss levels for trading signals.

Uses Fibonacci retracements and Elliott Wave projections to determine
realistic price targets based on wave patterns.
"""
import pandas as pd
from typing import List, Optional
from dataclasses import replace

from ..shared.types import TradingSignal, SignalType
from ..indicators.elliott_wave import Wave


class TargetCalculator:
    """Calculates target prices for trading signals."""
    
    def __init__(
        self,
        risk_reward_ratio: float = 2.0,
        use_fibonacci: bool = True,
        use_atr_stops: bool = False,
        atr_stop_multiplier: float = 2.0,
        atr_period: int = 14
    ):
        """
        Initialize the target calculator.
        
        Args:
            risk_reward_ratio: Desired risk/reward ratio (default: 2.0 = 2:1)
            use_fibonacci: Whether to use Fibonacci levels for targets
            use_atr_stops: Whether to use ATR-based stops instead of fixed percentage
            atr_stop_multiplier: Multiplier for ATR stop loss (default: 2.0)
            atr_period: Period for ATR calculation (default: 14)
        """
        self.risk_reward_ratio = risk_reward_ratio
        self.use_fibonacci = use_fibonacci
        self.use_atr_stops = use_atr_stops
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_period = atr_period
    
    def calculate_targets(
        self,
        signals: List[TradingSignal],
        data: pd.Series,
        all_waves: Optional[List[Wave]] = None
    ) -> List[TradingSignal]:
        """
        Calculate target prices and stop-loss for trading signals.
        
        Args:
            signals: List of trading signals
            data: Time series data for context
            all_waves: Optional list of all detected waves for wave relationship analysis
            
        Returns:
            Signals with calculated targets and stop-loss levels
        """
        signals_with_targets = []
        
        for signal in signals:
            signal_with_targets = self._calculate_signal_targets(signal, data, all_waves)
            signals_with_targets.append(signal_with_targets)
        
        return signals_with_targets
    
    def _calculate_signal_targets(
        self,
        signal: TradingSignal,
        data: pd.Series,
        all_waves: Optional[List[Wave]] = None
    ) -> TradingSignal:
        """Calculate targets for a single signal."""
        entry_price = signal.price

        # Check if signal has wave info (Elliott Wave signals have .wave attribute)
        wave = getattr(signal, 'wave', None)

        is_buy = signal.signal_type == SignalType.BUY
        
        if wave is not None:
            # Use wave-based target calculation
            if is_buy:
                target_price = self._calculate_buy_target(wave, data, entry_price, all_waves)
            else:  # SELL
                target_price = self._calculate_sell_target(wave, data, entry_price, all_waves)
            
            # Calculate stop loss (ATR-based if enabled, otherwise risk/reward based)
            if self.use_atr_stops:
                stop_loss = self._calculate_atr_stop_loss(entry_price, data, signal.signal_type)
            else:
                stop_loss = self._calculate_stop_loss(entry_price, target_price, signal.signal_type)
        else:
            # No wave info (indicator-only signal) - use ATR-based or percentage-based targets
            target_price, stop_loss = self._calculate_atr_based_targets(
                signal.signal_type, data, entry_price
            )


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
    
    def _calculate_atr_based_targets(
        self,
        signal_type: SignalType,
        data: pd.Series,
        entry_price: float,
    ) -> tuple:
        """
        Calculate targets using ATR (Average True Range) for signals without wave info.
        
        Falls back to percentage-based targets if insufficient data for ATR.
        """
        # Calculate ATR (14-period)
        if len(data) >= 14:
            # Simple ATR approximation using price range
            high_low_range = data.rolling(14).apply(lambda x: x.max() - x.min()).iloc[-1]
            atr = high_low_range / 14 if not pd.isna(high_low_range) else None
        else:
            atr = None
        
        is_buy = signal_type == SignalType.BUY
        
        if atr and atr > 0:
            # Use 2x ATR for target, 1x ATR for stop (2:1 risk/reward)
            if is_buy:
                target_price = entry_price + (atr * self.risk_reward_ratio)
                stop_loss = entry_price - atr
            else:  # SELL
                target_price = entry_price - (atr * self.risk_reward_ratio)
                stop_loss = entry_price + atr
        else:
            # Fallback: use 4% target, 2% stop (2:1 risk/reward)
            if is_buy:
                target_price = entry_price * (1 + 0.02 * self.risk_reward_ratio)
                stop_loss = entry_price * 0.98
            else:  # SELL
                target_price = entry_price * (1 - 0.02 * self.risk_reward_ratio)
                stop_loss = entry_price * 1.02
        
        return target_price, stop_loss
    
    def _calculate_buy_target(
        self,
        wave: 'Wave',
        data: pd.Series,
        entry_price: float,
        all_waves: Optional[List[Wave]] = None
    ) -> Optional[float]:
        """
        Calculate target price for buy signal using wave-specific Fibonacci relationships.
        
        Wave-specific targets:
        - Wave 3: 1.618-2.618 × Wave 1
        - Wave 5: = Wave 1 or 0.618 × Wave 3
        - Wave C: = Wave A or 1.618 × Wave A
        """
        if not self.use_fibonacci:
            # Simple approach: target is 1.5x the correction size
            wave_size = abs(wave.end_price - wave.start_price)
            return entry_price + (wave_size * 1.5)
        
        # Find related waves for wave-specific targets
        related_waves = self._find_related_waves(wave, all_waves)
        
        if wave.label.value == "2":
            # Wave 2 entry: Target is Wave 3 = 1.618-2.618 × Wave 1
            wave1 = related_waves.get('wave1')
            if wave1:
                wave1_size = abs(wave1.end_price - wave1.start_price)
                # Use conservative 1.618 extension
                target = entry_price + (wave1_size * 1.618)
                return target
        
        elif wave.label.value == "4":
            # Wave 4 entry: Target is Wave 5 = Wave 1 or 0.618 × Wave 3
            wave1 = related_waves.get('wave1')
            wave3 = related_waves.get('wave3')
            
            if wave3:
                # Wave 5 = 0.618 × Wave 3
                wave3_size = abs(wave3.end_price - wave3.start_price)
                target = entry_price + (wave3_size * 0.618)
                return target
            elif wave1:
                # Wave 5 = Wave 1
                wave1_size = abs(wave1.end_price - wave1.start_price)
                target = entry_price + wave1_size
                return target
        
        elif wave.label.value == "b" and wave.wave_type.value == "corrective":
            # Wave B entry: Target is Wave C = Wave A or 1.618 × Wave A
            wave_a = related_waves.get('wave_a')
            if wave_a:
                wave_a_size = abs(wave_a.end_price - wave_a.start_price)
                # Use 1.618 extension for Wave C
                target = entry_price + (wave_a_size * 1.618)
                return target
        
        # Fallback to original method if wave-specific relationships not found
        wave_size = abs(wave.end_price - wave.start_price)
        return entry_price + (wave_size * 1.618)
    
    def _find_related_waves(
        self,
        wave: 'Wave',
        all_waves: Optional[List[Wave]]
    ) -> dict:
        """
        Find related waves for wave-specific target calculation.
        
        Returns dict with keys: 'wave1', 'wave3', 'wave_a', etc.
        """
        related = {}
        
        if not all_waves:
            return related
        
        # Find waves in the same pattern
        wave_idx = None
        for i, w in enumerate(all_waves):
            if w.start_idx == wave.start_idx and w.end_idx == wave.end_idx:
                wave_idx = i
                break
        
        if wave_idx is None:
            return related
        
        # Find Wave 1 (for Wave 2 and Wave 4 entries)
        if wave.label.value in ["2", "4"]:
            for w in all_waves:
                if w.label.value == "1" and w.wave_type == wave.wave_type:
                    # Check if Wave 1 is before current wave
                    if w.end_idx < wave.start_idx:
                        related['wave1'] = w
                        break
        
        # Find Wave 3 (for Wave 4 entries)
        if wave.label.value == "4":
            for w in all_waves:
                if w.label.value == "3" and w.wave_type == wave.wave_type:
                    # Check if Wave 3 is between Wave 1 and Wave 4
                    if related.get('wave1') and related['wave1'].end_idx < w.start_idx < wave.start_idx:
                        related['wave3'] = w
                        break
        
        # Find Wave A (for Wave B entries)
        if wave.label.value == "b" and wave.wave_type.value == "corrective":
            for w in all_waves:
                if w.label.value == "a" and w.wave_type == wave.wave_type:
                    # Check if Wave A is before Wave B
                    if w.end_idx < wave.start_idx:
                        related['wave_a'] = w
                        break
        
        return related
    
    def _calculate_sell_target(
        self,
        wave: 'Wave',
        data: pd.Series,
        entry_price: float,
        all_waves: Optional[List[Wave]] = None
    ) -> Optional[float]:
        """
        Calculate target price for sell signal using wave-specific relationships.
        
        For Wave 5 exits: Target is 38.2-61.8% retracement of entire 5-wave move
        """
        wave_size = abs(wave.end_price - wave.start_price)
        
        if not self.use_fibonacci:
            # Simple approach: target is 0.382 of the wave size
            return entry_price - (wave_size * 0.382)
        
        # For Wave 5 exits, calculate retracement of entire 5-wave move
        if wave.label.value == "5":
            related_waves = self._find_related_waves(wave, all_waves)
            wave1 = related_waves.get('wave1')
            
            if wave1:
                # Calculate entire 5-wave move from Wave 1 start to Wave 5 end
                total_move = abs(entry_price - wave1.start_price)
                # Use 50% retracement (middle of 38.2-61.8% range)
                target = entry_price - (total_move * 0.5)
                return target
        
        # Fallback: Fibonacci retracement levels (conservative: 50%)
        return entry_price - (wave_size * 0.5)
    
    def _calculate_atr(self, data: pd.Series) -> Optional[float]:
        """
        Calculate ATR (Average True Range) from price data.
        
        Args:
            data: Price series
            
        Returns:
            ATR value, or None if insufficient data
        """
        if len(data) < self.atr_period:
            return None
        
        # Simplified ATR using price range
        # (True ATR requires High/Low/Close, we only have Close)
        high_low_range = data.rolling(self.atr_period).apply(lambda x: x.max() - x.min()).iloc[-1]
        
        if pd.isna(high_low_range) or high_low_range == 0:
            return None
        
        atr = high_low_range / self.atr_period
        return atr
    
    def _calculate_atr_stop_loss(
        self,
        entry_price: float,
        data: pd.Series,
        signal_type: SignalType
    ) -> Optional[float]:
        """
        Calculate stop-loss based on ATR (Average True Range).
        
        Args:
            entry_price: Entry price for the trade
            data: Price series for ATR calculation
            signal_type: BUY or SELL
            
        Returns:
            Stop loss price, or None if ATR cannot be calculated
        """
        atr = self._calculate_atr(data)
        
        if not atr or atr == 0:
            # Fallback to fixed percentage if ATR unavailable
            if signal_type == SignalType.BUY:
                return entry_price * 0.98  # 2% stop
            else:
                return entry_price * 1.02  # 2% stop
        
        # Calculate stop loss using ATR multiplier
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - (atr * self.atr_stop_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr * self.atr_stop_multiplier)
        
        return stop_loss
    
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
